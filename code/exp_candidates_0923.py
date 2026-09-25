#!/usr/bin/env python3
"""0923 candidate screen on the TEST-FREE panels only.  2026-09-23 (T1).

Reads nothing at or after 2026-01-01 (asserted by `lib.guard`).

Why
---
On the 2026 test panel the gate-free re-specification (`respec`, 0.7852) is
beaten by the 72-member fixed-grid kNN-LM (0.7880) and by validation-selected
AdaNPC-fused (0.7855); on the 9-fold walk-forward the matched kNN-LM (+0.0042),
`single_joint` (+0.0070) and `local_linear` (+0.0026) beat it.  This screen
asks whether a principled, still closed-form, zero-backward-pass change to the
R3 fast state wins honestly -- selected on the walk-forward, never on test.

Structural differences between respec and kNN-LM-72 (read off the code):
  respec   : shrunk local rate q_k = (S_k + lam*pi0)/(W_k + lam), lam in {2,8};
             k averaged INSIDE a member in logit space; 12 members
             (space x T x lam) averaged in probability space; beta = 0.5.
  kNN-LM-72: UNSHRUNK vote S_k/W_k clipped at 1e-6; k is a MEMBER axis;
             72 members (space x k x T x beta{.3,.5,.7}) averaged in
             probability space.  (Its beta=0.5 24-member twin scores the same on
             test, 0.7881 vs 0.7880, so the beta axis is not the reason.)

Candidates (declared here, before either panel is scored)
---------------------------------------------------------
Round 1 (hypotheses from the T1 brief):
  c01_kmem       (a/b) k promoted to a probability-space member axis; 48 members
  c02_wide72     (a)   c01 with lam in {0.5,2,8}; 72 members = kNN-LM-72 arity
  c03_ll_kmem    (c)   local-linear fast state, k as members; 48 members
  c04_ll_mix     (c)   respec members + local-linear members; 24 members
  c05_evid_logit (d)   logit fusion with beta_b = n_eff/(n_eff + k) (Kish
                       effective neighbour count; = 1/2 for a flat kernel)
  c06_evid_prob  (d)   Beta-binomial posterior with the FROZEN model as prior,
                       pseudo-count k: p = (S_k + k p_e)/(W_k + k), i.e. a
                       probability mixture with data-dependent weight W/(W+k)
  c07_joint2     (e)   respec with the joint space double-weighted (1/4,1/4,1/2)
  c08_postpred   (b')  posterior-predictive: E over theta~Beta(S+lam pi0,
                       F+lam(1-pi0)) of sigmoid(l_e/2 + logit(theta)/2), fixed
                       Gauss-Legendre nodes, k as members
Round 2 (<= 4, declared in exp_candidates_0923.py ROUND2 after round 1 is read,
         combinations of round-1 components only).

Selection rule (fixed here, applied once after round 2, on the walk-forward)
---------------------------------------------------------------------------
  final = the candidate with the highest pooled 9-fold walk-forward bar AUC,
  subject to
    R1  walk-forward delta vs respec > 0 with bootstrap P(delta<=0) < 0.05;
    R2  2025-Q4 validation delta vs respec >= -0.002;
    R3  one definition for every backbone and market (no per-backbone knob).
  Candidates within 0.0005 walk-forward AUC of the best are tie-broken toward
  fewer components.  If no candidate satisfies R1, final = respec.
  Beating kNN-LM-72 is NOT part of the rule -- it is the reported outcome.

Selection rule v2 (coordinator steer from the user, received 2026-09-23 AFTER
the round 1-3 pooled walk-forward and validation tables had been read, BEFORE
any protocol was written and before round 4 was scored; this ordering is
disclosed in protocol_0923.json, together with what the v1 rule above selects):
  eligible  (a) pooled WF bar AUC >= respec AND >= kNN-LM-72 fixed grid;
            (c) 2025-Q4 validation delta vs frozen >= 0;
            (R3) one definition for every backbone / market.
  ranking   (b1) fewest walk-forward folds with bar-AUC delta vs frozen < 0
                 (target 0/9);
            (b2) then the largest worst-fold delta vs frozen;
            (b3) then the largest pooled WF bar AUC;
            (b4) then fewer components.
  Worst case is ranked before the mean.  If no candidate is eligible, the
  final is the eligible-free best on (b1,b2,b3) and that is reported.
Round 4 (coordinator ideas, declared here before scoring):
  c13_superset    equal-weight average of four FAMILY averages: kNN-LM-72 fixed
                  grid, local-linear (12), c02_wide72, c10_evid_prob_joint2
  c14_selfcal     c10 with the anchor strength c chosen per deployment by
                  memory leave-one-bar-out (+-1 day purge, out-of-fold frozen
                  anchor on the memory bars), c in {1/4,1/2,1,2,4}, ties -> c
                  closest to 1.  Test-free, per backbone.
  c15_diagmetric  c10 with each routing coordinate scaled by |2 AUC_j - 1|
                  against the memory labels (T2 idea 3), sum w^2 = d
  c16_samworth    c10 with Samworth (2012) rank weights at k*=k replacing the
                  exponential kernel (T2 idea 1); nearest weight 1; no T axis

    python code/exp_candidates_0923.py --panel wf|val [--round 1..4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import exp_lib_0923 as lib                            # noqa: E402

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import common_protocol_clean as cpc                   # noqa: E402
import exp_ablation_sensitivity as eas                # noqa: E402

log = lib.log
REFS = ("frozen", "shipped72", "respec", "knnlm", "knnlm72", "knnlm72_prob",
        "knnlm24_b05", "adanpc_fused", "local_linear")
ROUND1 = ("c01_kmem", "c02_wide72", "c03_ll_kmem", "c04_ll_mix",
          "c05_evid_logit", "c06_evid_prob", "c07_joint2", "c08_postpred")
# Round 2, declared 2026-09-23 after reading round-1 walk-forward POINT AUCs only
# (fold means; c06_evid_prob was the best round-1 candidate).  Three
# combinations of round-1 parts, no new component:
#   c09_evid_prob_wide   c06 x (c05/c02 breadth): anchor pseudo-count c*k,
#                        c in {1/2,1,2}; at a flat kernel alpha = 2/3,1/2,1/3,
#                        the Bayesian counterpart of kNN-LM's beta grid; 72 members
#   c10_evid_prob_joint2 c06 x c07: joint space double-weighted
#   c11_evid_prob_neff   c06 x c05: Kish effective count n_eff replaces W
#   c12_evid_prob_lepski  added 2026-09-23 from T2's literature list (idea 2:
#                        Lepski & Spokoiny 1997; Balsubramani et al. 2019),
#                        declared BEFORE any round-2 number was read: per query
#                        keep the largest k in {8,16,32,64} whose vote
#                        h_k=(S_k+.5)/(W_k+1) lies inside h_k' +- z*sqrt(h_k'(1-h_k')
#                        /n_eff_k') for every k'<k, z=2 fixed a priori; then the
#                        c06 posterior at that k.  6 members (space x T).
# Total screened = 12 candidates (the budget).  c12 is scored as round "3".
ROUND2: tuple[str, ...] = ("c09_evid_prob_wide", "c10_evid_prob_joint2",
                           "c11_evid_prob_neff")
ROUND3: tuple[str, ...] = ("c12_evid_prob_lepski",)
ROUND4: tuple[str, ...] = ("c13_superset", "c14_selfcal", "c15_diagmetric",
                           "c16_samworth")
R4_BASE = {"knnlm72", "local_linear", "c02_wide72", "c10_evid_prob_joint2"}


def mem_anchor_oof(fold, folds=3):
    """Out-of-fold frozen GBT probability for every memory bar (fit window only)."""
    import exp_residual_20260921 as base
    oof = base.oof_frozen_bar(fold.fit, folds=folds)
    m = fold.memory.frame[["datetime"]].merge(oof, on="datetime", how="left")["p_oof"]
    m = m.to_numpy(float)
    return np.where(np.isfinite(m), m, np.nanmean(m))


def predict(fold, which, rnd, info_sink):
    if rnd == 4:
        P = lib.all_arms(fold, which=(which - set(ROUND4)) | R4_BASE)
        nm = P.pop("_n_members")
        anchor = mem_anchor_oof(fold) if "c14_selfcal" in which else None
        extra, info = lib.round4_arms(fold, which=which, mem_anchor=anchor,
                                      base_preds=P)
        P.update(extra)
        info_sink.append(info)
        nm.update({"c13_superset": 4, "c14_selfcal": 24, "c15_diagmetric": 24,
                   "c16_samworth": 12})
        P["_n_members"] = nm
        return P
    return lib.all_arms(fold, which=which)
FOLD_START = pd.Timestamp("2025-04-01")
RND = 1
INFO: list = []


def fold_months():
    edges = pd.date_range(FOLD_START, cpc.TEST_START, freq="MS")
    return list(zip(edges[:-1], edges[1:]))


def run_wf(panel, which, tag):
    pooled, y_rows, times, per_fold = {}, [], [], []
    for lo, hi in fold_months():
        fit = panel[panel["datetime"] < lo]
        stream = panel[(panel["datetime"] >= lo) & (panel["datetime"] < hi)]
        fold = lib.Fold(fit, stream, lo)
        preds = predict(fold, which, RND, INFO)
        nm = preds.pop("_n_members")
        rec = {"fold": str(lo.date()), "n_rows": fold.bi.n_rows,
               "n_bars": fold.bi.n_bars}
        for a, p in preds.items():
            pooled.setdefault(a, []).append(p)
            rec[f"auc_bar_{a}"] = fold.bi.auc_bar(p) if hasattr(fold.bi, "auc_bar") \
                else lib.auc(fold.bi.y_bar, fold.bi.pool(p))
        per_fold.append(rec)
        y_rows.append(fold.bi.y_row)
        times.append(fold.stream["datetime"].to_numpy())
        log(f"fold {lo.date()}: {fold.bi.n_bars} bars | respec "
            f"{rec['auc_bar_respec']:.4f} knnlm72 {rec['auc_bar_knnlm72']:.4f}")
    pf = pd.DataFrame(per_fold)
    pf.to_csv(lib.OUT / f"walkforward_per_fold_{tag}.csv", index=False)
    stream = pd.DataFrame({"datetime": np.concatenate(times),
                           "target_hi_vol": np.concatenate(y_rows)})
    bi = eas.BarIndex(stream)
    stacked = {a: np.concatenate(v) for a, v in pooled.items()}
    np.savez_compressed(lib.OUT / f"predictions_walkforward_{tag}.npz",
                        y_row=bi.y_row, **stacked)
    tab = lib.score_table(bi, stacked,
                          refs=("frozen", "respec", "knnlm72", "adanpc_fused",
                                "knnlm", "local_linear"))
    # fold sign tests
    for ref in ("respec", "knnlm72", "adanpc_fused"):
        w = {a: int((pf[f"auc_bar_{a}"] > pf[f"auc_bar_{ref}"]).sum())
             for a in stacked}
        tab[f"folds_beating_{ref}"] = tab["arm"].map(w)
        tab[f"sign_p_vs_{ref}"] = tab["arm"].map(
            lambda a: lib.sign_test(w[a], len(pf)))
    tab["n_members"] = tab["arm"].map(nm)
    tab.to_csv(lib.OUT / f"named_walkforward_{tag}.csv", index=False)
    json.dump({"n_folds": len(pf), "n_bars": bi.n_bars, "n_rows": bi.n_rows,
               "fold_start": str(FOLD_START), "fold_end": str(cpc.TEST_START)},
              open(lib.OUT / f"walkforward_meta_{tag}.json", "w"), indent=2)
    return tab


def run_val(panel, which, tag):
    fit = panel[panel["datetime"] < cpc.TRAIN_END]
    stream = panel[(panel["datetime"] >= cpc.TRAIN_END)
                   & (panel["datetime"] < cpc.TEST_START)]
    fold = lib.Fold(fit, stream, cpc.TRAIN_END)
    preds = predict(fold, which, RND, INFO)
    nm = preds.pop("_n_members")
    np.savez_compressed(lib.OUT / f"predictions_validation_{tag}.npz",
                        y_row=fold.bi.y_row, **preds)
    tab = lib.score_table(fold.bi, preds,
                          refs=("frozen", "respec", "knnlm72", "adanpc_fused",
                                "knnlm", "local_linear"))
    tab["n_members"] = tab["arm"].map(nm)
    tab.to_csv(lib.OUT / f"named_validation_{tag}.csv", index=False)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=["wf", "val"])
    ap.add_argument("--round", type=int, default=1, choices=[1, 2, 3, 4])
    args = ap.parse_args()
    panel = lib.load_panel(allow_test=False)
    global RND
    RND = args.round
    cands = {1: ROUND1, 2: ROUND2, 3: ROUND3, 4: ROUND4}[args.round]
    which = set(REFS) | set(cands) | ({"c06_evid_prob"} if args.round >= 2 else set())
    tag = f"r{args.round}"
    tab = (run_wf if args.panel == "wf" else run_val)(panel, which, tag)
    if INFO:
        json.dump(INFO, open(lib.OUT / f"round4_info_{args.panel}.json", "w"),
                  indent=1, default=float)
    cols = ["arm", "n_members", "auc_bar", "brier_bar", "vs_respec_bar_delta",
            "vs_respec_bar_ci_lo", "vs_respec_bar_ci_hi", "vs_respec_bar_p_le_zero",
            "vs_knnlm72_bar_delta", "vs_knnlm72_bar_p_le_zero"]
    cols += [c for c in tab.columns if c.startswith("folds_beating")]
    log("\n" + tab[cols].to_string(index=False))


if __name__ == "__main__":
    main()
