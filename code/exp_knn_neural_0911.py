#!/usr/bin/env python
"""Retrieval baselines on the MATCHED NEURAL backbones (MLP+BN, FT-Transformer).

The coverage gap this closes
----------------------------
R3-TTT is compared against the gradient-free RETRIEVAL family (fixed-grid
k-NN-LM, selected k-NN-LM, Nadaraya-Watson, AdaNPC, prototype, routing-space
T3A) only on the clean GBT tree backbone (``results/new_baselines``).  The
matched neural table (``neural_main.csv`` -> ``tab_m_neural_0908.tex``) carries
no retrieval-fusion arm at all: on MLP+BN and FT-Transformer R3-TTT is compared
only against gradient TTA methods.  The two strongest R3 results therefore have
no neighbour-fusion comparator.

This track runs the SAME retrieval family, with the SAME configurations and the
SAME selection rule, on the two neural backbones, on the same clean panel, the
same 525 test bars and the same paired 5-day moving-block bootstrap.

Nothing is tuned per backbone for the fixed-grid arms: their configuration
budget is 0 on the tree and 0 here.  The validation-selected arms get exactly
the budget they get on the tree, selected on the backbone's OWN validation
anchor, at validation per-bar AUC, before the test panel is read.

Stages
------
    protocol   write protocol.json (arms, budgets, guard) -- FIRST
    run        score everything; write metrics.csv + paired.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import common_protocol_clean as cpc            # noqa: E402  NEVER common_protocol
import exp_new_baselines as eas                # noqa: E402

OUT = ROOT / "results" / "knn_neural_0911"
OUT.mkdir(parents=True, exist_ok=True)
eas._OUT = OUT

T0 = time.time()
SEED = 42
N_BOOT = 5000
BLOCK_DAYS = 5
TEST_END = pd.Timestamp("2026-04-11")
FORWARD_START = pd.Timestamp("2026-04-16")
BACKBONES = ("gbt", "mlp_bn", "ft_trans")


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def jdump(obj, name):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=eas._jsonable))
    log(f"  wrote {p.name}")
    return p


def save(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    log(f"  wrote {p.name}  ({len(df)} rows)")
    return p


# =============================================================================
# arm definitions
# =============================================================================
FIXED_GRID_ARMS = {
    "knnlm_fixedgrid_logit": dict(
        kind="logit", blends=cpc.BLENDS,
        note="fixed-grid k-NN-LM, logit-space interpolation, 72 members "
             "(3 spaces x k in {8,16,32,64} x T in {0.25,1.0} x lambda in "
             "{0.3,0.5,0.7}), uniform average, budget 0.  Identical to "
             "results/new_baselines arm `knnlm_sf72_logit`."),
    "knnlm_fixedgrid_prob": dict(
        kind="prob", blends=cpc.BLENDS,
        note="the same 72 members interpolated in PROBABILITY space.  "
             "Identical to results/new_baselines arm `knnlm_sf72`."),
    "knnlm_fixedgrid_logit_lam05": dict(
        kind="logit", blends=(0.5,),
        note="matched-arity twin of the DECLARED R3 config: lambda pinned at "
             "0.5, 24 members.  Budget 0.  Secondary."),
}

SELECTED_ARMS = ("knn_vote", "knnlm_sel_prob", "knnlm_sel_logit",
                 "nadaraya_watson", "prototype", "t3a_routing",
                 "adanpc_vote", "adanpc_fused")
RETRIEVAL_ARMS = tuple(FIXED_GRID_ARMS) + SELECTED_ARMS
R3_ARMS = ("r3ttt_declared", "r3ttt_sf72")

BUDGETS = {
    "knnlm_fixedgrid_logit": 0,
    "knnlm_fixedgrid_prob": 0,
    "knnlm_fixedgrid_logit_lam05": 0,
    "knn_vote": 27,
    "knnlm_sel_prob": 3402,
    "knnlm_sel_logit": 3402,
    "nadaraya_watson": 630,
    "prototype": 378,
    "t3a_routing": 378,
    "adanpc_vote": 216,
    "adanpc_fused": 4536,
    "r3ttt_declared": 0,
    "r3ttt_sf72": 0,
    "frozen": 0,
}

DECLARED_R3 = dict(betas=(0.5,), gates=(False,), lambdas=(2.0, 8.0),
                   temperatures=(0.25, 1.0), scales=(8, 16, 32, 64),
                   spaces=("timing", "market_state", "joint"), n_members=12)


# =============================================================================
# stage: protocol
# =============================================================================
def stage_protocol():
    proto = {
        "track": "knn_neural_0911",
        "date": time.strftime("%Y-%m-%d"),
        "question": ("Does R3-TTT's advantage over simple neighbour fusion "
                     "(fixed-grid k-NN-LM) hold on the two MATCHED NEURAL "
                     "backbones, where the published table has no retrieval "
                     "comparator at all?"),
        "what_this_is": ("ADDING COMPARATORS to an existing comparison, not "
                         "selecting a configuration.  No R3 variant is searched "
                         "here; the R3 arms are the two already-declared ones."),
        "panel": ("CLEAN -- crawl-instant engagement snapshot removed; identical "
                  "to results/new_baselines/protocol.json"),
        "code": {
            "protocol_module": "code/common_protocol_clean.py",
            "reused": "code/exp_new_baselines.py (Env, retrieval scorers, "
                      "BarIndex, paired_boot) and code/exp_gradient_tta_lib.py "
                      "(neural backbones)",
            "script": "code/exp_knn_neural_0911.py",
        },
        "forward_window_guard": {
            "TEST_END": str(TEST_END),
            "FORWARD_START": str(FORWARD_START),
            "rule": "no row at or after TEST_END is read; the max datetime read "
                    "is recorded in run_guard.json",
        },
        "split": {
            "train": "datetime < 2025-10-01",
            "validation": "2025-10-01 <= datetime < 2026-01-01",
            "test": "2026-01-01 <= datetime < 2026-04-11",
        },
        "backbones": {
            "gbt": "GradientBoostingClassifier, 5 seeds averaged (the tree "
                   "backbone; reproduces results/new_baselines exactly -- used "
                   "as the provenance gate)",
            "mlp_bn": "MLP+BatchNorm, quantile prep, width 64 depth 3, 10 "
                      "epochs, 5 seeds averaged -- the SAME recipe and the SAME "
                      "architecture selection as results/new_baselines "
                      "stage_neural; torch threads pinned to 8, which is what "
                      "makes it bit-reproducible",
            "ft_trans": "FT-Transformer, quantile prep, d_model 64, 2 layers, 4 "
                        "heads, 20 epochs, 5 seeds averaged -- same recipe",
        },
        "backbone_reproduction_gate": (
            "the refitted neural frozen test probabilities must match "
            "results/new_baselines/neural_test_predictions.csv columns "
            "`<kind>__frozen` to < 1e-6; otherwise the run aborts.  This is what "
            "makes the new rows placeable in the SAME table as the published "
            "neural rows."),
        "gbt_reproduction_gate": (
            "the recomputed GBT-backbone retrieval arms must match "
            "results/new_baselines/retrieval_test_predictions.npz to < 1e-9."),
        "arms": {
            "frozen": "the backbone's own frozen slow probability, no adaptation",
            "r3ttt_sf72": "R3-TTT-SF as published: uniform average over the "
                          "72-member grid; nothing selected",
            "r3ttt_declared": ("the configuration declared in "
                               "results/method_respec_0911/protocol.json from "
                               "VALIDATION on the tree, before that track's "
                               "single test read: beta=0.5, gate OFF, lambda in "
                               "{2,8}, T in {0.25,1}, k-average over {8,16,32,64}, "
                               "all three routing spaces = 12 members.  It is "
                               "transferred UNCHANGED to the neural backbones; "
                               "its per-backbone budget here is 0."),
            **{k: v["note"] for k, v in FIXED_GRID_ARMS.items()},
            "knn_vote": "plain k-NN vote, retrieval only (no anchor); space and "
                        "k selected on validation per-bar AUC",
            "knnlm_sel_prob": "k-NN-LM with space, k, kernel temperature and "
                              "lambda selected on validation per-bar AUC, "
                              "probability-space interpolation",
            "knnlm_sel_logit": "the same, logit-space interpolation",
            "nadaraya_watson": "Nadaraya-Watson over the full memory; bandwidth "
                               "and lambda selected on validation",
            "prototype": "two-centroid prototype; tau and lambda selected on "
                         "validation",
            "t3a_routing": "T3A ported to the routing space (pseudo-labelled "
                           "support prototypes); M and lambda selected on "
                           "validation.  ANCHOR-DEPENDENT: its pseudo-labels come "
                           "from the backbone's own frozen model, so it is "
                           "genuinely re-run per backbone.",
            "adanpc_vote": "AdaNPC, vote replaces the classifier; retrieval only",
            "adanpc_fused": "AdaNPC vote fused with the anchor; lambda selected "
                            "on validation",
        },
        "configuration_budgets": BUDGETS,
        "budget_note": (
            "identical to results/new_baselines/protocol.json hyperparameter_"
            "budgets.  The fixed-grid k-NN-LM arms and both R3 arms consult "
            "validation for NOTHING on any backbone."),
        "anchor_dependence": {
            "anchor_independent_bar_scores": ["knn_vote", "knnlm (all forms)",
                                              "nadaraya_watson", "prototype",
                                              "adanpc_vote", "adanpc_fused"],
            "note": ("the retrieval bar SCORE of these arms does not involve the "
                     "slow model, so it is computed once and shared across "
                     "backbones; the FUSED prediction and every selected lambda "
                     "are backbone-specific.  t3a_routing's bar score is "
                     "anchor-dependent and is recomputed per backbone."),
        },
        "estimand": {
            "primary": "aligned 15-minute outcome bar; bar score = mean of the "
                       "arm's per-post probabilities in the bar (525 test bars)",
            "secondary": "per post (1,835 test rows)",
        },
        "metrics": ["auc", "brier", "logloss"],
        "uncertainty": {
            "estimator": "paired 5-day moving-block bootstrap",
            "clustered_by": "calendar day",
            "n_boot": N_BOOT,
            "block_days": BLOCK_DAYS,
            "seed": SEED,
            "reported": "point delta, 95% percentile interval, bootstrap tail "
                        "P(delta <= 0)",
            "identical_to": "results/new_baselines and "
                            "results/matched_backbone_paired",
        },
        "contrast_set_declared_before_scoring": {
            "primary": "r3ttt_declared - knnlm_fixedgrid_logit, on each of the "
                       "three backbones, at bar and post level",
            "also": "r3ttt_sf72 - knnlm_fixedgrid_logit; both R3 arms against "
                    "every other retrieval arm and against frozen; and "
                    "r3ttt_declared - r3ttt_sf72",
        },
        "multiplicity": ("Holm over the per-backbone family of R3-minus-"
                         "retrieval-arm per-bar AUC contrasts, reported in "
                         "paired_holm.csv; the family increment is enumerated in "
                         "family_count.json."),
        "honesty_clause": ("If fixed-grid k-NN-LM beats R3-TTT on a neural "
                           "backbone that is reported plainly.  k-NN-LM is given "
                           "exactly the treatment it gets on the tree; no "
                           "per-backbone retuning in either direction."),
        "adaptations_declared": [
            "NONE to the k-NN-LM configuration.  The 72-member fixed grid is "
            "applied verbatim; only the anchor changes, which is the whole "
            "point of the experiment.",
            "torch.set_num_threads(8) is required to reproduce the published "
            "neural frozen predictions bit-for-bit; the default thread count on "
            "this 224-core box does not.  Declared, verified by the "
            "reproduction gate.",
        ],
        "test_reads_spent": 1,
        "never_selected_on_test": True,
        "declared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    jdump(proto, "protocol.json")
    import hashlib
    h = hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest()
    (OUT / "protocol.sha256").write_text(h + "\n")
    log(f"  protocol sha256 {h}")
    return proto


# =============================================================================
# neural anchors
# =============================================================================
def neural_anchors(ev_val, ev_test):
    """Refit the two matched neural backbones and return their anchors.

    Returns {kind: {"val": p_row_val, "test": p_row_test,
                    "mem_val": p_memory_bars_val, "mem_test": p_memory_bars_test}}
    """
    import torch
    torch.set_num_threads(8)                    # DECLARED: required to reproduce
    import exp_gradient_tta_lib as L
    from sklearn.preprocessing import QuantileTransformer

    panel = cpc.load_event_panel()
    tr = panel[panel.datetime < cpc.TRAIN_END].reset_index(drop=True)
    pretest = panel[panel.datetime < cpc.TEST_START].reset_index(drop=True)
    FEATS = cpc.MODEL_FEATURES
    y_tr = tr.target_hi_vol.to_numpy(int)
    y_pre = pretest.target_hi_vol.to_numpy(int)
    x_tr_r = L.frame_matrix(tr, FEATS)
    x_pre_r = L.frame_matrix(pretest, FEATS)
    x_va_r = L.frame_matrix(ev_val.stream, FEATS)
    x_te_r = L.frame_matrix(ev_test.stream, FEATS)
    x_memv_r = L.frame_matrix(ev_val.memory.frame, FEATS)
    x_memt_r = L.frame_matrix(ev_test.memory.frame, FEATS)

    arch_sweep = json.load(
        open(ROOT / "config" / "arch_sweep2.json"))
    cfg = {}
    for kind in ("mlp_bn", "ft_trans"):
        best = max([r for r in arch_sweep if r["kind"] == kind],
                   key=lambda r: r["val_auc"])
        cfg[kind] = dict(prep=best["prep"], epochs=best["epochs"], lr=best["lr"],
                         wd=best["wd"],
                         arch={k: v for k, v in best.items()
                               if k not in ("prep", "kind", "epochs", "lr", "wd",
                                            "val_auc")})
    log(f"  backbone configs {json.dumps(cfg)}")

    def make_prep(kindprep, xfit):
        if kindprep == "zscore":
            s = L.Standardizer.fit(xfit)
            return lambda x: s(x)
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                 subsample=10 ** 9, random_state=0).fit(xfit)
        return lambda x: qt.transform(x)

    stored = pd.read_csv(
        ROOT / "results" / "new_baselines" / "neural_test_predictions.csv")
    assert (stored.datetime.to_numpy()
            == ev_test.stream.datetime.astype(str).to_numpy()).all()

    anchors, gate = {}, {}
    for kind, c in cfg.items():
        t0 = time.perf_counter()
        prep_v = make_prep(c["prep"], x_tr_r)
        mv = [L.train_backbone(kind, prep_v(x_tr_r), y_tr, seed=s,
                               epochs=c["epochs"], lr=c["lr"],
                               weight_decay=c["wd"], arch=c["arch"])
              for s in range(5)]
        p_val = np.mean([L.predict_frozen(m, prep_v(x_va_r)) for m in mv], axis=0)
        p_memv = np.mean([L.predict_frozen(m, prep_v(x_memv_r)) for m in mv], axis=0)
        prep_t = make_prep(c["prep"], x_pre_r)
        mt = [L.train_backbone(kind, prep_t(x_pre_r), y_pre, seed=s,
                               epochs=c["epochs"], lr=c["lr"],
                               weight_decay=c["wd"], arch=c["arch"])
              for s in range(5)]
        p_te = np.mean([L.predict_frozen(m, prep_t(x_te_r)) for m in mt], axis=0)
        p_memt = np.mean([L.predict_frozen(m, prep_t(x_memt_r)) for m in mt], axis=0)
        d = float(np.max(np.abs(p_te - stored[f"{kind}__frozen"].to_numpy())))
        gate[kind] = d
        log(f"  [{kind}] refit {time.perf_counter()-t0:.1f}s | frozen val bar "
            f"{ev_val.bi.auc_bar(p_val):.4f} test bar {ev_test.bi.auc_bar(p_te):.4f}"
            f" | reproduction gate max|diff| = {d:.3e}")
        if d > 1e-6:
            raise AssertionError(f"{kind}: frozen refit does not reproduce the "
                                 f"published column (max|diff| {d:.3e})")
        anchors[kind] = dict(val=p_val, test=p_te, mem_val=p_memv, mem_test=p_memt)
    jdump(gate, "backbone_reproduction_gate.json")
    return anchors


# =============================================================================
# generic (anchor-injected) scorers
# =============================================================================
def interp(frozen_row, bi, bar_score, lam, kind):
    r = bi.spread(bar_score)
    if kind == "prob":
        return (1.0 - lam) * frozen_row + lam * r
    rc = np.clip(r, 1e-6, 1 - 1e-6)
    return cpc.sigmoid((1.0 - lam) * cpc.logit(frozen_row) + lam * cpc.logit(rc))


def t3a_bar_scores_generic(env, space, m, mem_p, frozen_row):
    """eas.t3a_bar_scores with the anchor injected instead of env.frozen."""
    s = env.spaces[space]
    mem_x = s["mem_x"]
    mem_lab = (mem_p >= 0.5).astype(int)
    ent = -(mem_p * np.log(np.clip(mem_p, 1e-12, 1))
            + (1 - mem_p) * np.log(np.clip(1 - mem_p, 1e-12, 1)))
    sup = {c: [(float(ent[i]), mem_x[i]) for i in np.flatnonzero(mem_lab == c)]
           for c in (0, 1)}
    q = s["queries"]
    pbar = env.bi.pool(frozen_row)
    qent = -(pbar * np.log(np.clip(pbar, 1e-12, 1))
             + (1 - pbar) * np.log(np.clip(1 - pbar, 1e-12, 1)))
    scores = np.empty(len(q), dtype=float)
    for t in range(len(q)):
        c = int(pbar[t] >= 0.5)
        sup[c].append((float(qent[t]), q[t]))
        if m > 0:
            for cc in (0, 1):
                if len(sup[cc]) > m:
                    sup[cc] = sorted(sup[cc], key=lambda z: z[0])[:m]
        cent = {}
        for cc in (0, 1):
            cent[cc] = (np.mean([v for _, v in sup[cc]], axis=0) if sup[cc]
                        else np.zeros(q.shape[1]))
        z = q[t]
        nz = np.linalg.norm(z) + 1e-12
        s1 = float(z @ cent[1]) / (nz * (np.linalg.norm(cent[1]) + 1e-12))
        s0 = float(z @ cent[0]) / (nz * (np.linalg.norm(cent[0]) + 1e-12))
        scores[t] = cpc.sigmoid((s1 - s0) * 4.0)
    return scores


def fixed_grid_knnlm(env, frozen_row, kind, blends):
    """Fixed-grid k-NN-LM: the published R3 axis VALUES, uniform average, no
    validation consulted.  kind in {prob, logit}; blends = the lambda grid."""
    total = np.zeros(env.bi.n_rows, dtype=float)
    count = 0
    for sp in cpc.RETRIEVAL_SPACES:
        for k in cpc.SCALES:
            for tp in cpc.TEMPERATURES:
                bar = eas.knn_bar_scores(env, sp, k, tp)
                for lam in blends:
                    total += interp(frozen_row, env.bi, bar, lam, kind)
                    count += 1
    return total / count, count


# =============================================================================
# stage: run
# =============================================================================
def stage_run(args):
    log("building environments (clean panel, GBT anchor + neighbour spaces)")
    panel, ev_val, ev_test = eas.build_envs()
    cpc.assert_clean(panel, where="clean event panel")
    max_dt = pd.Timestamp(panel.datetime.max())
    n_fwd = int((panel.datetime >= FORWARD_START).sum())
    n_past_end = int((panel.datetime >= TEST_END).sum())
    guard = {"max_datetime_read": str(max_dt),
             "n_rows_at_or_after_TEST_END": n_past_end,
             "n_rows_in_forward_window_seen": n_fwd,
             "n_test_bars": int(ev_test.bi.n_bars),
             "n_test_rows": int(ev_test.bi.n_rows),
             "n_val_bars": int(ev_val.bi.n_bars),
             "n_val_rows": int(ev_val.bi.n_rows),
             "n_memory_bars_test": int(len(ev_test.memory)),
             "n_memory_bars_val": int(len(ev_val.memory))}
    assert n_fwd == 0 and n_past_end == 0, guard
    jdump(guard, "run_guard.json")
    log(f"  guard OK: max datetime read {max_dt}; {ev_test.bi.n_bars} test bars")

    # ---------------------------------------------------------------- anchors
    log("refitting the matched neural backbones")
    na_ = neural_anchors(ev_val, ev_test)
    anchor = {
        "gbt": dict(val=ev_val.frozen, test=ev_test.frozen,
                    mem_val=ev_val._mem_pseudo, mem_test=ev_test._mem_pseudo),
        "mlp_bn": na_["mlp_bn"], "ft_trans": na_["ft_trans"],
    }

    # ------------------------------------------- anchor-independent bar scores
    log("computing anchor-independent retrieval bar scores (shared)")
    BARS = {}     # (arm, hp_key) -> dict(val=..., test=..., hp=dict)

    def put(arm, hp, bv, bt):
        BARS[(arm, json.dumps(hp, sort_keys=True, default=str))] = dict(
            arm=arm, hp=hp, val=bv, test=bt)

    for sp in cpc.RETRIEVAL_SPACES:
        for k in eas.KNN_K:
            put("knn_vote", dict(space=sp, k=k, temp="unif"),
                eas.knn_bar_scores(ev_val, sp, k, "unif"),
                eas.knn_bar_scores(ev_test, sp, k, "unif"))
            for tp in eas.KNN_T:
                put("knnlm_sel", dict(space=sp, k=k, temp=str(tp)),
                    eas.knn_bar_scores(ev_val, sp, k, tp),
                    eas.knn_bar_scores(ev_test, sp, k, tp))
    for sp in cpc.RETRIEVAL_SPACES:
        for h in eas.NW_H:
            bv, neff = eas.nw_bar_scores(ev_val, sp, h)
            bt, _ = eas.nw_bar_scores(ev_test, sp, h)
            put("nadaraya_watson", dict(space=sp, bandwidth=h,
                                        eff_neighbours=round(neff, 2)), bv, bt)
        for tau in eas.PROTO_TAU:
            put("prototype", dict(space=sp, tau=tau),
                eas.prototype_bar_scores(ev_val, sp, tau),
                eas.prototype_bar_scores(ev_test, sp, tau))
        for k in (8, 16, 32, 64, 128, 256):
            for tau in eas.ADANPC_TAU:
                for mg in eas.ADANPC_MARGIN:
                    bv, nav = eas.adanpc_bar_scores(ev_val, sp, k, tau, mg)
                    bt, nat = eas.adanpc_bar_scores(ev_test, sp, k, tau, mg)
                    put("adanpc", dict(space=sp, k=k, tau=tau, margin=mg,
                                       n_pseudo_added_test=nat), bv, bt)
    log(f"  {len(BARS)} shared bar-score configurations cached")

    # --------------------------------------------------------------- scoring
    sweeps, metrics, preds_all = [], [], {}
    for bb in BACKBONES:
        fz_v, fz_t = anchor[bb]["val"], anchor[bb]["test"]
        log(f"backbone {bb}: frozen val bar {ev_val.bi.auc_bar(fz_v):.4f} "
            f"test bar {ev_test.bi.auc_bar(fz_t):.4f}")
        P, SEL = {}, {}
        P["frozen"] = fz_t

        # ---- R3 arms (budget 0 on every backbone) --------------------------
        p_sf, _, _, n_sf = cpc.r3ttt_selection_free(
            ev_test.memory, ev_test.stream, fz_t, groups=ev_test.bi.groups)
        assert n_sf == 72, n_sf
        P["r3ttt_sf72"] = p_sf
        p_dec, _, _, n_dec = cpc.r3ttt_selection_free(
            ev_test.memory, ev_test.stream, fz_t, groups=ev_test.bi.groups,
            blends=DECLARED_R3["betas"], gates=DECLARED_R3["gates"],
            prior_strengths=DECLARED_R3["lambdas"],
            temperatures=DECLARED_R3["temperatures"], scales=DECLARED_R3["scales"])
        assert n_dec == 12, n_dec
        P["r3ttt_declared"] = p_dec
        SEL["r3ttt_sf72"] = {"selection": "NONE -- 72-member uniform average"}
        SEL["r3ttt_declared"] = {"selection": "NONE on this backbone -- config "
                                 "declared on the tree's VALIDATION stream in "
                                 "results/method_respec_0911"}

        # ---- fixed-grid k-NN-LM (budget 0) --------------------------------
        for arm, spec in FIXED_GRID_ARMS.items():
            pv, nv = fixed_grid_knnlm(ev_val, fz_v, spec["kind"], spec["blends"])
            pt, nt = fixed_grid_knnlm(ev_test, fz_t, spec["kind"], spec["blends"])
            P[arm] = pt
            SEL[arm] = {"selection": f"NONE -- {nt}-member uniform average",
                        "kind": spec["kind"], "lambdas": list(spec["blends"])}
            log(f"  [{bb}] {arm:32s} n={nt:3d} val bar "
                f"{ev_val.bi.auc_bar(pv):.4f} -> TEST bar "
                f"{ev_test.bi.auc_bar(pt):.4f}")

        # ---- validation-selected retrieval arms ---------------------------
        # retrieval-only arms: the anchor plays no part
        for arm, src in (("knn_vote", "knn_vote"), ("adanpc_vote", "adanpc")):
            best = None
            for rec in BARS.values():
                if rec["arm"] != src:
                    continue
                a = ev_val.bi.auc_bar(ev_val.bi.spread(rec["val"]))
                sweeps.append(dict(backbone=bb, arm=arm, **rec["hp"], lam=1.0,
                                   kind="retrieval_only", val_auc_bar=a))
                if best is None or a > best[0]:
                    best = (a, rec)
            P[arm] = ev_test.bi.spread(best[1]["test"])
            SEL[arm] = dict(best[1]["hp"], lam=1.0, val_auc_bar=best[0])
            log(f"  [{bb}] {arm:32s} sel={json.dumps(best[1]['hp'], default=str)}"
                f" val {best[0]:.4f} -> TEST bar {ev_test.bi.auc_bar(P[arm]):.4f}")

        # fused arms: lambda (and the arm's knobs) chosen on validation per-bar AUC
        fused = [("knnlm_sel_prob", "knnlm_sel", "prob"),
                 ("knnlm_sel_logit", "knnlm_sel", "logit"),
                 ("nadaraya_watson", "nadaraya_watson", "prob"),
                 ("prototype", "prototype", "prob"),
                 ("adanpc_fused", "adanpc", "prob")]
        for arm, src, kind in fused:
            best = None
            for rec in BARS.values():
                if rec["arm"] != src:
                    continue
                for lam in eas.LAMBDAS:
                    pv = interp(fz_v, ev_val.bi, rec["val"], lam, kind)
                    a = ev_val.bi.auc_bar(pv)
                    sweeps.append(dict(backbone=bb, arm=arm, **rec["hp"],
                                       lam=float(lam), kind=kind, val_auc_bar=a))
                    if best is None or a > best[0]:
                        best = (a, rec, float(lam))
            P[arm] = interp(fz_t, ev_test.bi, best[1]["test"], best[2], kind)
            SEL[arm] = dict(best[1]["hp"], lam=best[2], kind=kind,
                            val_auc_bar=best[0])
            log(f"  [{bb}] {arm:32s} sel={json.dumps(SEL[arm], default=str)} "
                f"-> TEST bar {ev_test.bi.auc_bar(P[arm]):.4f}")

        # t3a_routing: anchor-dependent bar score, recomputed per backbone
        best = None
        for sp in cpc.RETRIEVAL_SPACES:
            for m in eas.T3A_M:
                bv = t3a_bar_scores_generic(ev_val, sp, m, anchor[bb]["mem_val"],
                                            fz_v)
                bt = t3a_bar_scores_generic(ev_test, sp, m, anchor[bb]["mem_test"],
                                            fz_t)
                for lam in eas.LAMBDAS:
                    pv = interp(fz_v, ev_val.bi, bv, lam, "prob")
                    a = ev_val.bi.auc_bar(pv)
                    sweeps.append(dict(backbone=bb, arm="t3a_routing", space=sp,
                                       support_M=m, lam=float(lam), kind="prob",
                                       val_auc_bar=a))
                    if best is None or a > best[0]:
                        best = (a, bt, float(lam), dict(space=sp, support_M=m))
        P["t3a_routing"] = interp(fz_t, ev_test.bi, best[1], best[2], "prob")
        SEL["t3a_routing"] = dict(best[3], lam=best[2], kind="prob",
                                  val_auc_bar=best[0])
        log(f"  [{bb}] {'t3a_routing':32s} sel={json.dumps(SEL['t3a_routing'])} "
            f"-> TEST bar {ev_test.bi.auc_bar(P['t3a_routing']):.4f}")

        preds_all[bb] = P
        for name, p in P.items():
            metrics.append(dict(backbone=bb, arm=name,
                                budget=BUDGETS.get(name, np.nan),
                                **ev_test.bi.scores(p),
                                selected=json.dumps(SEL.get(name, {}),
                                                    default=str)))

    save(pd.DataFrame(sweeps), "validation_sweep.csv")
    save(pd.DataFrame(metrics), "metrics.csv")

    # ------------------------------------------------- provenance gate vs 09-01
    z = np.load(ROOT / "results" / "new_baselines" / "retrieval_test_predictions.npz",
                allow_pickle=True)
    pg = {}
    for mine, theirs in (("frozen", "frozen"), ("r3ttt_sf72", "r3ttt_sf"),
                         ("knnlm_fixedgrid_logit", "knnlm_sf72_logit"),
                         ("knnlm_fixedgrid_prob", "knnlm_sf72"),
                         ("knn_vote", "knn_vote"),
                         ("knnlm_sel_prob", "knnlm"),
                         ("knnlm_sel_logit", "knnlm_logit"),
                         ("nadaraya_watson", "nadaraya_watson"),
                         ("prototype", "prototype"),
                         ("t3a_routing", "t3a_routing"),
                         ("adanpc_vote", "adanpc_vote"),
                         ("adanpc_fused", "adanpc_fused")):
        pg[mine] = float(np.max(np.abs(
            np.asarray(preds_all["gbt"][mine], float) - np.asarray(z[theirs], float))))
    worst = max(pg.values())
    pg["_worst"] = worst
    pg["_published_declared_r3_test_auc_bar"] = 0.7852333056827947
    pg["_our_declared_r3_test_auc_bar"] = float(
        ev_test.bi.auc_bar(preds_all["gbt"]["r3ttt_declared"]))
    jdump(pg, "provenance_gate.json")
    log(f"  GBT provenance gate: worst max|diff| vs 2026-09-01 npz = {worst:.3e}")
    if worst > 1e-9:
        raise AssertionError(f"GBT arms do not reproduce new_baselines ({worst:.3e})")

    # ---------------------------------------------------------------- paired
    log("paired 5-day moving-block bootstrap, 5,000 reps, seed 42")
    rows = []
    bi = ev_test.bi
    contrasts = []
    for bb in BACKBONES:
        for a in R3_ARMS:
            for b in ("frozen",) + RETRIEVAL_ARMS:
                contrasts.append((bb, a, b))
        contrasts.append((bb, "r3ttt_declared", "r3ttt_sf72"))
    for i, (bb, a, b) in enumerate(contrasts):
        pa, pb = preds_all[bb][a], preds_all[bb][b]
        for tag, (y, xa, xb, days) in {
            "bar": (bi.y_bar, bi.pool(pa), bi.pool(pb), bi.bar_times),
            "post": (bi.y_row, np.asarray(pa), np.asarray(pb), bi.times),
        }.items():
            for metric in ("auc", "brier", "logloss"):
                r = eas.paired_boot(y, xa, xb, days, metric=metric, n_boot=N_BOOT)
                rows.append(dict(backbone=bb, arm_a=a, arm_b=b, estimand=tag,
                                 metric=metric, value_a=r["point_a"],
                                 value_b=r["point_b"], delta=r["delta"],
                                 ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                 p_delta_le_zero=r["p_delta_le_zero"],
                                 n_boot=r["n_boot"]))
        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{len(contrasts)} contrasts")
    paired = pd.DataFrame(rows)
    save(paired, "paired.csv")

    # ---- Holm over the per-backbone R3-minus-retrieval per-bar AUC family ----
    holm = []
    for bb in BACKBONES:
        for a in R3_ARMS:
            sub = paired[(paired.backbone == bb) & (paired.arm_a == a)
                         & (paired.estimand == "bar") & (paired.metric == "auc")
                         & (paired.arm_b.isin(RETRIEVAL_ARMS))].copy()
            sub = sub.sort_values("p_delta_le_zero").reset_index(drop=True)
            m = len(sub)
            run = 0.0
            for j, r in sub.iterrows():
                adj = min(1.0, max(run, (m - j) * r.p_delta_le_zero))
                run = adj
                holm.append(dict(backbone=bb, arm_a=a, arm_b=r.arm_b,
                                 delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
                                 p_raw=r.p_delta_le_zero, p_holm=adj,
                                 family_size=m))
    save(pd.DataFrame(holm), "paired_holm.csv")

    # ----------------------------------------------------------- family count
    n_metrics_cells = int(len(pd.DataFrame(metrics)))
    n_paired_cells = int(len(contrasts) * 2)          # (contrast, estimand)
    fc = {
        "track": "knn_neural_0911",
        "family": "ACCURACY (AUC / Brier / log loss on the 2026 test panel). "
                  "This is NOT the returns family; the disclosed returns family "
                  "stays at 9,887 test cells and the returns deflation bar is "
                  "unchanged.",
        "convention": "one cell = one (arm|contrast, estimand) pair, the "
                      "convention used by results/matched_backbone_paired",
        "metrics_rows": n_metrics_cells,
        "metrics_cells_arm_by_estimand": n_metrics_cells * 2,
        "paired_contrasts": len(contrasts),
        "paired_cells_contrast_by_estimand": n_paired_cells,
        "paired_numbers_contrast_by_estimand_by_metric": len(contrasts) * 2 * 3,
        "already_disclosed_subset": {
            "gbt_backbone_rows": "the 12 GBT retrieval arms and their levels were "
                                 "disclosed on 2026-09-01 in "
                                 "results/new_baselines; this run reproduces them "
                                 "to 1e-9 and adds no new information for them",
            "gbt_r3_declared": "disclosed 2026-09-11 in results/method_respec_0911",
        },
        "genuinely_new_cells": None,   # filled below
    }
    new_contrasts = [c for c in contrasts if c[0] != "gbt"]
    fc["genuinely_new_paired_contrasts"] = len(new_contrasts)
    fc["genuinely_new_cells"] = len(new_contrasts) * 2
    fc["genuinely_new_arm_levels"] = int(
        len([m for m in metrics if m["backbone"] != "gbt"
             and m["arm"] not in ("frozen", "r3ttt_sf72")]) * 2)
    jdump(fc, "family_count.json")
    log("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "protocol", "run"])
    args = ap.parse_args()
    if args.stage in ("all", "protocol"):
        stage_protocol()
    if args.stage in ("all", "run"):
        stage_run(args)


if __name__ == "__main__":
    main()
