#!/usr/bin/env python3
"""Lock the 0923 final method BEFORE any test row is scored.  2026-09-23 (T1).

1. Applies the selection rules written in exp_candidates_0923.py to the 16
   screened candidates, reading ONLY the walk-forward and validation tables.
2. Verifies, on validation (test-free), that the declared closed-form spec
   (`exp_ablation_0923.evid`) reproduces the screened arm exactly.
3. Writes results/r3ttt_0923/protocol_0923.json: final arm + exact spec,
   selection evidence, the pre-registered test reads, the pre-registered
   (NOT run) forward-window prediction read, sha256 of every code file, and the
   sha256 of the protocol body.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import exp_lib_0923 as lib                            # noqa: E402

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import common_protocol_clean as cpc                   # noqa: E402

CANDS = ("c01_kmem", "c02_wide72", "c03_ll_kmem", "c04_ll_mix", "c05_evid_logit",
         "c06_evid_prob", "c07_joint2", "c08_postpred", "c09_evid_prob_wide",
         "c10_evid_prob_joint2", "c11_evid_prob_neff", "c12_evid_prob_lepski",
         "c13_superset", "c14_selfcal", "c15_diagmetric", "c16_samworth")
# components beyond respec's fixed-beta fusion, for the tie-break
COMPONENTS = {"c01_kmem": 1, "c02_wide72": 2, "c03_ll_kmem": 2, "c04_ll_mix": 1,
              "c05_evid_logit": 2, "c06_evid_prob": 1, "c07_joint2": 1,
              "c08_postpred": 2, "c09_evid_prob_wide": 2, "c10_evid_prob_joint2": 2,
              "c11_evid_prob_neff": 2, "c12_evid_prob_lepski": 2,
              "c13_superset": 4, "c14_selfcal": 3, "c15_diagmetric": 3,
              "c16_samworth": 3}
SPECS = {
    "c07_joint2": dict(family="respec", spaces=["timing", "market_state", "joint"],
                       temps=[0.25, 1.0], lams=[2.0, 8.0], scales=[8, 16, 32, 64],
                       beta=0.5, space_w={"joint": 2.0}, gate=False, shrink=True,
                       fusion="logit"),
    "c10_evid_prob_joint2": dict(spaces=["timing", "market_state", "joint"],
                                 temps=[0.25, 1.0], scales=[8, 16, 32, 64],
                                 cmult=[1.0], space_w={"joint": 2.0},
                                 weight="mass", center="anchor", fusion="prob"),
    "c11_evid_prob_neff": dict(spaces=["timing", "market_state", "joint"],
                               temps=[0.25, 1.0], scales=[8, 16, 32, 64],
                               cmult=[1.0], space_w=None,
                               weight="neff", center="anchor", fusion="prob"),
    "c06_evid_prob": dict(spaces=["timing", "market_state", "joint"],
                          temps=[0.25, 1.0], scales=[8, 16, 32, 64], cmult=[1.0],
                          space_w=None, weight="mass", center="anchor", fusion="prob"),
}
SECONDARY = "c10_evid_prob_joint2"      # the v1-rule pick, pre-declared secondary
CODE = ("exp_lib_0923.py", "exp_candidates_0923.py", "exp_final_test_0923.py",
        "exp_ablation_0923.py", "exp_crypto_0923.py", "exp_forward_0923.py",
        "exp_declare_0923.py")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ROUNDS = (1, 2, 3, 4)


def screen():
    wf = pd.concat([pd.read_csv(lib.OUT / f"named_walkforward_r{r}.csv")
                    for r in ROUNDS]).drop_duplicates("arm").set_index("arm")
    va = pd.concat([pd.read_csv(lib.OUT / f"named_validation_r{r}.csv")
                    for r in ROUNDS]).drop_duplicates("arm").set_index("arm")
    pf = None
    for r in ROUNDS:
        x = pd.read_csv(lib.OUT / f"walkforward_per_fold_r{r}.csv").set_index("fold")
        pf = x if pf is None else pf.join(x[[c for c in x.columns
                                             if c not in pf.columns]])
    rows = []
    for c in CANDS:
        rows.append(dict(
            arm=c, n_members=int(wf.loc[c, "n_members"]), components=COMPONENTS[c],
            wf_auc_bar=float(wf.loc[c, "auc_bar"]),
            wf_brier_bar=float(wf.loc[c, "brier_bar"]),
            wf_vs_respec=float(wf.loc[c, "vs_respec_bar_delta"]),
            wf_vs_respec_lo=float(wf.loc[c, "vs_respec_bar_ci_lo"]),
            wf_vs_respec_hi=float(wf.loc[c, "vs_respec_bar_ci_hi"]),
            wf_vs_respec_p=float(wf.loc[c, "vs_respec_bar_p_le_zero"]),
            wf_vs_knnlm72=float(wf.loc[c, "vs_knnlm72_bar_delta"]),
            wf_vs_knnlm72_lo=float(wf.loc[c, "vs_knnlm72_bar_ci_lo"]),
            wf_vs_knnlm72_hi=float(wf.loc[c, "vs_knnlm72_bar_ci_hi"]),
            wf_vs_knnlm72_p=float(wf.loc[c, "vs_knnlm72_bar_p_le_zero"]),
            wf_vs_adanpc=float(wf.loc[c, "vs_adanpc_fused_bar_delta"]),
            wf_vs_adanpc_p=float(wf.loc[c, "vs_adanpc_fused_bar_p_le_zero"]),
            wf_folds_beating_respec=int(wf.loc[c, "folds_beating_respec"]),
            wf_folds_beating_knnlm72=int(wf.loc[c, "folds_beating_knnlm72"]),
            wf_sign_p_vs_respec=float(wf.loc[c, "sign_p_vs_respec"]),
            val_auc_bar=float(va.loc[c, "auc_bar"]),
            val_vs_respec=float(va.loc[c, "vs_respec_bar_delta"]),
            val_vs_knnlm72=float(va.loc[c, "vs_knnlm72_bar_delta"]),
            val_vs_frozen=float(va.loc[c, "vs_frozen_bar_delta"]),
            **fold_stats(pf, c)))
    t = pd.DataFrame(rows).sort_values("wf_auc_bar", ascending=False)
    t["v1_R1"] = (t.wf_vs_respec > 0) & (t.wf_vs_respec_p < 0.05)
    t["v1_R2"] = t.val_vs_respec >= -0.002
    v1 = pick_v1(t)
    wf_respec = float(wf.loc["respec", "auc_bar"])
    wf_knn = float(wf.loc["knnlm72", "auc_bar"])
    t["v2_a"] = (t.wf_auc_bar >= wf_respec) & (t.wf_auc_bar >= wf_knn)
    t["v2_c"] = t.val_vs_frozen >= 0
    t["v2_eligible"] = t.v2_a & t.v2_c
    pool = t[t.v2_eligible] if t.v2_eligible.any() else t
    pool = pool.assign(_neg=pool.wf_folds_negative_vs_frozen,
                       _worst=-pool.wf_worst_fold_vs_frozen,
                       _auc=-pool.wf_auc_bar, _comp=pool.components)
    v2 = pool.sort_values(["_neg", "_worst", "_auc", "_comp"]).iloc[0]["arm"]
    t["v2_rank"] = t["arm"].map({a: i + 1 for i, a in enumerate(
        pool.sort_values(["_neg", "_worst", "_auc", "_comp"])["arm"])})
    return t, v1, v2, bool(t.v2_eligible.any())


def fold_stats(pf, c):
    d = pf[f"auc_bar_{c}"] - pf["auc_bar_frozen"]
    dk = pf[f"auc_bar_{c}"] - pf["auc_bar_knnlm72"]
    dr = pf[f"auc_bar_{c}"] - pf["auc_bar_respec"]
    return dict(wf_folds_negative_vs_frozen=int((d < 0).sum()),
                wf_worst_fold_vs_frozen=float(d.min()),
                wf_mean_fold_vs_frozen=float(d.mean()),
                wf_folds_negative_vs_knnlm72=int((dk < 0).sum()),
                wf_worst_fold_vs_knnlm72=float(dk.min()),
                wf_folds_negative_vs_respec=int((dr < 0).sum()),
                wf_worst_fold_vs_respec=float(dr.min()),
                wf_per_fold_vs_frozen=json.dumps({k: float(v) for k, v in d.items()}))


def pick_v1(t):
    ok = t[t.v1_R1 & t.v1_R2]
    if ok.empty:
        return "respec"
    best = ok.wf_auc_bar.max()
    tied = ok[ok.wf_auc_bar >= best - 0.0005]
    fewest = tied[tied.components == tied.components.min()]
    return fewest.sort_values("wf_auc_bar", ascending=False).iloc[0]["arm"]


def main():
    t, v1, final, eligible = screen()
    t.to_csv(lib.OUT / "candidate_table_0923.csv", index=False)
    show = [c for c in t.columns if c != "wf_per_fold_vs_frozen"]
    lib.log("\n" + t[show].round(5).to_string(index=False))
    lib.log(f"rule v1 -> {v1};  rule v2 (declared) -> {final}; any eligible: {eligible}")
    if "--dry" in sys.argv:
        return
    spec = SPECS[final]

    # spec == screened arm, on validation only
    import exp_ablation_0923 as ab
    panel = lib.load_panel(allow_test=False)
    fold = lib.Fold(panel[panel.datetime < cpc.TRAIN_END],
                    panel[panel.datetime >= cpc.TRAIN_END], cpc.TRAIN_END)
    p_lib = lib.all_arms(fold, which={final})[final]
    kw = dict(spec)
    for k in ("spaces", "temps", "scales", "cmult", "lams"):
        if k in kw:
            kw[k] = tuple(kw[k])
    p_ab = ab.build(fold, kw)
    p_sec = lib.all_arms(fold, which={SECONDARY})[SECONDARY]
    kw2 = dict(SPECS[SECONDARY])
    for k in ("spaces", "temps", "scales", "cmult"):
        kw2[k] = tuple(kw2[k])
    assert float(np.max(np.abs(p_sec - ab.build(fold, kw2)))) < 1e-12
    diff = float(np.max(np.abs(p_lib - p_ab)))
    lib.log(f"spec reproduces screened arm on validation: max|d| = {diff:.2e}")
    assert diff < 1e-12

    body = {
        "track": "r3ttt_0923",
        "declared_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "final_arm": final,
        "final_name": "R3-TTT, joint-emphasised three-space average (gate-free)",
        "final_spec": spec,
        "final_formula": (
            "p_e = sum_m w_m sigmoid((1-beta) l_e + beta ubar^(m)_b) / sum_m w_m, "
            "m = (space, T, lam) over 3 spaces x T{0.25,1} x lam{2,8} = 12 members, "
            "ubar^(m)_b = mean over k in {8,16,32,64} of logit((S_k + lam pi0)/"
            "(W_k + lam)) (eqs. m_faststate, m_scalemean), beta = 1/2, no gate; "
            "w_m = 2 for the joint space, 1 otherwise (joint space carries half the "
            "ensemble weight).  Zero backward passes; memory frozen at the "
            "boundary; one fast state per bar."),
        "secondary_arms": [SECONDARY],
        "secondary_note": (
            "c10_evid_prob_joint2 is what the ORIGINAL (v1) rule selects; it is "
            "pre-declared here as a secondary arm, scored in the same single test "
            "read and reported, never substituted for the primary."),
        "selection_rule": (
            "v2 (exp_candidates_0923.py docstring, coordinator steer): eligible = "
            "pooled WF bar AUC >= respec and >= kNN-LM-72, validation delta vs "
            "frozen >= 0, one definition for all backbones; rank by fewest WF folds "
            "negative vs frozen, then largest worst-fold delta vs frozen, then "
            "pooled WF AUC, then fewer components"),
        "selection_disclosure": (
            "The v1 rule (highest pooled WF AUC with P<0.05 vs respec, validation "
            "vs respec >= -0.002) was written before any candidate was scored and "
            "selects c10_evid_prob_joint2.  The v2 rule arrived from the "
            "coordinator AFTER the round 1-3 pooled WF and validation tables had "
            "been read, and was written into the code before round 4 (c13-c16) "
            "was scored and before any per-fold-vs-frozen statistic was "
            "computed.  v2 selects c07_joint2.  Both are reported; c07 is primary "
            "because v2 is the rule in force at declaration.  c14_selfcal has the "
            "highest pooled WF AUC (0.7263) but fails v2(c) (validation -0.0007 vs "
            "frozen) and v1 R2; it is not scored on test."),
        "rule_v1_pick": v1,
        "rule_v2_pick": final,
        "n_candidates_screened": len(CANDS),
        "screening_rounds": {"1": list(CANDS[:8]), "2": list(CANDS[8:11]),
                             "3 (T2 literature, declared before round-2 numbers)":
                                 [CANDS[11]],
                             "4 (coordinator ideas, declared with rule v2)":
                                 list(CANDS[12:])},
        "selection_evidence": t.set_index("arm").loc[final].to_dict(),
        "all_candidates_file": "results/r3ttt_0923/candidate_table_0923.csv",
        "test_reads": {
            "what": "ONE read of the 2026 test panel (2026-01-01 .. 2026-04-10, "
                    "525 bars) for the final arm and its references, split over "
                    "three scripts run once each, nothing retuned",
            "scripts": ["exp_final_test_0923.py (GBT, MLP+BN, FT-Transformer; "
                        "cost)", "exp_ablation_0923.py --panel test (ablations + "
                        "sensitivity; reporting only)",
                        "exp_crypto_0923.py (BTC/ETH + 40-perpetual breadth)"],
            "primary_contrasts": ["final - frozen", "final - respec",
                                  "final - knnlm72", "final - adanpc_fused"],
            "secondary_contrasts": ["c10 - frozen", "c10 - respec",
                                    "c10 - knnlm72", "c10 - adanpc_fused"],
            "comparators_fixed": {"knnlm72": "72-member fixed grid, logit, budget 0",
                                  "adanpc_fused": lib.ADANPC_CFG},
        },
        "forward_window_prediction_read": {
            "status": "PRE-REGISTERED, NOT RUN; requires coordinator authorisation",
            "window": "2026-04-16 .. 2026-08-25 (unscored for prediction)",
            "script": "exp_forward_0923.py --authorised-by-coordinator",
            "deployment": "fresh deployment at 2026-04-16: frozen GBT + memory "
                          "from every row before 2026-04-11; stream = forward "
                          "vintage posts aligned in [2026-04-16, 2026-08-26)",
            "backbone": "gbt",
            "primary": "final - frozen (per-bar AUC)",
            "secondary": ["final - knnlm72", "final - adanpc_fused"],
            "also_reported_not_in_holm_family": ["c10 (secondary arm) - frozen"],
            "test": "paired 5,000-replicate 5-day moving-block bootstrap, seed 42; "
                    "one-sided H1 delta>0, p = P_boot(delta<=0); Holm over the "
                    "three contrasts at 0.05; two-sided 95% interval reported",
            "reads_permitted": 1,
        },
        "uncertainty": "paired 5,000-replicate 5-day moving-block bootstrap, "
                       "clustered by calendar day, seed 42, pooled to the bar",
        "threads_pinned": 1,
        "code_sha256": {f"code/{f}": sha(ROOT / "code" / f) for f in CODE},
    }
    blob = json.dumps(body, sort_keys=True, default=str).encode()
    body = json.loads(blob)
    out = dict(body)
    out["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    p = lib.OUT / "protocol_0923.json"
    assert not p.exists(), "protocol already locked"
    p.write_text(json.dumps(out, indent=2))
    lib.log(f"LOCKED {final} sha256:{out['sha256']}")


if __name__ == "__main__":
    main()
