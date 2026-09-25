#!/usr/bin/env python3
"""The single 0923 test read.  2026-09-23 (T1).

Refuses to run unless results/r3ttt_0923/protocol_0923.json exists, its body
hashes to its recorded sha256, and the sha256 of every code file it lists
matches the file on disk -- i.e. the method scored here is byte-for-byte the
one declared from the walk-forward before any test row was read.

Scores, on the 2026 test panel (525 bars; nothing at/after 2026-04-11 exists
in the panel, nothing at/after 2026-04-16 is read -- asserted):
  * the declared final arm and every reference arm on GBT, MLP+BN and
    FT-Transformer (neural anchors refit exactly as results/knn_neural_0911,
    reproduction-gated to <1e-6 against the stored frozen columns);
  * provenance gates: frozen / respec / shipped72 / kNN-LM-72 / AdaNPC-fused
    must reproduce the saved 0911 levels;
  * cost: backward passes (0 by construction) and wall-clock per bar.
Nothing is retuned per backbone.
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

log = lib.log
PROTO = lib.OUT / "protocol_0923.json"
REFS = ("frozen", "shipped72", "respec", "knnlm", "knnlm72", "knnlm72_prob",
        "adanpc_fused", "local_linear")
CONTRAST_REFS = ("frozen", "respec", "knnlm72", "adanpc_fused", "knnlm72_prob",
                 "shipped72", "knnlm")
GATE_0911 = {   # backbone -> {our arm: (0911 arm name)}
    "frozen": "frozen", "respec": "r3ttt_declared", "shipped72": "r3ttt_sf72",
    "knnlm72": "knnlm_fixedgrid_logit", "knnlm72_prob": "knnlm_fixedgrid_prob",
    "adanpc_fused": "adanpc_fused",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_protocol():
    d = json.loads(PROTO.read_text())
    body = {k: v for k, v in d.items() if k != "sha256"}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    assert h == d["sha256"], "protocol body does not hash to its sha256"
    for rel, want in d["code_sha256"].items():
        got = sha(ROOT / rel)
        assert got == want, f"{rel} changed after declaration ({got[:12]} != {want[:12]})"
    log(f"protocol OK sha256:{d['sha256'][:16]} final={d['final_arm']}")
    return d


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-val", metavar="ARM", default=None,
                    help="exercise the whole code path on the 2025-Q4 VALIDATION "
                         "panel for ARM, before the protocol exists; no test row")
    args = ap.parse_args()
    dry = args.dry_val is not None
    if dry:
        final, out_dir = args.dry_val, lib.OUT / "dry_val"
        out_dir.mkdir(exist_ok=True)
    else:
        proto = load_protocol()
        final, out_dir = proto["final_arm"], lib.OUT
    secondary = [] if dry else list(proto.get("secondary_arms", []))
    if dry:
        secondary = ["c10_evid_prob_joint2"]
    which = set(REFS) | {final} | set(secondary)
    stamp = {"test_read_started": time.strftime("%Y-%m-%d %H:%M:%S"), "dry_val": dry}

    panel = lib.load_panel(allow_test=not dry)
    lo = cpc.TRAIN_END if dry else cpc.TEST_START
    fit = panel[panel["datetime"] < lo]
    stream = panel[(panel["datetime"] >= lo) & (panel["datetime"] < lib.TEST_END)]
    fold = lib.Fold(fit, stream, lo)
    log(f"test: {fold.bi.n_rows} rows / {fold.bi.n_bars} bars, memory "
        f"{len(fold.memory)}")

    # ------------------------------------------------------------ anchors
    import exp_new_baselines as nb
    import exp_knn_neural_0911 as kn
    kn.OUT = out_dir                        # keep 0911 artefacts untouched
    nb._OUT = out_dir
    _, ev_val, ev_test = nb.build_envs()
    ev = ev_val if dry else ev_test
    assert (ev.stream["datetime"].to_numpy()
            == fold.stream["datetime"].to_numpy()).all()
    assert np.max(np.abs(ev.frozen - fold.frozen)) < 1e-12
    na = kn.neural_anchors(ev_val, ev_test)
    part = "val" if dry else "test"
    anchors = {"gbt": fold.frozen, "mlp_bn": na["mlp_bn"][part],
               "ft_trans": na["ft_trans"][part]}

    m0911 = pd.read_csv(ROOT / "results" / "knn_neural_0911" / "metrics.csv")
    preds_all, tables, gates = {}, [], {}
    for bb, anchor in anchors.items():
        f = fold if bb == "gbt" else fold.with_anchor(anchor)
        preds = lib.all_arms(f, which=which, backbone=bb)
        nm = preds.pop("_n_members")
        preds_all[bb] = preds
        ref = m0911[m0911.backbone == bb].set_index("arm")
        g = {}
        for mine, theirs in GATE_0911.items():
            g[mine] = abs(f.bi.scores(preds[mine])["auc_bar"]
                          - float(ref.loc[theirs, "auc_bar"]))
        if dry:
            g = {"respec_vs_0923_val": abs(f.bi.scores(preds["respec"])["auc_bar"]
                                            - 0.7495732) if bb == "gbt" else 0.0}
        gates[bb] = g
        log(f"[{bb}] provenance max|dAUC| = {max(g.values()):.2e}  {g}")
        assert max(g.values()) < (1e-6 if not dry else 1e-5), \
            f"{bb}: reference arms do not reproduce"
        tab = lib.score_table(f.bi, preds, refs=CONTRAST_REFS)
        tab.insert(0, "backbone", bb)
        tab["declared"] = tab["arm"] == final
        tab["secondary"] = tab["arm"].isin(secondary)
        tab["n_members"] = tab["arm"].map(nm)
        tables.append(tab)
        cols = ["arm", "auc_bar", "brier_bar", "vs_frozen_bar_delta",
                "vs_respec_bar_delta", "vs_knnlm72_bar_delta",
                "vs_knnlm72_bar_p_le_zero", "vs_adanpc_fused_bar_delta",
                "vs_adanpc_fused_bar_p_le_zero"]
        log(f"\n[{bb}]\n" + tab[cols].to_string(index=False))

    allt = pd.concat(tables, ignore_index=True)
    allt.to_csv(out_dir / "test_all_backbones_long.csv", index=False)
    allt[allt.backbone == "gbt"].drop(columns="backbone").to_csv(
        out_dir / "named_test_final.csv", index=False)
    np.savez_compressed(out_dir / "predictions_test_0923.npz",
                        y_row=fold.bi.y_row,
                        **{f"{bb}__{a}": p for bb, P in preds_all.items()
                           for a, p in P.items()})

    # backbone_generality.csv (wide, T5 schema)
    rows = []
    for bb in anchors:
        t = allt[allt.backbone == bb].set_index("arm")
        r = dict(backbone=bb, final_arm=final,
                 frozen_auc=t.loc["frozen", "auc_bar"],
                 frozen_brier=t.loc["frozen", "brier_bar"],
                 respec_auc=t.loc["respec", "auc_bar"],
                 final_auc=t.loc[final, "auc_bar"],
                 final_brier=t.loc[final, "brier_bar"],
                 knnlm72_auc=t.loc["knnlm72", "auc_bar"],
                 knnlm72_prob_auc=t.loc["knnlm72_prob", "auc_bar"],
                 adanpc_fused_auc=t.loc["adanpc_fused", "auc_bar"])
        for ref, tag in (("frozen", "vsfrozen"), ("respec", "vsrespec"),
                         ("knnlm72", "vsknnlm72"), ("adanpc_fused", "vsadanpc_fused"),
                         ("knnlm72_prob", "vsknnlm72_prob")):
            r[f"final_{tag}_auc_d"] = t.loc[final, f"vs_{ref}_bar_delta"]
            r[f"final_{tag}_auc_lo"] = t.loc[final, f"vs_{ref}_bar_ci_lo"]
            r[f"final_{tag}_auc_hi"] = t.loc[final, f"vs_{ref}_bar_ci_hi"]
            r[f"final_{tag}_auc_p"] = t.loc[final, f"vs_{ref}_bar_p_le_zero"]
            r[f"final_{tag}_brier_d"] = t.loc[final, f"vs_{ref}_bar_brier_delta"]
            r[f"final_{tag}_brier_lo"] = t.loc[final, f"vs_{ref}_bar_brier_ci_lo"]
            r[f"final_{tag}_brier_hi"] = t.loc[final, f"vs_{ref}_bar_brier_ci_hi"]
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "backbone_generality.csv", index=False)

    # ------------------------------------------------------------- cost
    reps = []
    for _ in range(7):
        t0 = time.perf_counter()
        f = lib.Fold(fit, stream, lo, frozen=fold.frozen,
                     memory=fold.memory)                   # retrieval included
        lib.all_arms(f, which={final})
        reps.append(time.perf_counter() - t0)
    reps_r = []
    for _ in range(7):
        t0 = time.perf_counter()
        f = lib.Fold(fit, stream, lo, frozen=fold.frozen,
                     memory=fold.memory)
        lib.all_arms(f, which={"respec"})
        reps_r.append(time.perf_counter() - t0)
    cost = {"final_arm": final, "backward_passes": 0,
            "n_bars": fold.bi.n_bars, "threads": 1,
            "final_ms_per_bar_median": 1e3 * float(np.median(reps)) / fold.bi.n_bars,
            "respec_ms_per_bar_median": 1e3 * float(np.median(reps_r)) / fold.bi.n_bars,
            "note": "wall clock of retrieval (3 spaces, 256-candidate pool) + "
                    "fast state + fusion for the whole test stream, median of 7, "
                    "divided by bars; frozen scoring excluded for both",
            "provenance_gates_max_abs_dauc": gates}
    stamp["test_read_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cost.update(stamp)
    (out_dir / "cost_and_gates_0923.json").write_text(json.dumps(cost, indent=2))
    log(json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()
