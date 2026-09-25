#!/usr/bin/env python3
"""R3-TTT residual (offset) fast state -- R3TTT track, 2026-09-21.

Question
--------
The shipped fast state REPLACES the frozen logit with an absolute local logit

    p = sigmoid( (1 - beta*rho) * l_e  +  beta*rho * logit(q_b) ),
    q_b = (sum_j w_j y_j + lam*pi0) / (W_b + lam).

Two consequences the manuscript already documents:
  * a shared bar-level fast state cannot reorder rows inside a bar (Sec. 4,
    asset-ordering remark), because l_e is scaled away rather than kept;
  * the simplest retrieval comparator (fixed-grid kNN-LM), which mixes in
    PROBABILITY space and therefore retains more of the frozen prediction,
    ranks at least as well on all seven backbones.

Proposed respecification: estimate the local BIAS of the frozen model with the
same kernel and the same shrinkage, and add it as an offset.

    qbar_b = (sum_j w_j phat_j + lam*pibar0) / (W_b + lam)     <- frozen, OOF
    delta_b = logit(q_b) - logit(qbar_b)
    p = sigmoid( l_e + beta*rho_b * delta_b )

phat_j is an OUT-OF-FOLD frozen probability for memory bar j, fitted with
contiguous calendar-day blocks inside the pre-deployment window only, so
Requirement 2 (immutability) still holds: nothing after the deployment
boundary is read.

Why this should be better, stated before measuring:
  (i)  if the frozen model is locally unbiased, delta_b -> 0 and the operator
       is a no-op; the absolute form injects error even then;
  (ii) l_e survives, so within-bar ordering is preserved rather than collapsed;
  (iii) the correction is on the model's residual, which is what the labels
       actually carry information about once the frozen prediction is given.

Protocol
--------
Stage `repro`  : reproduce the published frozen / shipped-72 test numbers.
Stage `val`    : every arm on VALIDATION ONLY (slow model + memory from train).
Stage `declare`: write protocol.json (hashed) from validation alone.
Stage `test`   : ONE test read of the declared arm and its references.

Thread pinning is load-bearing: top-k over near-tied distances is not
reproducible across BLAS thread counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/r3ttt-residual-mpl")

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common_protocol_clean as cpc                   # noqa: E402
import exp_ablation_sensitivity as eas                # noqa: E402

OUT = ROOT / "results" / "r3ttt_R3TTT_20260921"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
BLOCK_DAYS = 5
SEED = 42
SCALES = tuple(cpc.SCALES)                 # 8,16,32,64
SPACES = dict(cpc.RETRIEVAL_SPACES)        # timing, market_state, joint
TEMPS = (0.25, 1.0)
LAMS = (2.0, 8.0)
BETAS = (0.3, 0.5, 0.7)
OOF_FOLDS = 5

# Forward-window guard: the unspent forward window must stay unspent.
TEST_END = pd.Timestamp("2026-04-11")
FORWARD_START = pd.Timestamp("2026-04-16")

auc, brier, logloss = eas.auc, eas.brier, eas.logloss
BarIndex, paired_boot = eas.BarIndex, eas.paired_boot


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jdump(obj, name: str) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    log(f"  wrote {p.name}")
    return p


def save(df: pd.DataFrame, name: str) -> Path:
    p = OUT / name
    df.to_csv(p, index=False)
    log(f"  wrote {p.name}  ({len(df)} rows)")
    return p


# ===========================================================================
# panel, with the forward-window guard
# ===========================================================================
def panel_guarded() -> pd.DataFrame:
    df = cpc.load_event_panel()
    cpc.assert_clean(df, where="event panel")
    n_fwd = int((df["datetime"] >= FORWARD_START).sum())
    df = df[df["datetime"] < TEST_END].reset_index(drop=True)
    assert n_fwd == 0 or df["datetime"].max() < TEST_END
    log(f"panel {len(df)} rows, max datetime {df['datetime'].max()}")
    return df


# ===========================================================================
# out-of-fold frozen probability for the memory bars
# ===========================================================================
def oof_frozen_bar(fit_source: pd.DataFrame, *, folds: int = OOF_FOLDS) -> pd.DataFrame:
    """OOF frozen probability per aligned bar, inside the fit window only.

    Contiguous calendar-day blocks, so a bar's own day never trains its own
    prediction.  Returns a frame ``datetime, p_oof``.
    """
    src = fit_source.reset_index(drop=True)
    days = pd.to_datetime(src["datetime"]).dt.normalize()
    uniq = np.array(sorted(days.unique()))
    edges = np.array_split(np.arange(len(uniq)), folds)
    p = np.full(len(src), np.nan, dtype=float)
    for fi, block in enumerate(edges):
        held_days = set(uniq[block])
        held = days.isin(held_days).to_numpy()
        tr = src.loc[~held]
        if len(tr) == 0 or tr["target_hi_vol"].nunique() < 2:
            continue
        p[held] = cpc.slow_probability(tr, src.loc[held])
        log(f"  oof fold {fi+1}/{folds}: {int(held.sum())} rows held out")
    assert np.isfinite(p).all(), "OOF frozen probability has gaps"
    out = pd.DataFrame({"datetime": src["datetime"].to_numpy(), "p_oof": p})
    return out.groupby("datetime", as_index=False)["p_oof"].mean()


# ===========================================================================
# retrieval, shared by every arm
# ===========================================================================
class Retrieved:
    """Per-bar neighbour indices/distances in one routing space, plus the
    per-scale kernel aggregates every arm needs."""

    def __init__(self, memory, stream, groups, features):
        self.idx, self.dist = cpc._route(memory.frame, stream, groups, features)
        self.y = memory.labels[self.idx].astype(np.float64)
        self.finite = np.isfinite(self.dist)
        self.n_query, self.keep = self.y.shape
        self.position = np.arange(self.keep)[None, :]
        d0 = self.dist[:, 0].astype(np.float64)
        self.d_min = np.where(np.isfinite(d0), d0, 0.0)[:, None]

    def kernel(self, temperature: float) -> np.ndarray:
        gap = np.where(self.finite, self.dist.astype(np.float64) - self.d_min, np.inf)
        return np.exp(-np.clip(gap / temperature, 0.0, 700.0))

    def mask(self, scale: int) -> np.ndarray:
        return ((self.position < scale) & self.finite).astype(np.float64)


def local_moments(ret: Retrieved, temperature: float, mem_side: np.ndarray | None):
    """Per-scale retrieved mass, positive label mass and (optionally) frozen
    probability mass.  ``mem_side`` is a memory-aligned vector, e.g. p_oof."""
    weight = ret.kernel(temperature)
    side = None if mem_side is None else mem_side[ret.idx].astype(np.float64)
    out = {}
    for scale in SCALES:
        sel = weight * ret.mask(scale)
        mass = sel.sum(axis=1)
        pos = (sel * ret.y).sum(axis=1)
        s = None if side is None else (sel * side).sum(axis=1)
        out[scale] = (mass, pos, s)
    return out


# ===========================================================================
# the arms
# ===========================================================================
def arm_absolute(mom, prior_label, lam):
    """Shipped fast state: absolute local logit, averaged over scales.
    Returns (bias, consensus, spread)."""
    us, agrees = [], []
    for scale in SCALES:
        mass, pos, _ = mom[scale]
        share = np.where(mass > 0.0, pos / np.maximum(mass, 1e-12), prior_label)
        agrees.append(np.abs(share - 0.5) * 2.0)
        q = (pos + lam * prior_label) / (mass + lam)
        us.append(cpc.logit(q))
    u = np.stack(us, axis=0)
    return u.mean(axis=0), np.mean(np.stack(agrees, 0), 0), u.std(axis=0)


def arm_residual(mom, prior_label, prior_frozen, lam):
    """Residual fast state: matched-kernel local bias of the frozen model.
    Returns (delta, consensus, spread)."""
    ds, agrees = [], []
    for scale in SCALES:
        mass, pos, s = mom[scale]
        share = np.where(mass > 0.0, pos / np.maximum(mass, 1e-12), prior_label)
        agrees.append(np.abs(share - 0.5) * 2.0)
        q = (pos + lam * prior_label) / (mass + lam)
        qbar = (s + lam * prior_frozen) / (mass + lam)
        ds.append(cpc.logit(q) - cpc.logit(qbar))
    d = np.stack(ds, axis=0)
    return d.mean(axis=0), np.mean(np.stack(agrees, 0), 0), d.std(axis=0)


def arm_knnlm(mom, prior_label, lam):
    """Fixed-grid kNN-LM: probability-space mixture of the shrunk local vote."""
    qs = []
    for scale in SCALES:
        mass, pos, _ = mom[scale]
        qs.append((pos + lam * prior_label) / (mass + lam))
    return np.stack(qs, axis=0).mean(axis=0)


def rho_raw(consensus, spread):
    return consensus / (1.0 + spread)


def ecdf_transform(reference: np.ndarray):
    """Rank-normalising map built from a pre-deployment reference sample."""
    ref = np.sort(np.asarray(reference, dtype=float))

    def apply(x):
        return np.searchsorted(ref, np.asarray(x, dtype=float), side="right") / len(ref)

    return apply


# ===========================================================================
# stage machinery
# ===========================================================================
def build_environment(panel: pd.DataFrame, which: str):
    """`val`  -> fit on train, stream = validation, boundary 2025-10-01.
    `test` -> fit on pretest, stream = test, boundary 2026-01-01."""
    if which == "val":
        boundary = cpc.TRAIN_END
    elif which == "test":
        boundary = cpc.TEST_START
    else:
        raise ValueError(which)
    fit_source = panel[panel["datetime"] < boundary].reset_index(drop=True)
    if which == "val":
        stream = panel[(panel["datetime"] >= cpc.TRAIN_END)
                       & (panel["datetime"] < cpc.TEST_START)].reset_index(drop=True)
    else:
        stream = panel[panel["datetime"] >= cpc.TEST_START].reset_index(drop=True)
    log(f"env {which}: fit {len(fit_source)} rows, stream {len(stream)} rows")

    frozen = cpc.slow_probability(fit_source, stream)
    memory = cpc.build_bar_memory(fit_source, deploy_start=boundary)
    log(f"env {which}: memory {len(memory)} bars, frozen AUC(row) "
        f"{auc(stream['target_hi_vol'], frozen):.4f}")

    oof = oof_frozen_bar(fit_source)
    mem_p = memory.frame[["datetime"]].merge(oof, on="datetime", how="left")["p_oof"]
    assert mem_p.notna().all(), "memory bar without an OOF frozen probability"
    mem_p = mem_p.to_numpy(dtype=float)

    groups = cpc.bar_groups(stream)
    bi = BarIndex(stream)
    return dict(which=which, boundary=boundary, fit=fit_source, stream=stream,
                frozen=frozen, memory=memory, mem_p_oof=mem_p, groups=groups, bi=bi)


def retrieve_all(env):
    return {name: Retrieved(env["memory"], env["stream"], env["groups"], feats)
            for name, feats in SPACES.items()}


def loo_rho_reference(env, retrieved_spaces):
    """Rank-normalising reference for rho, built from pre-deployment queries
    only: every memory bar is used as a query against the memory.  Nothing
    after the deployment boundary is read."""
    mem_frame = env["memory"].frame
    groups = [np.array([i]) for i in range(len(mem_frame))]
    ref = {}
    for name, feats in SPACES.items():
        r = Retrieved(env["memory"], mem_frame, groups, feats)
        prior = float(env["memory"].labels.mean())
        vals = []
        for T in TEMPS:
            mom = local_moments(r, T, None)
            for lam in LAMS:
                _, cons, spr = arm_absolute(mom, prior, lam)
                vals.append(rho_raw(cons, spr))
        ref[name] = np.concatenate(vals)
    return ref


def score_grid(env, retrieved, rho_ref, *, arms, betas=BETAS):
    """Every (arm, space, T, lam, beta, gate) member, plus grid averages."""
    frozen = env["frozen"]
    lfrozen = cpc.logit(frozen)
    bi = env["bi"]
    row_bar = bi.row_bar
    prior_label = float(env["memory"].labels.mean())
    prior_frozen = float(env["mem_p_oof"].mean())

    members: dict[str, list[np.ndarray]] = {a: [] for a in arms}
    rows = []
    for space, ret in retrieved.items():
        ecdf = ecdf_transform(rho_ref[space])
        for T in TEMPS:
            mom_lab = local_moments(ret, T, None)
            mom_res = local_moments(ret, T, env["mem_p_oof"])
            for lam in LAMS:
                u_abs, cons, spr = arm_absolute(mom_lab, prior_label, lam)
                d_res, cons_r, spr_r = arm_residual(mom_res, prior_label,
                                                    prior_frozen, lam)
                q_knn = arm_knnlm(mom_lab, prior_label, lam)
                rho_a = rho_raw(cons, spr)
                rho_r = rho_raw(cons_r, spr_r)
                rho_a_rn = ecdf(rho_a)
                gates = {"off": np.ones(ret.n_query), "raw": rho_a,
                         "rank": rho_a_rn}
                gates_r = {"off": np.ones(ret.n_query), "raw": rho_r,
                           "rank": ecdf(rho_r)}
                for beta in betas:
                    for gname in ("off", "raw", "rank"):
                        g = gates[gname][row_bar]
                        gr = gates_r[gname][row_bar]
                        eff = beta * g
                        eff_r = beta * gr
                        cand = {}
                        if "absolute" in arms:
                            cand["absolute"] = cpc.sigmoid(
                                (1.0 - eff) * lfrozen + eff * u_abs[row_bar])
                        if "residual" in arms:
                            cand["residual"] = cpc.sigmoid(
                                lfrozen + eff_r * d_res[row_bar])
                        if "knnlm" in arms:
                            cand["knnlm"] = ((1.0 - eff) * frozen
                                             + eff * q_knn[row_bar])
                        for a, p in cand.items():
                            members[a].append(p)
                            s = bi.scores(p)
                            rows.append(dict(arm=a, space=space, temperature=T,
                                             prior_strength=lam, beta=beta,
                                             gate=gname, **s))
    return pd.DataFrame(rows), members


def grid_average(members, keys=None):
    """Uniform probability average over a member subset."""
    stack = np.stack(members, axis=0)
    return stack.mean(axis=0)


def subset(df_rows, members, arm, **where):
    """Indices of grid members matching a filter, in df row order."""
    mask = df_rows["arm"] == arm
    for col, allowed in where.items():
        if allowed is None:
            continue
        allowed = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        mask &= df_rows[col].isin(list(allowed))
    local = df_rows.index[mask].to_numpy()
    # member list for an arm is in the same order as its rows in df_rows
    arm_rows = df_rows.index[df_rows["arm"] == arm].to_numpy()
    pos = {r: i for i, r in enumerate(arm_rows)}
    return [members[arm][pos[r]] for r in local]


# ===========================================================================
# stages
# ===========================================================================
def stage_repro(panel):
    log("STAGE repro -- reproduce the published headline")
    run = cpc.headline_event_run(n_boot=200, panel=panel)
    bi = BarIndex(run["test"])
    out = {
        "frozen_auc_row": run["frozen_auc"], "sf_auc_row": run["sf_auc"],
        "frozen_auc_bar": auc(bi.y_bar, bi.pool(run["frozen"])),
        "sf_auc_bar": auc(bi.y_bar, bi.pool(run["sf"])),
        "frozen_brier_bar": brier(bi.y_bar, bi.pool(run["frozen"])),
        "sf_brier_bar": brier(bi.y_bar, bi.pool(run["sf"])),
        "n_members": run["n_members"], "n_memory": run["n_memory"],
        "n_test": run["n_test"], "n_test_bars": bi.n_bars,
        "published_frozen_auc_bar": 0.7679, "published_sf_auc_bar": 0.7800,
        "published_frozen_auc_row": 0.8138, "published_sf_auc_row": 0.8283,
    }
    out["match_bar"] = bool(abs(out["sf_auc_bar"] - 0.7800) < 2e-3
                            and abs(out["frozen_auc_bar"] - 0.7679) < 2e-3)
    jdump(out, "repro.json")
    log(f"  frozen bar {out['frozen_auc_bar']:.4f} (pub 0.7679) | "
        f"SF bar {out['sf_auc_bar']:.4f} (pub 0.7800) | match={out['match_bar']}")
    return out


def stage_val(panel):
    log("STAGE val -- VALIDATION ONLY, no test row is read")
    env = build_environment(panel, "val")
    retrieved = retrieve_all(env)
    rho_ref = loo_rho_reference(env, retrieved)
    jdump({k: {"n": len(v), "mean": float(v.mean()), "max": float(v.max())}
           for k, v in rho_ref.items()}, "rho_reference_val.json")

    rows, members = score_grid(env, retrieved, rho_ref,
                               arms=("absolute", "residual", "knnlm"))
    save(rows, "grid_validation.csv")

    bi = env["bi"]
    frozen = env["frozen"]

    # named arms -----------------------------------------------------------
    named = {
        "frozen": frozen,
        "shipped72": grid_average(subset(rows, members, "absolute",
                                         gate=["off", "raw"], beta=BETAS)),
        "respec": grid_average(subset(rows, members, "absolute",
                                      gate="off", beta=0.5)),
        "absolute_rank": grid_average(subset(rows, members, "absolute",
                                             gate="rank", beta=0.5)),
        "residual_off": grid_average(subset(rows, members, "residual",
                                            gate="off", beta=0.5)),
        "residual_rank": grid_average(subset(rows, members, "residual",
                                             gate="rank", beta=0.5)),
        "residual_raw": grid_average(subset(rows, members, "residual",
                                            gate="raw", beta=0.5)),
        "residual_grid": grid_average(subset(rows, members, "residual",
                                             gate="off", beta=BETAS)),
        "knnlm_off": grid_average(subset(rows, members, "knnlm",
                                         gate="off", beta=0.5)),
        "knnlm_grid": grid_average(subset(rows, members, "knnlm",
                                          gate="off", beta=BETAS)),
    }
    np.savez_compressed(OUT / "predictions_validation.npz",
                        y_row=bi.y_row, y_bar=bi.y_bar, **named)

    recs = []
    for name, p in named.items():
        s = bi.scores(p)
        rec = dict(arm=name, n_members=1, **s)
        for ref_name in ("frozen", "respec", "knnlm_off"):
            if name == ref_name:
                continue
            c = eas.contrast(bi, p, named[ref_name], n_boot=N_BOOT,
                             prefix=f"vs_{ref_name}_")
            rec.update(c)
        recs.append(rec)
    val = pd.DataFrame(recs).sort_values("auc_bar", ascending=False)
    save(val, "named_validation.csv")
    log("\n" + val[["arm", "auc_bar", "brier_bar", "vs_respec_bar_delta",
                    "vs_respec_bar_ci_lo", "vs_respec_bar_p_le_zero"]]
        .to_string(index=False))
    return val


def stage_declare():
    log("STAGE declare -- pre-register the test configuration from validation")
    val = pd.read_csv(OUT / "named_validation.csv").set_index("arm")
    candidates = ["residual_off", "residual_rank", "residual_grid",
                  "absolute_rank", "respec"]
    table = val.loc[[c for c in candidates if c in val.index]]
    # Pre-registered rule: highest validation bar AUC among the candidates,
    # required to beat `respec` with P(delta<=0) < 0.05 on validation; if no
    # candidate clears the gate, the declared arm is `respec` itself.
    gate = table.copy()
    if "vs_respec_bar_p_le_zero" in gate.columns:
        ok = gate[(gate.index != "respec")
                  & (gate["vs_respec_bar_p_le_zero"] < 0.05)]
    else:
        ok = gate.iloc[0:0]
    declared = ok["auc_bar"].idxmax() if len(ok) else "respec"
    proto = {
        "track": "r3ttt_R3TTT_20260921",
        "declared_arm": declared,
        "selection_rule": ("highest validation bar AUC among the candidate arms "
                           "with P(delta<=0) < 0.05 against `respec`; fallback "
                           "`respec`"),
        "candidates": candidates,
        "validation_auc_bar": {k: float(val.loc[k, "auc_bar"])
                               for k in table.index},
        "validation_p_vs_respec": {
            k: (None if k == "respec"
                else float(val.loc[k, "vs_respec_bar_p_le_zero"]))
            for k in table.index},
        "test_references": ["frozen", "shipped72", "respec", "knnlm_off",
                            "knnlm_grid"],
        "primary_metric": "per-bar AUC (bar score = mean of the arm's per-post "
                          "probabilities)",
        "uncertainty": f"paired {N_BOOT}-replicate {BLOCK_DAYS}-day moving-block "
                       f"bootstrap clustered by calendar day, seed {SEED}",
        "test_reads_permitted": 1,
        "forward_window_untouched": str(FORWARD_START),
        "threads_pinned": 1,
    }
    blob = json.dumps(proto, sort_keys=True).encode()
    proto["sha256"] = hashlib.sha256(blob).hexdigest()
    jdump(proto, "protocol.json")
    log(f"  DECLARED: {declared}  sha256:{proto['sha256'][:12]}")
    return proto


def stage_test(panel):
    log("STAGE test -- ONE read")
    proto = json.loads((OUT / "protocol.json").read_text())
    declared = proto["declared_arm"]
    log(f"  declared arm: {declared} (protocol sha256:{proto['sha256'][:12]})")

    env = build_environment(panel, "test")
    retrieved = retrieve_all(env)
    rho_ref = loo_rho_reference(env, retrieved)
    rows, members = score_grid(env, retrieved, rho_ref,
                               arms=("absolute", "residual", "knnlm"))
    save(rows, "grid_test.csv")

    bi = env["bi"]
    named = {
        "frozen": env["frozen"],
        "shipped72": grid_average(subset(rows, members, "absolute",
                                         gate=["off", "raw"], beta=BETAS)),
        "respec": grid_average(subset(rows, members, "absolute",
                                      gate="off", beta=0.5)),
        "absolute_rank": grid_average(subset(rows, members, "absolute",
                                             gate="rank", beta=0.5)),
        "residual_off": grid_average(subset(rows, members, "residual",
                                            gate="off", beta=0.5)),
        "residual_rank": grid_average(subset(rows, members, "residual",
                                             gate="rank", beta=0.5)),
        "residual_grid": grid_average(subset(rows, members, "residual",
                                             gate="off", beta=BETAS)),
        "knnlm_off": grid_average(subset(rows, members, "knnlm",
                                         gate="off", beta=0.5)),
        "knnlm_grid": grid_average(subset(rows, members, "knnlm",
                                          gate="off", beta=BETAS)),
    }
    np.savez_compressed(OUT / "predictions_test.npz",
                        y_row=bi.y_row, y_bar=bi.y_bar, **named)

    recs = []
    for name, p in named.items():
        s = bi.scores(p)
        rec = dict(arm=name, declared=bool(name == declared), **s)
        for ref in ("frozen", "shipped72", "respec", "knnlm_off", "knnlm_grid"):
            if name == ref:
                continue
            rec.update(eas.contrast(bi, p, named[ref], n_boot=N_BOOT,
                                    prefix=f"vs_{ref}_"))
        recs.append(rec)
    test = pd.DataFrame(recs).sort_values("auc_bar", ascending=False)
    save(test, "named_test.csv")
    log("\n" + test[["arm", "declared", "auc_bar", "brier_bar",
                     "vs_frozen_bar_delta", "vs_knnlm_off_bar_delta",
                     "vs_knnlm_off_bar_ci_lo", "vs_knnlm_off_bar_p_le_zero"]]
        .to_string(index=False))
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["repro", "val", "declare", "test", "all"])
    args = ap.parse_args()
    panel = panel_guarded()
    if args.stage in ("repro", "all"):
        stage_repro(panel)
    if args.stage in ("val", "all"):
        stage_val(panel)
    if args.stage in ("declare", "all"):
        stage_declare()
    if args.stage in ("test", "all"):
        stage_test(panel)
    log("done")


if __name__ == "__main__":
    main()
