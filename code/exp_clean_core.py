#!/usr/bin/env python3
"""Core claims of the R3-TTT paper, re-established on the CLEAN panel.

Every number this track produces is computed inside ONE environment: the
look-ahead-free panel exposed by ``common_protocol_clean``.  Nothing here is
ever differenced against a contaminated-panel point estimate.

Stages
------
  protocol   protocol.json + panel census
  main       main comparison table (frozen x2, periodic retraining, rolling GBT,
             online logistic calibration, unlabelled prior shift, Tent-style
             entropy, retrieval-only, validation-selected R3-TTT, R3-TTT-SF),
             every row with a paired day-clustered bootstrap interval against SF
  mech       mechanism controls: query-state permutation (>=5000), memory-label
             shuffle, global/unrouted fast state, random routing (>=20 seeds),
             random-memory draw, uniform weights
  delay      label-delay audit at 0 / 1 / 4 / 24 hours
  abl        ablations: class-balanced retrieval on/off, bar dedup on/off
  dm         Diebold-Mariano with HAC on Brier and log loss
  e2         the double label-shuffle control, diagnosed rather than waved
  anchor     capacity sweep of the frozen anchor, selected on validation
  repro      determinism audit: BLAS thread count x memory-order tie band
  fig        PDF figures

Usage
-----
  python exp_clean_core.py --help
  python exp_clean_core.py --stage all --jobs 48
  python exp_clean_core.py --stage main
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/r3ttt-clean-core-mpl")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import scipy.stats as sps
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common_protocol_clean as cpc  # noqa: E402

OUT = ROOT / "results" / "clean_core"
CACHE = OUT / "_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
BLOCK_DAYS = 5
SEED = 42
N_PERM = 5000          # query-state permutation replicates
N_SHUFFLE = 200        # label-shuffle replicates
N_ROUTE_SEEDS = 25     # random-routing / random-memory seeds
N_NOISE_REPS = 10      # replicates per label-noise rate


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def jdump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value))


# =============================================================================
# panel / context
# =============================================================================
def context(delay_hours: float = 1.0) -> dict:
    """Panel, splits and pre-deployment frame at a given label delay."""
    panel = cpc.load_event_panel()
    cpc.assert_clean(panel, where="event panel")
    if float(delay_hours) != 1.0:
        panel = panel.copy()
        panel["available_time"] = panel["datetime"] + pd.Timedelta(hours=float(delay_hours))
    train, validation, test = cpc.split(panel)
    pretest = panel[panel["datetime"] < cpc.TEST_START].reset_index(drop=True)
    return {
        "panel": panel, "train": train, "validation": validation,
        "test": test, "pretest": pretest, "delay_hours": float(delay_hours),
    }


def loss_brier(y, p):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def loss_log(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def auc(y, p):
    y = np.asarray(y)
    return cpc.fast_binary_auc(y, p) if np.unique(y).size == 2 else float("nan")


# =============================================================================
# retrieval with pluggable failure modes
# =============================================================================
def route(memory_frame: pd.DataFrame, stream: pd.DataFrame, groups, features,
          *, max_k: int = cpc.MAX_K, mode: str = "nearest", seed: int = 0):
    """``cpc._route`` plus the two destruction modes the controls need.

    mode='nearest'        verbatim reproduction of cpc._route
    mode='random_space'   routing coordinates replaced by i.i.d. N(0,1) for BOTH
                          memory and stream: the router is real, the space is noise
    mode='random_memory'  a uniformly random subset of memory replaces the k
                          nearest; their TRUE distances are kept for weighting
    """
    features = list(features)
    mem_raw = memory_frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stream_raw = stream[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if mode == "random_space":
        rng = np.random.default_rng(seed)
        mem_values = rng.standard_normal((len(mem_raw), len(features)))
        stream_values = rng.standard_normal((len(stream_raw), len(features)))
    else:
        mem_values = mem_raw.to_numpy(float)
        stream_values = stream_raw.to_numpy(float)
    scaler = StandardScaler().fit(mem_values)
    memory_x = scaler.transform(mem_values)
    stream_x = scaler.transform(stream_values)
    queries = np.stack([stream_x[locs].mean(axis=0) for locs in groups], axis=0)
    keep = min(max_k, len(memory_x))
    d = (
        np.einsum("ij,ij->i", queries, queries)[:, None]
        - 2.0 * (queries @ memory_x.T)
        + np.einsum("ij,ij->i", memory_x, memory_x)[None, :]
    ) / float(len(features))
    np.maximum(d, 0.0, out=d)
    if mode == "random_memory":
        rng = np.random.default_rng(seed)
        local = np.stack(
            [rng.choice(len(memory_x), size=keep, replace=False) for _ in range(len(queries))]
        )
    else:
        local = np.argpartition(d, keep - 1, axis=1)[:, :keep]
    local_d = np.take_along_axis(d, local, axis=1)
    order = np.argsort(local_d, axis=1, kind="stable")
    return (np.take_along_axis(local, order, axis=1),
            np.take_along_axis(local_d, order, axis=1))


def collect_states(memory, stream, groups, *, mode="nearest", seed=0,
                   balanced=cpc.BALANCED_DEFAULT, scales=cpc.SCALES,
                   temperatures=cpc.TEMPERATURES, prior_strengths=cpc.PRIOR_STRENGTHS,
                   gates=cpc.GATES, spaces=None, max_k=cpc.MAX_K):
    """Per-bar (bias, trust) for every (space, T, lambda, gate) member."""
    spaces = cpc.RETRIEVAL_SPACES if spaces is None else spaces
    out = []
    for name, features in spaces.items():
        idx, dist = route(memory.frame, stream, groups, features, mode=mode,
                          seed=seed, max_k=max_k)
        states = cpc._fast_states(
            idx, dist, memory.labels, balanced=balanced, scales=scales,
            temperatures=temperatures, prior_strengths=prior_strengths, gates=gates,
        )
        for (temperature, prior, gate), (bias_bar, trust_bar) in states.items():
            out.append({"space": name, "temperature": temperature,
                        "prior_strength": prior, "gate": bool(gate),
                        "bias_bar": bias_bar, "trust_bar": trust_bar})
    return out


def assemble(states, frozen, row_bar, *, blends=cpc.BLENDS, bar_permutation=None):
    """Uniform average over states x blends.  ``bar_permutation`` implements the
    query-state permutation control: bar i is handed bar perm[i]'s fast state."""
    total = np.zeros(len(frozen), float)
    count = 0
    slow_logit = cpc.logit(frozen)
    for state in states:
        bias_bar, trust_bar = state["bias_bar"], state["trust_bar"]
        if bar_permutation is not None:
            bias_bar = bias_bar[bar_permutation]
            trust_bar = trust_bar[bar_permutation]
        bias = bias_bar[row_bar]
        trust = trust_bar[row_bar]
        for beta in blends:
            effective = beta * trust
            total += cpc.sigmoid((1.0 - effective) * slow_logit + effective * bias)
            count += 1
    return total / count, count


def row_bar_index(stream, groups):
    row_bar = np.empty(len(stream), int)
    for position, locs in enumerate(groups):
        row_bar[locs] = position
    return row_bar


# =============================================================================
# baselines
# =============================================================================
def stream_groups(stream: pd.DataFrame):
    for _, positions in sorted(stream.groupby("datetime", sort=True).indices.items()):
        locs = np.asarray(positions, int)
        yield locs, stream.iloc[locs[0]]["datetime"]


def matured_positions(stream: pd.DataFrame, now) -> np.ndarray:
    return np.flatnonzero(stream["available_time"].to_numpy() <= np.datetime64(now))


def _fit(frame, seed, trees=cpc.SLOW_TREES, depth=cpc.SLOW_DEPTH):
    x = frame[cpc.MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return GradientBoostingClassifier(
        n_estimators=trees, max_depth=depth, random_state=seed
    ).fit(x, frame["target_hi_vol"].to_numpy(int))


def _predict(models, frame):
    x = frame[cpc.MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)


def periodic_retraining(initial, stream, *, frequency_days, rolling_events,
                        seeds=cpc.SLOW_SEEDS):
    """Refit the SAME backbone every ``frequency_days`` on all matured labels."""
    output = np.empty(len(stream), float)
    models = [_fit(initial, s) for s in seeds]
    last_day = None
    for locs, now in stream_groups(stream):
        day = now.normalize()
        if last_day is None or (day - last_day).days >= frequency_days:
            mature = matured_positions(stream, now)
            data = pd.concat([initial, stream.iloc[mature]], ignore_index=True)
            data = data.sort_values(["datetime", "event_time_et", "post_id"])
            if rolling_events is not None:
                data = data.tail(rolling_events)
            if data["target_hi_vol"].nunique() == 2:
                models = [_fit(data, s) for s in seeds]
            last_day = day
        output[locs] = _predict(models, stream.iloc[locs])
    return output


def online_logistic_calibration(stream, base, *, window, regularization_c):
    output = np.empty(len(stream), float)
    base_logit = np.clip(cpc.logit(base), -8.0, 8.0).reshape(-1, 1)
    for locs, now in stream_groups(stream):
        mature = matured_positions(stream, now)[-window:]
        if len(mature) >= 16 and stream.iloc[mature]["target_hi_vol"].nunique() == 2:
            model = LogisticRegression(C=regularization_c, max_iter=200,
                                       solver="liblinear", random_state=0
                                       ).fit(base_logit[mature],
                                             stream.iloc[mature]["target_hi_vol"])
            output[locs] = model.predict_proba(base_logit[locs])[:, 1]
        else:
            output[locs] = base[locs]
    return output


def unlabeled_prior_shift(initial, stream, base, *, window, strength):
    output = np.empty(len(stream), float)
    training_prior = float(initial["target_hi_vol"].mean())
    observed: list[int] = []
    for locs, _ in stream_groups(stream):
        current = np.r_[np.asarray(observed[-window:], int), locs]
        predicted_prior = float(np.mean(base[current]))
        shift = cpc.logit(training_prior) - cpc.logit(predicted_prior)
        output[locs] = cpc.sigmoid(cpc.logit(base[locs]) + strength * shift)
        observed.extend(locs.tolist())
    return output


def tent_entropy(stream, base, *, window, learning_rate, anchor_strength):
    output = np.empty(len(stream), float)
    base_logit = np.clip(cpc.logit(base), -8.0, 8.0)
    observed: list[int] = []
    scale, bias = 1.0, 0.0
    for locs, _ in stream_groups(stream):
        use = np.r_[np.asarray(observed[-window:], int), locs]
        z = base_logit[use]
        adapted = np.clip(scale * z + bias, -12.0, 12.0)
        probability = cpc.sigmoid(adapted)
        gradient = -adapted * probability * (1.0 - probability)
        grad_scale = float(np.mean(gradient * z)) + anchor_strength * (scale - 1.0)
        grad_bias = float(np.mean(gradient)) + anchor_strength * bias
        scale = float(np.clip(scale - learning_rate * grad_scale, 0.1, 5.0))
        bias = float(np.clip(bias - learning_rate * grad_bias, -2.0, 2.0))
        output[locs] = cpc.sigmoid(scale * base_logit[locs] + bias)
        observed.extend(locs.tolist())
    return output


def retrieval_only(states, row_bar, n_rows):
    """The fast state with the frozen anchor removed: sigmoid(u), no fusion."""
    distinct = {}
    for state in states:
        key = (state["space"], state["temperature"], state["prior_strength"])
        distinct.setdefault(key, state["bias_bar"])
    total = np.zeros(n_rows, float)
    for bias_bar in distinct.values():
        total += cpc.sigmoid(bias_bar[row_bar])
    return total / len(distinct), len(distinct)


# =============================================================================
# stage: protocol
# =============================================================================
def stage_protocol():
    log("stage protocol")
    ctx = context()
    memory = cpc.build_bar_memory(ctx["pretest"], deploy_start=cpc.TEST_START)
    record = cpc.protocol_json(
        track="clean_core",
        question="every claim the paper's main argument rests on, re-run inside "
                 "the clean (look-ahead-free) environment",
        module="common_protocol_clean",
        bootstrap={"n_boot": N_BOOT, "block_days": BLOCK_DAYS, "seed": SEED,
                   "clustered_by": "calendar day"},
        replicates={"query_state_permutations": N_PERM,
                    "label_shuffles": N_SHUFFLE,
                    "random_routing_seeds": N_ROUTE_SEEDS,
                    "label_noise_reps_per_rate": N_NOISE_REPS},
        census={
            "n_pretest": int(len(ctx["pretest"])),
            "n_train": int(len(ctx["train"])),
            "n_validation": int(len(ctx["validation"])),
            "n_test": int(len(ctx["test"])),
            "n_test_bars": int(ctx["test"]["datetime"].nunique()),
            "n_test_days": int(ctx["test"]["datetime"].dt.normalize().nunique()),
            "n_memory_bars": int(len(memory)),
            "test_positive_rate": float(ctx["test"]["target_hi_vol"].mean()),
        },
        test_split_touched="once, at the end of every stage; no tuning on test",
    )
    jdump(record, OUT / "protocol.json")
    log(f"  census {record['census']}")
    return record


# =============================================================================
# stage: main comparison table
# =============================================================================
GRID_PERIODIC = [dict(frequency_days=f, rolling_events=None) for f in (1, 7, 14, 30)]
GRID_ROLLING = [dict(frequency_days=f, rolling_events=w)
                for f in (1, 7, 14) for w in (1000, 2500, 5000)]
GRID_CALIB = [dict(window=w, regularization_c=c)
              for w in (128, 256, 512, 1024) for c in (0.1, 1.0, 10.0)]
GRID_PRIOR = [dict(window=w, strength=s)
              for w in (64, 128, 256, 512) for s in (0.25, 0.5, 1.0)]
GRID_TENT = [dict(window=w, learning_rate=lr, anchor_strength=a)
             for w in (64, 128, 256) for lr in (0.01, 0.05, 0.1) for a in (0.01, 0.1)]


def _val_periodic(cfg, ctx):
    p = periodic_retraining(ctx["train"], ctx["validation"], **cfg)
    return cfg, auc(ctx["validation"]["target_hi_vol"], p)


def r3ttt_config_grid():
    """The pre-specified single-configuration universe a selector could search:
    3 spaces x (4 single k + the k-ensemble) x 3 beta x 2 T x 2 lambda x 2 gate."""
    grid = []
    for space in cpc.RETRIEVAL_SPACES:
        for k in list(cpc.SCALES) + ["ensemble"]:
            for beta in cpc.BLENDS:
                for temperature in cpc.TEMPERATURES:
                    for prior in cpc.PRIOR_STRENGTHS:
                        for gate in cpc.GATES:
                            grid.append(dict(space=space, k=k, beta=beta,
                                             temperature=temperature,
                                             prior_strength=prior, gate=gate))
    return grid


def score_config_grid(memory, stream, frozen, groups, row_bar):
    """AUC of every single R3-TTT configuration on one split."""
    scales_all = tuple(cpc.SCALES)
    y = stream["target_hi_vol"].to_numpy(int)
    rows = []
    for space, features in cpc.RETRIEVAL_SPACES.items():
        idx, dist = route(memory.frame, stream, groups, features)
        for k in list(cpc.SCALES) + ["ensemble"]:
            scales = scales_all if k == "ensemble" else (k,)
            states = cpc._fast_states(idx, dist, memory.labels,
                                      balanced=cpc.BALANCED_DEFAULT, scales=scales,
                                      temperatures=cpc.TEMPERATURES,
                                      prior_strengths=cpc.PRIOR_STRENGTHS,
                                      gates=cpc.GATES)
            for (temperature, prior, gate), (bias_bar, trust_bar) in states.items():
                bias = bias_bar[row_bar]
                trust = trust_bar[row_bar]
                for beta in cpc.BLENDS:
                    p = cpc.fuse(frozen, bias, trust, beta)
                    rows.append({"space": space, "k": str(k), "beta": beta,
                                 "temperature": temperature, "prior_strength": prior,
                                 "gate": bool(gate), "auc": auc(y, p)})
    return pd.DataFrame(rows)


def stage_main(jobs: int, n_boot: int):
    log("stage main")
    ctx = context()
    train, validation, test, pretest = ctx["train"], ctx["validation"], ctx["test"], ctx["pretest"]
    y_test = test["target_hi_vol"].to_numpy(int)
    y_val = validation["target_hi_vol"].to_numpy(int)

    # ---- validation arm: slow model on TRAIN only -------------------------
    log("  validation arm")
    frozen_val = cpc.slow_probability(train, validation)
    memory_val = cpc.build_bar_memory(train, deploy_start=cpc.TRAIN_END)
    groups_val = cpc.bar_groups(validation)
    row_bar_val = row_bar_index(validation, groups_val)

    choice_cache = CACHE / "validation_choices.json"
    grid_cache = CACHE / "validation_config_grid.csv"
    chosen = {}
    if choice_cache.exists() and grid_cache.exists():
        log("    reusing cached validation arm")
        cached = json.loads(choice_cache.read_text())
        chosen = {k: (v["config"], v["validation_auc"]) for k, v in cached.items()}
        val_grid = pd.read_csv(grid_cache)
        val_grid["k"] = val_grid["k"].astype(str)
        val_grid["gate"] = val_grid["gate"].astype(bool)
    else:
        log("    periodic / rolling grids")
        results = Parallel(n_jobs=jobs)(
            delayed(_val_periodic)(cfg, ctx) for cfg in GRID_PERIODIC + GRID_ROLLING
        )
        periodic_scores = [(c, a) for c, a in results if c["rolling_events"] is None]
        rolling_scores = [(c, a) for c, a in results if c["rolling_events"] is not None]
        chosen["periodic_retraining"] = max(periodic_scores, key=lambda t: t[1])
        chosen["rolling_gbt"] = max(rolling_scores, key=lambda t: t[1])

        log("    calibration / prior-shift / tent grids")
        chosen["online_logistic_calibration"] = max(
            ((c, auc(y_val, online_logistic_calibration(validation, frozen_val, **c)))
             for c in GRID_CALIB), key=lambda t: t[1])
        chosen["unlabelled_prior_shift"] = max(
            ((c, auc(y_val, unlabeled_prior_shift(train, validation, frozen_val, **c)))
             for c in GRID_PRIOR), key=lambda t: t[1])
        chosen["tent_entropy"] = max(
            ((c, auc(y_val, tent_entropy(validation, frozen_val, **c)))
             for c in GRID_TENT), key=lambda t: t[1])

        log("    R3-TTT configuration grid on validation")
        val_grid = score_config_grid(memory_val, validation, frozen_val,
                                     groups_val, row_bar_val)
        val_grid = val_grid.rename(columns={"auc": "validation_auc"})
        val_grid.to_csv(grid_cache, index=False)
        jdump({k: {"config": v[0], "validation_auc": v[1]} for k, v in chosen.items()},
              choice_cache)
    best_row = val_grid.loc[val_grid["validation_auc"].idxmax()]
    chosen["r3ttt_validation_selected"] = (
        {k: best_row[k] for k in ("space", "k", "beta", "temperature",
                                  "prior_strength", "gate")},
        float(best_row["validation_auc"]),
    )
    for name, (cfg, score) in chosen.items():
        log(f"    {name}: {cfg} val_auc={score:.4f}")

    # ---- test arm ---------------------------------------------------------
    log("  test arm")
    frozen = cpc.slow_probability(pretest, test)
    frozen_single = cpc.slow_probability(pretest, test, seeds=(0,))
    memory = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    groups = cpc.bar_groups(test)
    row_bar = row_bar_index(test, groups)
    states = collect_states(memory, test, groups)
    sf, n_members = assemble(states, frozen, row_bar)

    reference, n_ref = cpc.r3ttt_selection_free(memory, test, frozen)[0], 72
    assert np.max(np.abs(sf - reference)) < 1e-12, "SF re-implementation drifted"
    log(f"    SF reproduces cpc.r3ttt_selection_free to {np.max(np.abs(sf-reference)):.1e}")

    predictions = {
        "frozen_single_seed": frozen_single,
        "frozen_seed_averaged": frozen,
        "periodic_retraining": periodic_retraining(pretest, test,
                                                   **chosen["periodic_retraining"][0]),
        "rolling_gbt": periodic_retraining(pretest, test, **chosen["rolling_gbt"][0]),
        "online_logistic_calibration": online_logistic_calibration(
            test, frozen, **chosen["online_logistic_calibration"][0]),
        "unlabelled_prior_shift": unlabeled_prior_shift(
            pretest, test, frozen, **chosen["unlabelled_prior_shift"][0]),
        "tent_entropy": tent_entropy(test, frozen, **chosen["tent_entropy"][0]),
    }
    ronly, n_ronly = retrieval_only(states, row_bar, len(test))
    predictions["retrieval_only"] = ronly

    cfg = chosen["r3ttt_validation_selected"][0]
    scales = tuple(cpc.SCALES) if cfg["k"] == "ensemble" else (int(cfg["k"]),)
    idx_sel, dist_sel = route(memory.frame, test, groups, cpc.RETRIEVAL_SPACES[cfg["space"]])
    sel_states = cpc._fast_states(idx_sel, dist_sel, memory.labels,
                                  balanced=cpc.BALANCED_DEFAULT, scales=scales,
                                  temperatures=(cfg["temperature"],),
                                  prior_strengths=(cfg["prior_strength"],),
                                  gates=(bool(cfg["gate"]),))
    bias_bar, trust_bar = sel_states[(cfg["temperature"], cfg["prior_strength"], bool(cfg["gate"]))]
    predictions["r3ttt_validation_selected"] = cpc.fuse(
        frozen, bias_bar[row_bar], trust_bar[row_bar], cfg["beta"])
    predictions["r3ttt_selection_free"] = sf

    # ---- anchor sweep: where does VALIDATION put the blend weight? --------
    log("  blend sweep on validation and test")
    states_val = collect_states(memory_val, validation, groups_val)
    ronly_val, _ = retrieval_only(states_val, row_bar_val, len(validation))
    sf_val, _ = assemble(states_val, frozen_val, row_bar_val)
    blend_rows = []
    for beta in np.round(np.arange(0.0, 1.0001, 0.1), 3):
        p_val, _ = assemble(states_val, frozen_val, row_bar_val, blends=(float(beta),))
        p_test, _ = assemble(states, frozen, row_bar, blends=(float(beta),))
        blend_rows.append({"beta": float(beta),
                           "validation_auc": auc(y_val, p_val),
                           "test_auc": auc(y_test, p_test)})
    blend = pd.DataFrame(blend_rows)
    blend.to_csv(OUT / "blend_sweep.csv", index=False)
    log("\n" + blend.to_string(index=False))
    anchor = {
        "frozen_validation_auc": auc(y_val, frozen_val),
        "selection_free_validation_auc": auc(y_val, sf_val),
        "retrieval_only_validation_auc": auc(y_val, ronly_val),
        "validation_argmax_beta": float(blend.loc[blend["validation_auc"].idxmax(), "beta"]),
        "test_argmax_beta": float(blend.loc[blend["test_auc"].idxmax(), "beta"]),
        "note": ("beta=0 is the frozen anchor, beta=1 removes it up to the trust "
                 "gate; the argmax on VALIDATION is the only one a deployable "
                 "system could have chosen"),
    }
    log(f"  anchor sweep {anchor}")

    # ---- does the effect replicate in the VALIDATION period? --------------
    validation_replication = cpc.block_bootstrap_paired(
        y_val, sf_val, frozen_val, validation["datetime"], n_boot=n_boot,
        block_days=BLOCK_DAYS, metric="auc", seed=SEED)
    validation_replication["n_memory_bars"] = int(len(memory_val))
    validation_replication["note"] = (
        "same machine, same protocol, memory built from train only and deployed "
        "at 2025-10-01; this is the ONLY out-of-sample period available before "
        "the test split is opened")
    log(f"  validation-period replication: delta {validation_replication['delta']:+.5f} "
        f"CI {validation_replication['ci_95']} P {validation_replication['p_delta_le_zero']}")

    hyper = {k: v[0] for k, v in chosen.items()}
    hyper["frozen_seed_averaged"] = {"seeds": list(cpc.SLOW_SEEDS)}
    hyper["retrieval_only"] = {"n_distinct_states": n_ronly, "anchor": "removed"}
    hyper["r3ttt_selection_free"] = {"n_members": n_members, "selection": "none"}

    rows = []
    for name, p in predictions.items():
        against_sf = cpc.block_bootstrap_paired(
            y_test, sf, p, test["datetime"], n_boot=n_boot,
            block_days=BLOCK_DAYS, metric="auc", seed=SEED)
        against_frozen = cpc.block_bootstrap_paired(
            y_test, p, frozen, test["datetime"], n_boot=n_boot,
            block_days=BLOCK_DAYS, metric="auc", seed=SEED)
        rows.append({
            "method": name,
            "test_auc": auc(y_test, p),
            "test_brier": cpc.brier(y_test, p),
            "test_log_loss": float(np.mean(loss_log(y_test, p))),
            "sf_minus_method": against_sf["delta"],
            "sf_minus_method_ci_lo": against_sf["ci_95"][0],
            "sf_minus_method_ci_hi": against_sf["ci_95"][1],
            "p_sf_not_better": against_sf["p_delta_le_zero"],
            "method_minus_frozen": against_frozen["delta"],
            "method_minus_frozen_ci_lo": against_frozen["ci_95"][0],
            "method_minus_frozen_ci_hi": against_frozen["ci_95"][1],
            "p_method_not_better_than_frozen": against_frozen["p_delta_le_zero"],
            "uses_test_labels": name in {"periodic_retraining", "rolling_gbt",
                                         "online_logistic_calibration"},
            "gradient_steps_at_test_time": 0,
            "hyperparameters": json.dumps(hyper.get(name, {}), default=str),
        })
    table = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    table.to_csv(OUT / "main_table.csv", index=False)
    log("\n" + table[["method", "test_auc", "sf_minus_method",
                      "sf_minus_method_ci_lo", "sf_minus_method_ci_hi",
                      "p_sf_not_better"]].to_string(index=False))

    # ---- selection regret --------------------------------------------------
    test_grid = score_config_grid(memory, test, frozen, groups, row_bar)
    test_grid = test_grid.rename(columns={"auc": "test_auc"})
    key = ["space", "k", "beta", "temperature", "prior_strength", "gate"]
    for frame_ in (val_grid, test_grid):
        frame_["k"] = frame_["k"].astype(str)
        frame_["gate"] = frame_["gate"].astype(bool)
    grid = val_grid.merge(test_grid, on=key, validate="one_to_one")
    assert len(grid) == len(val_grid) == len(test_grid), (
        f"config grid merge lost rows: {len(grid)} vs {len(val_grid)}")
    grid["is_validation_selected"] = (
        grid["validation_auc"] == grid["validation_auc"].max())
    grid.to_csv(OUT / "r3ttt_config_grid.csv", index=False)
    regret = {
        "n_configurations": int(len(grid)),
        "validation_selected_test_auc": float(
            grid.loc[grid["is_validation_selected"], "test_auc"].iloc[0]),
        "oracle_test_auc": float(grid["test_auc"].max()),
        "worst_test_auc": float(grid["test_auc"].min()),
        "selection_regret_auc": float(
            grid["test_auc"].max()
            - grid.loc[grid["is_validation_selected"], "test_auc"].iloc[0]),
        "selection_free_test_auc": float(auc(y_test, sf)),
        "selection_free_minus_validation_selected": float(
            auc(y_test, sf) - grid.loc[grid["is_validation_selected"], "test_auc"].iloc[0]),
        "frozen_test_auc": float(auc(y_test, frozen)),
        "share_of_configs_beating_frozen": float(
            np.mean(grid["test_auc"] > auc(y_test, frozen))),
    }
    jdump({"chosen_hyperparameters": hyper, "selection_regret": regret,
           "anchor_sweep": anchor, "blend_sweep": blend.to_dict("records"),
           "validation_period_replication": validation_replication,
           "n_members_selection_free": int(n_members),
           "n_test": int(len(test)), "n_memory": int(len(memory))},
          OUT / "main_summary.json")
    log(f"  selection regret {regret}")

    np.savez(CACHE / "test_predictions.npz",
             y=y_test, **{k: v for k, v in predictions.items()})
    pd.DataFrame({"datetime": test["datetime"], "y": y_test,
                  **{k: v for k, v in predictions.items()}}
                 ).to_csv(OUT / "test_predictions.csv", index=False)
    return table


# =============================================================================
# stage: mechanism controls
# =============================================================================
def shuffle_within_day(values, days, rng):
    values = np.asarray(values).copy()
    d = pd.to_datetime(pd.Series(np.asarray(days))).dt.normalize().to_numpy()
    for day in np.unique(d):
        loc = np.flatnonzero(d == day)
        values[loc] = rng.permutation(values[loc])
    return values


def shuffle_global(values, days, rng):
    return rng.permutation(np.asarray(values).copy())


def shuffle_between_day(values, days, rng):
    """Preserve the empirical distribution of DAY-LEVEL positive rates but
    reassign those rates to the wrong days, then redraw i.i.d. within day."""
    values = np.asarray(values, float)
    d = pd.to_datetime(pd.Series(np.asarray(days))).dt.normalize().to_numpy()
    unique = np.unique(d)
    rates = np.array([values[d == day].mean() for day in unique])
    permuted = rates[rng.permutation(len(rates))]
    out = np.empty_like(values)
    for day, rate in zip(unique, permuted):
        loc = np.flatnonzero(d == day)
        out[loc] = (rng.random(len(loc)) < rate).astype(float)
    return out


SHUFFLERS = {"within_day": shuffle_within_day,
             "global": shuffle_global,
             "between_day": shuffle_between_day}


def _memory_shuffle_worker(draw, kind):
    ctx = context()
    test, pretest = ctx["test"], ctx["pretest"]
    frozen = cpc.slow_probability(pretest, test)
    memory = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    rng = np.random.default_rng(70_000 + 1000 * list(SHUFFLERS).index(kind) + draw)
    labels = SHUFFLERS[kind](memory.labels, memory.frame["datetime"], rng)
    memory2 = cpc.BarMemory(memory.frame.assign(target_hi_vol=labels), labels)
    groups = cpc.bar_groups(test)
    row_bar = row_bar_index(test, groups)
    states = collect_states(memory2, test, groups)
    p, _ = assemble(states, frozen, row_bar)
    ronly, _ = retrieval_only(states, row_bar, len(test))
    y = test["target_hi_vol"].to_numpy(int)
    return {"draw": draw, "kind": kind, "sf_auc": auc(y, p),
            "frozen_auc": auc(y, frozen), "delta": auc(y, p) - auc(y, frozen),
            "retrieval_only_auc": auc(y, ronly)}


def _random_route_worker(seed, mode):
    ctx = context()
    test, pretest = ctx["test"], ctx["pretest"]
    frozen = cpc.slow_probability(pretest, test)
    memory = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    groups = cpc.bar_groups(test)
    row_bar = row_bar_index(test, groups)
    states = collect_states(memory, test, groups, mode=mode, seed=seed)
    p, _ = assemble(states, frozen, row_bar)
    ronly, _ = retrieval_only(states, row_bar, len(test))
    y = test["target_hi_vol"].to_numpy(int)
    return {"seed": seed, "mode": mode, "sf_auc": auc(y, p),
            "frozen_auc": auc(y, frozen), "delta": auc(y, p) - auc(y, frozen),
            "retrieval_only_auc": auc(y, ronly)}


def _perm_chunk(chunk_seeds, bias, trust, blends, slow_logit, y, row_bar, n_bars,
                ronly_rows):
    """Vectorised query-state permutation chunk.  bias/trust: (n_states, n_bars)."""
    out = np.empty(len(chunk_seeds), float)
    out_ronly = np.empty(len(chunk_seeds), float)
    for position, seed in enumerate(chunk_seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_bars)
        b = bias[:, perm][:, row_bar]
        t = trust[:, perm][:, row_bar]
        total = np.zeros(len(y), float)
        for beta in blends:
            effective = beta * t
            total += cpc.sigmoid((1.0 - effective) * slow_logit[None, :]
                                 + effective * b).sum(axis=0)
        out[position] = auc(y, total / (b.shape[0] * len(blends)))
        out_ronly[position] = auc(y, cpc.sigmoid(b[ronly_rows]).mean(axis=0))
    return out, out_ronly


def stage_mech(jobs: int, n_boot: int, n_perm: int):
    log("stage mech")
    ctx = context()
    test, pretest = ctx["test"], ctx["pretest"]
    y = test["target_hi_vol"].to_numpy(int)
    frozen = cpc.slow_probability(pretest, test)
    memory = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    groups = cpc.bar_groups(test)
    row_bar = row_bar_index(test, groups)
    states = collect_states(memory, test, groups)
    sf, _ = assemble(states, frozen, row_bar)
    sf_auc, frozen_auc = auc(y, sf), auc(y, frozen)
    observed = sf_auc - frozen_auc
    log(f"  observed SF {sf_auc:.6f} frozen {frozen_auc:.6f} delta {observed:+.6f}")

    rows = []

    def record(control, sf_values, delta_values, n_reps, note, p_value=None,
               retrieval_only_auc=None):
        sf_values = np.asarray(sf_values, float)
        delta_values = np.asarray(delta_values, float)
        rows.append({
            "control": control, "n_reps": int(n_reps),
            "sf_auc_mean": float(np.mean(sf_values)),
            "sf_auc_sd": float(np.std(sf_values, ddof=1)) if n_reps > 1 else 0.0,
            "delta_mean": float(np.mean(delta_values)),
            "delta_sd": float(np.std(delta_values, ddof=1)) if n_reps > 1 else 0.0,
            "delta_p2.5": float(np.percentile(delta_values, 2.5)),
            "delta_p97.5": float(np.percentile(delta_values, 97.5)),
            "share_reps_positive": float(np.mean(delta_values > 0.0)),
            "observed_delta": float(observed),
            "p_observed_under_control": p_value,
            "retrieval_only_auc_mean": (float(np.mean(retrieval_only_auc))
                                        if retrieval_only_auc is not None else None),
            "note": note,
        })

    # ---- 1. query-state permutation --------------------------------------
    log(f"  query-state permutation, {n_perm} replicates")
    bias = np.stack([s["bias_bar"] for s in states])
    trust = np.stack([s["trust_bar"] for s in states])
    slow_logit = cpc.logit(frozen)
    n_bars = bias.shape[1]
    seeds = np.arange(n_perm) + 500_000
    ronly_rows = np.array([i for i, s in enumerate(states) if not s["gate"]])
    chunks = np.array_split(seeds, max(1, min(jobs, 64)))
    parts = Parallel(n_jobs=jobs)(
        delayed(_perm_chunk)(c, bias, trust, list(cpc.BLENDS), slow_logit, y,
                             row_bar, n_bars, ronly_rows) for c in chunks)
    perm_auc = np.concatenate([p[0] for p in parts])
    perm_ronly = np.concatenate([p[1] for p in parts])
    perm_delta = perm_auc - frozen_auc
    p_perm = float((1.0 + np.sum(perm_delta >= observed)) / (1.0 + len(perm_delta)))
    record("query_state_permutation", perm_auc, perm_delta, len(perm_delta),
           "bar i is handed bar perm(i)'s fast state; routing destroyed, "
           "marginal distribution of fast states preserved", p_perm,
           retrieval_only_auc=perm_ronly)
    pd.DataFrame({"seed": seeds, "sf_auc": perm_auc, "delta": perm_delta,
                  "retrieval_only_auc": perm_ronly}
                 ).to_csv(OUT / "query_state_permutation.csv", index=False)
    log(f"    permutation p = {p_perm:.5f}  mean delta {perm_delta.mean():+.5f}")
    ronly_real = auc(y, retrieval_only(states, row_bar, len(test))[0])
    p_perm_ronly = float((1.0 + np.sum(perm_ronly >= ronly_real)) / (1.0 + len(perm_ronly)))
    log(f"    retrieval-only real {ronly_real:.6f}  perm mean {perm_ronly.mean():.6f} "
        f"p = {p_perm_ronly:.5f}")

    # ---- 2. memory-label shuffles ----------------------------------------
    for kind in ("within_day", "global"):
        log(f"  memory-label shuffle ({kind}), {N_SHUFFLE} replicates")
        res = pd.DataFrame(Parallel(n_jobs=jobs)(
            delayed(_memory_shuffle_worker)(d, kind) for d in range(N_SHUFFLE)))
        res.to_csv(OUT / f"memory_label_shuffle_{kind}.csv", index=False)
        p = float((1.0 + np.sum(res["delta"].to_numpy() >= observed))
                  / (1.0 + len(res)))
        record(f"memory_label_shuffle_{kind}", res["sf_auc"], res["delta"], len(res),
               "frozen anchor untouched; only the memory labels are destroyed", p,
               retrieval_only_auc=res["retrieval_only_auc"])

    # ---- 3. global / unrouted fast state ----------------------------------
    log("  global (unrouted) fast state")
    prior = float(np.mean(memory.labels))
    unrouted = []
    for state in states:
        lam = state["prior_strength"]
        n_mem = len(memory.labels)
        q = (memory.labels.sum() + lam * prior) / (n_mem + lam)
        bias_bar = np.full(n_bars, cpc.logit(np.array([q]))[0])
        consensus = abs(prior - 0.5) * 2.0
        trust_bar = np.full(n_bars, consensus if state["gate"] else 1.0)
        unrouted.append({**state, "bias_bar": bias_bar, "trust_bar": trust_bar})
    p_unrouted, _ = assemble(unrouted, frozen, row_bar)
    record("global_unrouted_fast_state", [auc(y, p_unrouted)],
           [auc(y, p_unrouted) - frozen_auc], 1,
           "fast state replaced by the memory-wide prior; no query conditioning")

    # ---- 4. random routing -------------------------------------------------
    log(f"  random routing, {N_ROUTE_SEEDS} seeds")
    res = pd.DataFrame(Parallel(n_jobs=jobs)(
        delayed(_random_route_worker)(s, "random_space") for s in range(N_ROUTE_SEEDS)))
    res.to_csv(OUT / "random_routing.csv", index=False)
    record("random_routing", res["sf_auc"], res["delta"], len(res),
           "routing coordinates replaced by i.i.d. N(0,1) in memory and stream",
           float((1.0 + np.sum(res["delta"] >= observed)) / (1.0 + len(res))),
           retrieval_only_auc=res["retrieval_only_auc"])

    # ---- 5. random memory draw --------------------------------------------
    log(f"  random-memory draw, {N_ROUTE_SEEDS} seeds")
    res = pd.DataFrame(Parallel(n_jobs=jobs)(
        delayed(_random_route_worker)(s, "random_memory") for s in range(N_ROUTE_SEEDS)))
    res.to_csv(OUT / "random_memory_draw.csv", index=False)
    record("random_memory_draw", res["sf_auc"], res["delta"], len(res),
           "a uniformly random memory subset replaces the k nearest; true "
           "distances retained for the weighting kernel",
           float((1.0 + np.sum(res["delta"] >= observed)) / (1.0 + len(res))),
           retrieval_only_auc=res["retrieval_only_auc"])

    # ---- 6. uniform weights ------------------------------------------------
    log("  uniform retrieval weights (T -> inf)")
    states_uniform = collect_states(memory, test, groups, temperatures=(1e12,))
    p_uniform, n_uniform = assemble(states_uniform, frozen, row_bar)
    record("uniform_weights", [auc(y, p_uniform)], [auc(y, p_uniform) - frozen_auc], 1,
           f"distance kernel switched off (T=1e12); {n_uniform} members, not 72, "
           "because the two temperatures collapse")

    frame = pd.DataFrame(rows)
    frame.insert(1, "sf_auc_real", sf_auc)
    frame.insert(2, "frozen_auc_real", frozen_auc)
    frame.to_csv(OUT / "mechanism_controls.csv", index=False)
    log("\n" + frame[["control", "n_reps", "sf_auc_mean", "delta_mean",
                      "p_observed_under_control"]].to_string(index=False))

    # single-run controls also get a paired bootstrap against frozen
    single = {}
    for name, p in (("global_unrouted_fast_state", p_unrouted),
                    ("uniform_weights", p_uniform)):
        single[name] = cpc.block_bootstrap_paired(
            y, p, frozen, test["datetime"], n_boot=n_boot,
            block_days=BLOCK_DAYS, metric="auc", seed=SEED)
    jdump({"observed_delta": observed, "sf_auc": sf_auc, "frozen_auc": frozen_auc,
           "retrieval_only": {
               "auc": ronly_real,
               "permutation_mean_auc": float(perm_ronly.mean()),
               "permutation_p": p_perm_ronly},
           "single_run_bootstraps": single}, OUT / "mechanism_summary.json")
    return frame


# =============================================================================
# stage: label-delay audit
# =============================================================================
def _delay_worker(delay_hours, n_boot):
    ctx = context(delay_hours)
    test, pretest = ctx["test"], ctx["pretest"]
    y = test["target_hi_vol"].to_numpy(int)
    frozen = cpc.slow_probability(pretest, test)
    memory = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    sf, *_ = cpc.r3ttt_selection_free(memory, test, frozen)
    boot = cpc.block_bootstrap_paired(y, sf, frozen, test["datetime"],
                                      n_boot=n_boot, block_days=BLOCK_DAYS,
                                      metric="auc", seed=SEED)
    calib = online_logistic_calibration(test, frozen, window=512, regularization_c=1.0)
    periodic = periodic_retraining(pretest, test, frequency_days=7, rolling_events=None)
    return {
        "label_delay_hours": delay_hours,
        "n_memory_bars": int(len(memory)),
        "memory_last_bar": str(memory.frame["datetime"].max()),
        "sf_auc": auc(y, sf), "frozen_auc": auc(y, frozen),
        "delta": boot["delta"], "ci_lo": boot["ci_95"][0], "ci_hi": boot["ci_95"][1],
        "p_delta_le_zero": boot["p_delta_le_zero"],
        "sf_brier": cpc.brier(y, sf), "frozen_brier": cpc.brier(y, frozen),
        "online_calibration_auc": auc(y, calib),
        "periodic_retraining_7d_auc": auc(y, periodic),
    }


def stage_delay(jobs: int, n_boot: int):
    log("stage delay")
    res = pd.DataFrame(Parallel(n_jobs=min(jobs, 4))(
        delayed(_delay_worker)(d, n_boot) for d in (0.0, 1.0, 4.0, 24.0)))
    res.to_csv(OUT / "label_delay_audit.csv", index=False)
    log("\n" + res.to_string(index=False))
    return res


# =============================================================================
# stage: ablations
# =============================================================================
def stage_abl(jobs: int, n_boot: int):
    log("stage abl")
    ctx = context()
    test, pretest = ctx["test"], ctx["pretest"]
    y = test["target_hi_vol"].to_numpy(int)
    frozen = cpc.slow_probability(pretest, test)
    groups = cpc.bar_groups(test)
    row_bar = row_bar_index(test, groups)

    memory_bar = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    matured = pretest[pretest["available_time"] <= cpc.TEST_START].reset_index(drop=True)
    memory_event = cpc.BarMemory(matured, matured["target_hi_vol"].to_numpy(float))

    variants = {}
    for dedup, memory in (("bar_dedup_on", memory_bar), ("bar_dedup_off", memory_event)):
        for balanced in (False, True):
            name = f"{dedup}__balanced_{'on' if balanced else 'off'}"
            states = collect_states(memory, test, groups, balanced=balanced)
            p, n_members = assemble(states, frozen, row_bar)
            variants[name] = (p, n_members, len(memory))

    rows = []
    for name, (p, n_members, n_memory) in variants.items():
        boot = cpc.block_bootstrap_paired(y, p, frozen, test["datetime"],
                                          n_boot=n_boot, block_days=BLOCK_DAYS,
                                          metric="auc", seed=SEED)
        rows.append({"variant": name, "n_memory_items": n_memory,
                     "n_members": n_members, "sf_auc": auc(y, p),
                     "frozen_auc": auc(y, frozen), "delta": boot["delta"],
                     "ci_lo": boot["ci_95"][0], "ci_hi": boot["ci_95"][1],
                     "p_delta_le_zero": boot["p_delta_le_zero"],
                     "sf_brier": cpc.brier(y, p),
                     "is_headline": name == "bar_dedup_on__balanced_off"})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "ablations.csv", index=False)
    log("\n" + frame.to_string(index=False))
    return frame


# =============================================================================
# stage: Diebold-Mariano
# =============================================================================
def newey_west_lag(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def diebold_mariano(loss_a, loss_b, lag=None) -> dict:
    """DM on d = L_a - L_b.  Negative statistic => a has the lower loss."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if lag is None:
        lag = newey_west_lag(n)
    lag = max(0, min(int(lag), n - 2))
    dbar = float(d.mean())
    centered = d - dbar
    gamma0 = float(np.mean(centered ** 2))
    long_run = gamma0
    for l in range(1, lag + 1):
        long_run += 2.0 * (1.0 - l / (lag + 1.0)) * float(
            np.mean(centered[l:] * centered[:-l]))
    long_run = max(long_run, 1e-30)
    dm = dbar / np.sqrt(long_run / n)
    correction = np.sqrt(max((n - 1) / n, 1e-12))
    dm_hln = dm * correction
    return {"n_obs": int(n), "hac_lag": int(lag), "mean_loss_differential": dbar,
            "dm_stat": float(dm),
            "p_value_two_sided": float(2.0 * (1.0 - sps.norm.cdf(abs(dm)))),
            "dm_stat_hln": float(dm_hln),
            "p_value_hln_t": float(2.0 * (1.0 - sps.t.cdf(abs(dm_hln), df=n - 1))),
            "p_value_one_sided_a_better": float(sps.norm.cdf(dm)),
            "long_run_variance": float(long_run), "iid_variance": float(gamma0),
            "variance_inflation_hac_over_iid": float(long_run / max(gamma0, 1e-30))}


def stage_dm():
    log("stage dm")
    data = np.load(CACHE / "test_predictions.npz")
    y = data["y"]
    ctx = context()
    days = ctx["test"]["datetime"].dt.normalize().to_numpy()
    unique_days = np.array(sorted(set(days)))
    sf = data["r3ttt_selection_free"]
    rows = []
    for name in data.files:
        if name in ("y", "r3ttt_selection_free"):
            continue
        for metric, fn in (("brier", loss_brier), ("log_loss", loss_log)):
            la, lb = fn(y, sf), fn(y, data[name])
            for level in ("row", "day"):
                if level == "row":
                    a, b = la, lb
                else:
                    a = np.array([la[days == d].mean() for d in unique_days])
                    b = np.array([lb[days == d].mean() for d in unique_days])
                out = diebold_mariano(a, b)
                rows.append({"comparison": f"sf_vs_{name}", "metric": metric,
                             "aggregation": level, **out})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "diebold_mariano.csv", index=False)
    show = frame[frame["comparison"].isin(
        ["sf_vs_frozen_seed_averaged", "sf_vs_frozen_single_seed"])]
    log("\n" + show[["comparison", "metric", "aggregation", "n_obs", "hac_lag",
                     "mean_loss_differential", "dm_stat", "p_value_two_sided"]
                    ].to_string(index=False))
    return frame


# =============================================================================
# stage: e2 diagnosis
# =============================================================================
def _e2_worker(draw, kind):
    """Shuffle TRAINING+MEMORY labels, refit, evaluate against REAL test labels."""
    ctx = context()
    panel, test = ctx["panel"], ctx["test"]
    rng = np.random.default_rng(90_000 + 1000 * list(SHUFFLERS).index(kind) + draw)
    corrupted = panel.copy()
    corrupted["target_hi_vol"] = SHUFFLERS[kind](
        panel["target_hi_vol"].to_numpy(float), panel["datetime"], rng).astype(int)
    pretest = corrupted[corrupted["datetime"] < cpc.TEST_START].reset_index(drop=True)
    frozen = cpc.slow_probability(pretest, test)
    memory = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    groups = cpc.bar_groups(test)
    row_bar = row_bar_index(test, groups)
    states = collect_states(memory, test, groups)
    sf, _ = assemble(states, frozen, row_bar)
    bias_only, _ = retrieval_only(states, row_bar, len(test))
    y = test["target_hi_vol"].to_numpy(int)
    return {"draw": draw, "kind": kind, "frozen_auc": auc(y, frozen),
            "sf_auc": auc(y, sf), "delta": auc(y, sf) - auc(y, frozen),
            "fast_state_alone_auc": auc(y, bias_only)}


def _damage_worker(spec):
    """Degrade the slow model WITHOUT corrupting the memory, and measure delta."""
    ctx = context()
    test, pretest = ctx["test"], ctx["pretest"]
    y = test["target_hi_vol"].to_numpy(int)
    trees = spec.get("trees", cpc.SLOW_TREES)
    depth = spec.get("depth", cpc.SLOW_DEPTH)
    noise = spec.get("train_label_noise", 0.0)
    corrupt_memory = spec.get("corrupt_memory", False)
    rep = spec.get("rep", 0)

    train_frame = pretest
    if noise > 0.0:
        rng = np.random.default_rng(310_000 + 7919 * rep + int(1000 * noise))
        labels = pretest["target_hi_vol"].to_numpy(int).copy()
        flip = rng.random(len(labels)) < noise
        labels[flip] = 1 - labels[flip]
        train_frame = pretest.assign(target_hi_vol=labels)
    frozen = cpc.slow_probability(train_frame, test, trees=trees, depth=depth)
    memory_source = train_frame if corrupt_memory else pretest
    memory = cpc.build_bar_memory(memory_source, deploy_start=cpc.TEST_START)
    groups = cpc.bar_groups(test)
    row_bar = row_bar_index(test, groups)
    states = collect_states(memory, test, groups)
    sf, _ = assemble(states, frozen, row_bar)
    bias_only, _ = retrieval_only(states, row_bar, len(test))
    return {**{k: v for k, v in spec.items()},
            "frozen_auc": auc(y, frozen), "sf_auc": auc(y, sf),
            "delta": auc(y, sf) - auc(y, frozen),
            "fast_state_alone_auc": auc(y, bias_only)}


def stage_e2(jobs: int, n_shuffle: int):
    log("stage e2")
    ctx = context()
    test, pretest = ctx["test"], ctx["pretest"]
    y = test["target_hi_vol"].to_numpy(int)
    frozen = cpc.slow_probability(pretest, test)
    memory = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    groups = cpc.bar_groups(test)
    row_bar = row_bar_index(test, groups)
    states = collect_states(memory, test, groups)
    sf, _ = assemble(states, frozen, row_bar)
    real = {"frozen_auc": auc(y, frozen), "sf_auc": auc(y, sf),
            "delta": auc(y, sf) - auc(y, frozen),
            "fast_state_alone_auc": retrieval_only(states, row_bar, len(test))[0]}
    real["fast_state_alone_auc"] = auc(y, real["fast_state_alone_auc"])

    # how much of the label is day-level in the first place?
    day_rate = ctx["panel"].groupby(ctx["panel"]["datetime"].dt.normalize()
                                    )["target_hi_vol"].transform("mean")
    test_mask = (ctx["panel"]["datetime"] >= cpc.TEST_START).to_numpy()
    day_oracle_auc = auc(y, day_rate.to_numpy()[test_mask])

    # ---- A. the shuffle-granularity ladder --------------------------------
    ladder = []
    for kind in ("within_day", "between_day", "global"):
        log(f"  e2 ladder: {kind}, {n_shuffle} draws")
        res = pd.DataFrame(Parallel(n_jobs=jobs)(
            delayed(_e2_worker)(d, kind) for d in range(n_shuffle)))
        res.to_csv(OUT / f"e2_draws_{kind}.csv", index=False)
        ladder.append({
            "shuffle": kind, "n_draws": len(res),
            "frozen_auc_mean": res["frozen_auc"].mean(),
            "sf_auc_mean": res["sf_auc"].mean(),
            "delta_mean": res["delta"].mean(),
            "delta_p2.5": np.percentile(res["delta"], 2.5),
            "delta_p97.5": np.percentile(res["delta"], 97.5),
            "share_positive": float(np.mean(res["delta"] > 0)),
            "fast_state_alone_auc_mean": res["fast_state_alone_auc"].mean(),
        })
    ladder = pd.DataFrame(ladder)
    ladder.to_csv(OUT / "e2_shuffle_ladder.csv", index=False)
    log("\n" + ladder.to_string(index=False))

    # ---- B. slow-model damage that does NOT touch labels ------------------
    specs = [{"axis": "trees", "trees": t} for t in
             (1, 2, 3, 5, 8, 13, 25, 50, 100, 200, 400)]
    specs += [{"axis": "depth", "depth": d} for d in (1, 2, 3, 4, 6)]
    # ---- C. training-label noise, memory clean vs memory corrupted --------
    for rate in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        for rep in range(N_NOISE_REPS):
            specs.append({"axis": "train_noise_memory_clean",
                          "train_label_noise": rate, "rep": rep,
                          "corrupt_memory": False})
            specs.append({"axis": "train_noise_memory_corrupt",
                          "train_label_noise": rate, "rep": rep,
                          "corrupt_memory": True})
    log(f"  damage ladder: {len(specs)} runs")
    damage = pd.DataFrame(Parallel(n_jobs=jobs)(
        delayed(_damage_worker)(s) for s in specs))
    damage.to_csv(OUT / "e2_damage_ladder.csv", index=False)

    summary = (damage.groupby(["axis", "trees", "depth", "train_label_noise",
                               "corrupt_memory"], dropna=False)
               [["frozen_auc", "sf_auc", "delta", "fast_state_alone_auc"]]
               .mean().reset_index())
    summary.to_csv(OUT / "e2_damage_summary.csv", index=False)
    log("\n" + summary.to_string(index=False))

    # correlation between slow-model damage and apparent advantage
    correlations = {}
    for axis, sub in damage.groupby("axis"):
        if len(sub) >= 4:
            r, p = sps.pearsonr(sub["frozen_auc"], sub["delta"])
            rs, ps = sps.spearmanr(sub["frozen_auc"], sub["delta"])
            correlations[axis] = {"n": int(len(sub)), "pearson_r": float(r),
                                  "pearson_p": float(p), "spearman_r": float(rs),
                                  "spearman_p": float(ps)}
    jdump({"real": real, "day_rate_oracle_auc": day_oracle_auc,
           "shuffle_ladder": ladder.to_dict("records"),
           "damage_delta_vs_frozen_auc_correlation": correlations},
          OUT / "e2_summary.json")
    log(f"  correlations {correlations}")
    return ladder, damage


# =============================================================================
# stage: is the pre-specified anchor the anchor validation would have picked?
# =============================================================================
def _anchor_worker(trees, depth, n_boot):
    ctx = context()
    train, validation = ctx["train"], ctx["validation"]
    test, pretest = ctx["test"], ctx["pretest"]
    y_val = validation["target_hi_vol"].to_numpy(int)
    y_test = test["target_hi_vol"].to_numpy(int)
    frozen_val = cpc.slow_probability(train, validation, trees=trees, depth=depth)
    frozen = cpc.slow_probability(pretest, test, trees=trees, depth=depth)
    memory = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    groups = cpc.bar_groups(test)
    row_bar = row_bar_index(test, groups)
    states = collect_states(memory, test, groups)
    sf, _ = assemble(states, frozen, row_bar)
    boot = cpc.block_bootstrap_paired(y_test, sf, frozen, test["datetime"],
                                      n_boot=n_boot, block_days=BLOCK_DAYS,
                                      metric="auc", seed=SEED)
    return {"trees": trees, "depth": depth,
            "frozen_validation_auc": auc(y_val, frozen_val),
            "frozen_test_auc": auc(y_test, frozen), "sf_test_auc": auc(y_test, sf),
            "delta": boot["delta"], "ci_lo": boot["ci_95"][0],
            "ci_hi": boot["ci_95"][1], "p_delta_le_zero": boot["p_delta_le_zero"],
            "is_prespecified": trees == cpc.SLOW_TREES and depth == cpc.SLOW_DEPTH}


def stage_anchor(jobs: int, n_boot: int):
    log("stage anchor")
    specs = [(t, d) for t in (25, 50, 100, 200, 400) for d in (2, 3, 4)]
    frame = pd.DataFrame(Parallel(n_jobs=min(jobs, len(specs)))(
        delayed(_anchor_worker)(t, d, n_boot) for t, d in specs))
    frame = frame.sort_values("frozen_validation_auc", ascending=False)
    frame.to_csv(OUT / "anchor_capacity_sweep.csv", index=False)
    best = frame.iloc[0]
    prespecified = frame[frame["is_prespecified"]].iloc[0]
    summary = {
        "validation_selected_anchor": {"trees": int(best["trees"]),
                                       "depth": int(best["depth"])},
        "validation_selected_frozen_test_auc": float(best["frozen_test_auc"]),
        "validation_selected_delta": float(best["delta"]),
        "validation_selected_ci": [float(best["ci_lo"]), float(best["ci_hi"])],
        "validation_selected_p": float(best["p_delta_le_zero"]),
        "prespecified_frozen_test_auc": float(prespecified["frozen_test_auc"]),
        "prespecified_delta": float(prespecified["delta"]),
        "note": ("the delta is measured against whichever anchor the protocol "
                 "fixes; a stronger anchor selected on VALIDATION is the fair "
                 "stress test of the adaptation claim"),
    }
    jdump(summary, OUT / "anchor_capacity_summary.json")
    log("\n" + frame.to_string(index=False))
    log(f"  {summary}")
    return frame


# =============================================================================
# stage: determinism audit
# =============================================================================
_CHILD = """
import sys
sys.path.insert(0, __CODE_DIR__)
import numpy as np, common_protocol_clean as cpc
panel = cpc.load_event_panel()
pretest = panel[panel["datetime"] < cpc.TEST_START].reset_index(drop=True)
test = panel[panel["datetime"] >= cpc.TEST_START].reset_index(drop=True)
frozen = cpc.slow_probability(pretest, test)
memory = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
sf, *_ = cpc.r3ttt_selection_free(memory, test, frozen)
y = test["target_hi_vol"].to_numpy(int)
print(repr(cpc.fast_binary_auc(y, sf)), repr(cpc.fast_binary_auc(y, frozen)))
"""


def stage_repro():
    """Two determinism questions the clean module's tie-break rule leaves open."""
    log("stage repro")
    import subprocess
    rows = []
    for threads in (1, 2, 4, 8, 16, 32):
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = str(threads)
        out = subprocess.run([sys.executable, "-c", _CHILD.replace("__CODE_DIR__", repr(str(Path(__file__).resolve().parent)))], env=env,
                             capture_output=True, text=True, check=True)
        sf_auc, frozen_auc = (float(v) for v in out.stdout.split())
        rows.append({"blas_threads": threads, "sf_auc": sf_auc,
                     "frozen_auc": frozen_auc, "delta": sf_auc - frozen_auc})
        log(f"  threads={threads:2d}  sf={sf_auc!r}  frozen={frozen_auc!r}")
    threading = pd.DataFrame(rows)
    threading.to_csv(OUT / "determinism_threads.csv", index=False)

    # memory-order tie band, re-established at 1 BLAS thread
    ctx = context()
    test, pretest = ctx["test"], ctx["pretest"]
    y = test["target_hi_vol"].to_numpy(int)
    frozen = cpc.slow_probability(pretest, test)
    memory = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    groups = cpc.bar_groups(test)
    row_bar = row_bar_index(test, groups)
    band = []
    for seed in range(24):
        order = (np.arange(len(memory)) if seed == 0
                 else np.random.default_rng(seed).permutation(len(memory)))
        frame = memory.frame.iloc[order].reset_index(drop=True)
        permuted = cpc.BarMemory(frame, memory.labels[order])
        states = collect_states(permuted, test, groups)
        p, _ = assemble(states, frozen, row_bar)
        band.append({"permutation": seed, "sf_auc": auc(y, p),
                     "frozen_auc": auc(y, frozen),
                     "delta": auc(y, p) - auc(y, frozen)})
    band = pd.DataFrame(band)
    band.to_csv(OUT / "determinism_memory_order.csv", index=False)
    summary = {
        "blas_thread_sensitivity": {
            "sf_auc_min": float(threading["sf_auc"].min()),
            "sf_auc_max": float(threading["sf_auc"].max()),
            "sf_auc_range": float(threading["sf_auc"].max() - threading["sf_auc"].min()),
            "frozen_auc_unique": sorted(set(threading["frozen_auc"])),
        },
        "memory_order_band": {
            "sf_auc_min": float(band["sf_auc"].min()),
            "sf_auc_max": float(band["sf_auc"].max()),
            "sf_auc_sd": float(band["sf_auc"].std(ddof=1)),
            "delta_min": float(band["delta"].min()),
            "delta_max": float(band["delta"].max()),
            "n_positive": int((band["delta"] > 0).sum()), "n": int(len(band)),
        },
        "track_configuration": "all clean_core numbers computed with BLAS threads = 1",
    }
    jdump(summary, OUT / "determinism_summary.json")
    log(f"  {summary}")
    return threading, band


# =============================================================================
# stage: figures
# =============================================================================
def stage_fig():
    log("stage fig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "pdf.fonttype": 42})
    DARK, ACC, GREY = "#1b2838", "#b3452b", "#8a93a0"

    # ---- figure 1: main table forest --------------------------------------
    table = pd.read_csv(OUT / "main_table.csv")
    table = table[table["method"] != "r3ttt_selection_free"].sort_values("sf_minus_method")
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ypos = np.arange(len(table))
    ax.errorbar(table["sf_minus_method"], ypos,
                xerr=[table["sf_minus_method"] - table["sf_minus_method_ci_lo"],
                      table["sf_minus_method_ci_hi"] - table["sf_minus_method"]],
                fmt="o", ms=3.2, lw=0.9, color=DARK, ecolor=GREY, capsize=2)
    ax.axvline(0.0, color=ACC, lw=0.8, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([m.replace("_", " ") for m in table["method"]])
    ax.set_xlabel(r"AUC(R3-TTT-SF) $-$ AUC(method), clean panel")
    ax.set_title("Main comparison, paired 5-day block bootstrap clustered by day",
                 fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_main_table.pdf")
    plt.close(fig)

    # ---- figure 2: mechanism controls -------------------------------------
    mech = pd.read_csv(OUT / "mechanism_controls.csv")
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    order = mech.sort_values("delta_mean")
    ypos = np.arange(len(order))
    lo = order["delta_mean"] - order["delta_p2.5"]
    hi = order["delta_p97.5"] - order["delta_mean"]
    ax.errorbar(order["delta_mean"], ypos, xerr=[lo, hi], fmt="s", ms=3.0,
                lw=0.9, color=DARK, ecolor=GREY, capsize=2)
    ax.axvline(0.0, color=GREY, lw=0.7)
    ax.axvline(order["observed_delta"].iloc[0], color=ACC, lw=1.0,
               label="observed R3-TTT-SF gain")
    ax.set_yticks(ypos)
    ax.set_yticklabels([c.replace("_", " ") for c in order["control"]])
    ax.set_xlabel(r"AUC(control) $-$ AUC(frozen)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.set_title("Mechanism controls, clean panel", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mechanism.pdf")
    plt.close(fig)

    # ---- figure 3: e2 diagnosis -------------------------------------------
    damage = pd.read_csv(OUT / "e2_damage_ladder.csv")
    ladder = pd.read_csv(OUT / "e2_shuffle_ladder.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax = axes[0]
    markers = {"trees": "o", "depth": "^", "train_noise_memory_clean": "s",
               "train_noise_memory_corrupt": "x"}
    for axis, sub in damage.groupby("axis"):
        ax.scatter(sub["frozen_auc"], sub["delta"], s=9, alpha=0.75,
                   marker=markers.get(axis, "o"), label=axis.replace("_", " "))
    for _, r in ladder.iterrows():
        ax.scatter([r["frozen_auc_mean"]], [r["delta_mean"]], s=34, marker="*",
                   color=ACC, zorder=5)
        ax.annotate(r["shuffle"], (r["frozen_auc_mean"], r["delta_mean"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6, color=ACC)
    ax.axhline(0.0, color=GREY, lw=0.7)
    ax.set_xlabel("frozen slow-model AUC (damaged)")
    ax.set_ylabel(r"AUC(SF) $-$ AUC(frozen)")
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    ax.set_title("Apparent gain vs slow-model quality", fontsize=7.5)

    ax = axes[1]
    ypos = np.arange(len(ladder))
    ax.errorbar(ladder["delta_mean"], ypos,
                xerr=[ladder["delta_mean"] - ladder["delta_p2.5"],
                      ladder["delta_p97.5"] - ladder["delta_mean"]],
                fmt="o", ms=3.2, lw=0.9, color=DARK, ecolor=GREY, capsize=2)
    ax.axvline(0.0, color=GREY, lw=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ladder["shuffle"])
    ax.set_xlabel(r"AUC(SF) $-$ AUC(frozen), labels destroyed")
    ax.set_title("Double label-shuffle, by shuffle granularity", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e2_diagnosis.pdf")
    plt.close(fig)

    # ---- figure 4: delay + ablations --------------------------------------
    delay = pd.read_csv(OUT / "label_delay_audit.csv")
    abl = pd.read_csv(OUT / "ablations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax = axes[0]
    ax.errorbar(delay["label_delay_hours"], delay["delta"],
                yerr=[delay["delta"] - delay["ci_lo"], delay["ci_hi"] - delay["delta"]],
                fmt="o-", ms=3.2, lw=0.9, color=DARK, ecolor=GREY, capsize=2)
    ax.axhline(0.0, color=ACC, lw=0.8, ls="--")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xticks(delay["label_delay_hours"])
    ax.set_xticklabels([f"{v:g}" for v in delay["label_delay_hours"]])
    ax.set_xlabel("label delay (hours)")
    ax.set_ylabel(r"AUC(SF) $-$ AUC(frozen)")
    ax.set_title("Label-delay audit", fontsize=7.5)

    ax = axes[1]
    ypos = np.arange(len(abl))
    ax.errorbar(abl["delta"], ypos,
                xerr=[abl["delta"] - abl["ci_lo"], abl["ci_hi"] - abl["delta"]],
                fmt="o", ms=3.2, lw=0.9, color=DARK, ecolor=GREY, capsize=2)
    ax.axvline(0.0, color=GREY, lw=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels([v.replace("__", " / ").replace("_", " ") for v in abl["variant"]])
    ax.set_xlabel(r"AUC(variant) $-$ AUC(frozen)")
    ax.set_title("Retrieval ablations", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_delay_ablations.pdf")
    plt.close(fig)
    log("  4 figures written")


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all",
                        choices=["all", "protocol", "main", "mech", "delay",
                                 "abl", "dm", "e2", "anchor", "repro", "fig"],
                        help="which stage to run (default: all)")
    parser.add_argument("--jobs", type=int, default=48, help="parallel workers")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help="bootstrap replicates (default 5000)")
    parser.add_argument("--n-perm", type=int, default=N_PERM,
                        help="query-state permutation replicates (default 5000)")
    parser.add_argument("--n-shuffle", type=int, default=N_SHUFFLE,
                        help="label-shuffle replicates per kind (default 200)")
    args = parser.parse_args()

    start = time.time()
    stages = (["protocol", "main", "mech", "delay", "abl", "dm", "e2",
               "anchor", "repro", "fig"] if args.stage == "all" else [args.stage])
    for stage in stages:
        if stage == "protocol":
            stage_protocol()
        elif stage == "main":
            stage_main(args.jobs, args.n_boot)
        elif stage == "mech":
            stage_mech(args.jobs, args.n_boot, args.n_perm)
        elif stage == "delay":
            stage_delay(args.jobs, args.n_boot)
        elif stage == "abl":
            stage_abl(args.jobs, args.n_boot)
        elif stage == "dm":
            stage_dm()
        elif stage == "e2":
            stage_e2(args.jobs, args.n_shuffle)
        elif stage == "anchor":
            stage_anchor(args.jobs, args.n_boot)
        elif stage == "repro":
            stage_repro()
        elif stage == "fig":
            stage_fig()
    log(f"done in {time.time() - start:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
