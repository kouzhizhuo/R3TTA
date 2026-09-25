#!/usr/bin/env python3
"""Component ablation and one-axis sensitivity for R3-TTT, CLEAN panel, PER-BAR estimand.

This is the experiment behind the paper's missing "Component ablation" and
"Sensitivity" sections.  Everything is computed in the **clean** environment at
the **primary per-bar estimand** (525 aligned outcome bars, 68 test days);
per-post is carried as a secondary column on every row so that a reader can see
exactly which conclusions were artefacts of burst weighting.

Design provenance: ``refnotes/V_ablation_sensitivity_design.md``.  V was written
against the *regenerated common-protocol* environment (frozen 0.8324) and is
therefore stale in its levels; the structure (build-up ladder + one-axis-at-a-
time marginals with paired bands and a validation twin) is retained, the numbers
are all re-emitted here.

Nothing is ever selected on the 2026 test panel.  The validation stream
(2025-10-01..2025-12-31, memory and slow model from train only) is scored on
every axis purely so the paper can show that validation and test disagree; no
configuration is chosen from it for the headline.

Stages
------
  context     panel census, bar indices, frozen anchors, member predictions
  ablation    component removal (leave-one-out) + cumulative build-up ladder
  sensitivity seven one-axis marginals, both streams, paired bands
  figs        two publication PDFs
  report      REPORT.md + protocol.json + drop-in LaTeX
  all         everything above, in order

Usage
-----
  python code/exp_ablation_sensitivity.py --help
  python code/exp_ablation_sensitivity.py all --jobs 96
  python code/exp_ablation_sensitivity.py sensitivity --n-boot 5000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/r3ttt-abl-mpl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from joblib import Parallel, delayed                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common_protocol_clean as cpc                  # noqa: E402
import exp_clean_core as ecc                         # noqa: E402

OUT = ROOT / "results" / "ablation_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
BLOCK_DAYS = 5
SEED = 42
N_NULL_SEEDS = 200          # replicates for the corrected random-memory null
N_MEM_DRAWS = 10            # draws per memory fraction
N_SUBSET_DRAWS = 200        # random member subsets per ensemble size
BIG_T = 1e12                # numerical stand-in for T -> infinity

# The shipped configuration, for the "distance from the optimum" statements.
SHIPPED = {
    "beta": "grid {0.3, 0.5, 0.7}, mean 0.5",
    "T": "grid {0.25, 1.0}",
    "lambda": "grid {2, 8}",
    "k": "averaged over {8, 16, 32, 64}",
    "gate": "grid {off, on}",
    "memory": "100% (2,249 bars)",
    "ensemble": "72 members, uniform average",
}

_BASE_ROUTE = ecc.route


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (pd.Timestamp, Path)):
        return str(v)
    raise TypeError(type(v))


def jdump(obj, name: str) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=_jsonable))
    log(f"  wrote {p}")
    return p


def save(df: pd.DataFrame, name: str) -> Path:
    p = OUT / name
    df.to_csv(p, index=False)
    log(f"  wrote {p}  ({len(df)} rows)")
    return p


# =============================================================================
# metrics
# =============================================================================
def auc(y, s) -> float:
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return float("nan")
    return cpc.fast_binary_auc(y, np.asarray(s, dtype=float))


def brier(y, p) -> float:
    return cpc.brier(np.asarray(y), np.asarray(p, dtype=float))


def logloss(y, p, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class BarIndex:
    """The decision-unit index for a stream: one outcome bar = one unit."""

    def __init__(self, stream: pd.DataFrame):
        groups = cpc.bar_groups(stream)
        self.groups = groups
        self.n_rows = len(stream)
        self.n_bars = len(groups)
        self.row_bar = np.empty(self.n_rows, dtype=int)
        for pos, locs in enumerate(groups):
            self.row_bar[locs] = pos
        self.sizes = np.array([len(g) for g in groups], dtype=float)
        yrow = stream["target_hi_vol"].to_numpy(dtype=int)
        self.y_row = yrow
        for pos, g in enumerate(groups):
            if np.unique(yrow[g]).size != 1:
                raise AssertionError(f"bar {pos} carries more than one label")
        self.y_bar = np.array([yrow[g][0] for g in groups], dtype=int)
        times = stream["datetime"].to_numpy()
        self.times = times
        self.bar_times = np.array([times[g[0]] for g in groups])

    def pool(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.bincount(self.row_bar, weights=p, minlength=self.n_bars) / self.sizes

    def scores(self, p) -> dict:
        pb = self.pool(p)
        return {
            "auc_bar": auc(self.y_bar, pb), "brier_bar": brier(self.y_bar, pb),
            "logloss_bar": logloss(self.y_bar, pb),
            "auc_row": auc(self.y_row, p), "brier_row": brier(self.y_row, p),
            "logloss_row": logloss(self.y_row, p),
        }


# =============================================================================
# bootstrap -- paired 5-day moving-block, clustered by calendar day
# =============================================================================
_BLOCK_CACHE: dict = {}


def _block_units(days, n_boot, block_days, seed):
    """Row positions of each block-bootstrap replicate.  Cached per stream:
    the resample is a function of the calendar-day vector and the seed only, so
    every arm sees the SAME replicates -- which is what makes the interval
    paired."""
    key = (len(days), hash(np.asarray(days).tobytes()), n_boot, block_days, seed)
    hit = _BLOCK_CACHE.get(key)
    if hit is not None:
        return hit
    d = pd.to_datetime(pd.Series(np.asarray(days))).dt.normalize().to_numpy()
    uniq = np.array(sorted(set(d)))
    index = {u: np.flatnonzero(d == u) for u in uniq}
    n_blocks = max(1, int(np.ceil(len(uniq) / block_days)))
    high = max(1, len(uniq) - block_days + 1)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, high, size=n_blocks)
        chosen = [u for s in starts for u in uniq[s:s + block_days]]
        out.append(np.concatenate([index[u] for u in chosen]))
    if len(_BLOCK_CACHE) < 8:
        _BLOCK_CACHE[key] = out
    return out


def paired_boot(y, a, b, days, *, metric="auc", n_boot=N_BOOT,
                block_days=BLOCK_DAYS, seed=SEED) -> dict:
    """Positive delta always means arm ``a`` is better."""
    y = np.asarray(y)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if metric == "auc":
        fn, sign = auc, 1.0
    elif metric == "brier":
        fn, sign = brier, -1.0
    elif metric == "logloss":
        fn, sign = logloss, -1.0
    else:
        raise ValueError(metric)
    point = sign * (fn(y, a) - fn(y, b))
    draws = []
    for rows in _block_units(days, n_boot, block_days, seed):
        va, vb = fn(y[rows], a[rows]), fn(y[rows], b[rows])
        if np.isfinite(va) and np.isfinite(vb):
            draws.append(sign * (va - vb))
    draws = np.asarray(draws, dtype=float)
    return {
        "metric": metric, "delta": float(point),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "p_delta_le_zero": float(np.mean(draws <= 0.0)),
        "n_boot": int(len(draws)), "block_days": int(block_days), "seed": int(seed),
    }


def contrast(bi: BarIndex, p, ref, *, n_boot=N_BOOT, prefix="") -> dict:
    """Per-bar (primary) and per-post (secondary) paired contrast of p over ref."""
    pb, rb = bi.pool(p), bi.pool(ref)
    out = {}
    for tag, (y, x, z, days) in {
        "bar": (bi.y_bar, pb, rb, bi.bar_times),
        "row": (bi.y_row, np.asarray(p), np.asarray(ref), bi.times),
    }.items():
        r = paired_boot(y, x, z, days, metric="auc", n_boot=n_boot)
        out[f"{prefix}{tag}_delta"] = r["delta"]
        out[f"{prefix}{tag}_ci_lo"] = r["ci_lo"]
        out[f"{prefix}{tag}_ci_hi"] = r["ci_hi"]
        out[f"{prefix}{tag}_p_le_zero"] = r["p_delta_le_zero"]
    rb_ = paired_boot(bi.y_bar, pb, rb, bi.bar_times, metric="brier", n_boot=n_boot)
    out[f"{prefix}bar_brier_delta"] = rb_["delta"]
    out[f"{prefix}bar_brier_ci_lo"] = rb_["ci_lo"]
    out[f"{prefix}bar_brier_ci_hi"] = rb_["ci_hi"]
    out[f"{prefix}bar_brier_p_le_zero"] = rb_["p_delta_le_zero"]
    return out


# =============================================================================
# routing with the CORRECTED random-memory null
# =============================================================================
def route_ext(memory_frame, stream, groups, features, *, max_k=cpc.MAX_K,
              mode="nearest", seed=0):
    """``ecc.route`` plus ``random_memory_scrambled`` (the corrected null).

    ``ecc.route(mode="random_memory")`` is MIS-SPECIFIED as a retrieval
    destruction control: it draws a random 256-of-2,249 candidate pool and then
    sorts it by TRUE distance, so ``_fast_states`` still selects the nearest
    k inside an 11.4% subsample.  ``random_memory_scrambled`` zeroes the
    distances and permutes the order, so the k retrieved slots are k uniformly
    random memory bars -- the control the published one was described as being.
    (``results/control_hardening/REPORT_ADDENDUM.md``.)
    """
    if mode != "random_memory_scrambled":
        return _BASE_ROUTE(memory_frame, stream, groups, features,
                           max_k=max_k, mode=mode, seed=seed)
    idx, dist = _BASE_ROUTE(memory_frame, stream, groups, features,
                            max_k=max_k, mode="random_memory", seed=seed)
    rng = np.random.default_rng(900_000 + seed)
    perm = np.argsort(rng.random(idx.shape), axis=1)
    return np.take_along_axis(idx, perm, axis=1), np.zeros_like(dist)


def collect(memory, stream, groups, *, mode="nearest", seed=0,
            spaces=None, scales=cpc.SCALES, temperatures=cpc.TEMPERATURES,
            prior_strengths=cpc.PRIOR_STRENGTHS, gates=cpc.GATES,
            balanced=cpc.BALANCED_DEFAULT, max_k=None):
    """Per-bar (bias, trust) for every (space, T, lambda, gate) member."""
    spaces = cpc.RETRIEVAL_SPACES if spaces is None else spaces
    max_k = cpc.MAX_K if max_k is None else max_k
    max_k = min(max(max_k, max(int(s) for s in scales)), len(memory.frame))
    out = []
    for name, features in spaces.items():
        idx, dist = route_ext(memory.frame, stream, groups, features,
                              mode=mode, seed=seed, max_k=max_k)
        states = cpc._fast_states(
            idx, dist, memory.labels, balanced=balanced, scales=tuple(scales),
            temperatures=tuple(temperatures), prior_strengths=tuple(prior_strengths),
            gates=tuple(gates),
        )
        for (temperature, prior, gate), (bias_bar, trust_bar) in states.items():
            out.append({"space": name, "temperature": temperature,
                        "prior_strength": prior, "gate": bool(gate),
                        "bias_bar": bias_bar, "trust_bar": trust_bar})
    return out


def assemble(states, frozen, row_bar, *, blends=cpc.BLENDS):
    total = np.zeros(len(frozen), float)
    count = 0
    slow_logit = cpc.logit(frozen)
    for st in states:
        bias = st["bias_bar"][row_bar]
        trust = st["trust_bar"][row_bar]
        for beta in blends:
            eff = beta * trust
            total += cpc.sigmoid((1.0 - eff) * slow_logit + eff * bias)
            count += 1
    return total / count, count


def member_predictions(states, frozen, row_bar, *, blends=cpc.BLENDS):
    """One probability vector per (space, T, lambda, gate, beta) member."""
    slow_logit = cpc.logit(frozen)
    specs, preds = [], []
    for st in states:
        bias = st["bias_bar"][row_bar]
        trust = st["trust_bar"][row_bar]
        for beta in blends:
            eff = beta * trust
            preds.append(cpc.sigmoid((1.0 - eff) * slow_logit + eff * bias))
            specs.append({"space": st["space"], "temperature": st["temperature"],
                          "prior_strength": st["prior_strength"],
                          "gate": st["gate"], "blend": beta})
    return specs, np.stack(preds, axis=0)


# =============================================================================
# streams
# =============================================================================
class Bench:
    """Both streams of the clean panel, each with its own frozen anchor+memory.

    test        slow model and memory from train+validation, scored 2026-01-01+
    validation  slow model and memory from train only, scored 2025-10-01..12-31
    """

    def __init__(self):
        panel = cpc.load_event_panel()
        cpc.assert_clean(panel, where="clean event panel")
        train, validation, test = cpc.split(panel)
        pretest = panel[panel["datetime"] < cpc.TEST_START].reset_index(drop=True)
        self.panel, self.train, self.validation, self.test = panel, train, validation, test
        self.pretest = pretest
        self.stream = {"test": test, "validation": validation}
        self.source = {"test": pretest, "validation": train}
        self.deploy = {"test": cpc.TEST_START, "validation": cpc.TRAIN_END}
        self.bi = {k: BarIndex(v) for k, v in self.stream.items()}
        self.groups = {k: self.bi[k].groups for k in self.stream}
        self.row_bar = {k: self.bi[k].row_bar for k in self.stream}
        self.frozen = {k: cpc.slow_probability(self.source[k], self.stream[k])
                       for k in self.stream}
        self.memory = {k: cpc.build_bar_memory(self.source[k], deploy_start=self.deploy[k])
                       for k in self.stream}
        # event-level (NOT bar-deduplicated) memory, for the dedup ablation
        self.memory_event = {}
        for k in self.stream:
            m = self.source[k]
            m = m[m["available_time"] <= self.deploy[k]].reset_index(drop=True)
            self.memory_event[k] = cpc.BarMemory(m, m["target_hi_vol"].to_numpy(float))

    def run(self, key="test", **kw):
        blends = kw.pop("blends", cpc.BLENDS)
        memory = kw.pop("memory", None)
        memory = self.memory[key] if memory is None else memory
        frozen = kw.pop("frozen", None)
        frozen = self.frozen[key] if frozen is None else frozen
        states = collect(memory, self.stream[key], self.groups[key], **kw)
        p, n = assemble(states, frozen, self.row_bar[key], blends=blends)
        return p, n


_BENCH = None


def bench() -> Bench:
    global _BENCH
    if _BENCH is None:
        log("building bench (panel, splits, frozen anchors, memories)")
        _BENCH = Bench()
    return _BENCH


# =============================================================================
# stage: context
# =============================================================================
def stage_context(args) -> dict:
    b = bench()
    rec = {}
    for k in ("test", "validation"):
        bi = b.bi[k]
        sizes = pd.Series(bi.sizes)
        p_sf, n_members = b.run(k)
        rec[k] = {
            "n_posts": int(bi.n_rows), "n_bars": int(bi.n_bars),
            "n_days": int(pd.to_datetime(pd.Series(bi.times)).dt.normalize().nunique()),
            "posts_per_bar_mean": float(sizes.mean()),
            "posts_per_bar_max": int(sizes.max()),
            "singleton_bars": int((sizes == 1).sum()),
            "n_memory_bars": len(b.memory[k]),
            "n_memory_events": len(b.memory_event[k]),
            "memory_prior": float(np.mean(b.memory[k].labels)),
            "n_members": int(n_members),
            "frozen": b.bi[k].scores(b.frozen[k]),
            "sf": b.bi[k].scores(p_sf),
        }
        rec[k]["delta_auc_bar"] = rec[k]["sf"]["auc_bar"] - rec[k]["frozen"]["auc_bar"]
        rec[k]["delta_auc_row"] = rec[k]["sf"]["auc_row"] - rec[k]["frozen"]["auc_row"]
        np.save(CACHE / f"sf_{k}.npy", p_sf)
        np.save(CACHE / f"frozen_{k}.npy", b.frozen[k])
        log(f"  {k}: {rec[k]['n_bars']} bars / {rec[k]['n_posts']} posts, "
            f"frozen {rec[k]['frozen']['auc_bar']:.4f} -> SF {rec[k]['sf']['auc_bar']:.4f} "
            f"({rec[k]['delta_auc_bar']:+.4f} bar) | row "
            f"{rec[k]['frozen']['auc_row']:.4f} -> {rec[k]['sf']['auc_row']:.4f}")
    # cache the 72 member predictions for the ensemble-size axis
    for k in ("test", "validation"):
        states = collect(b.memory[k], b.stream[k], b.groups[k])
        specs, preds = member_predictions(states, b.frozen[k], b.row_bar[k])
        np.save(CACHE / f"members_{k}.npy", preds)
        pd.DataFrame(specs).to_csv(CACHE / f"members_{k}.csv", index=False)
    jdump(rec, "context.json")
    return rec


# =============================================================================
# stage: ablation
# =============================================================================
def _destruction_worker(seed: int, mode: str):
    """One replicate of a retrieval-destruction control on the test stream.

    Also records the RETRIEVAL-ONLY arm -- sigma(u) with the anchor removed --
    because a destruction control that leaves the fused arm near frozen is only
    interpretable if the retrieval component itself is at chance.
    """
    b = bench()
    bi = b.bi["test"]
    states = collect(b.memory["test"], b.test, b.groups["test"], mode=mode, seed=seed)
    p, _ = assemble(states, b.frozen["test"], b.row_bar["test"])
    distinct = {}
    for st in states:
        distinct.setdefault((st["space"], st["temperature"], st["prior_strength"]),
                            st["bias_bar"])
    ronly = np.mean([cpc.sigmoid(v[b.row_bar["test"]]) for v in distinct.values()],
                    axis=0)
    return {"seed": seed, "mode": mode,
            "auc_bar": auc(bi.y_bar, bi.pool(p)),
            "auc_row": auc(bi.y_row, p),
            "brier_bar": brier(bi.y_bar, bi.pool(p)),
            "retrieval_only_auc_bar": auc(bi.y_bar, bi.pool(ronly)),
            "retrieval_only_auc_row": auc(bi.y_row, ronly)}


def _seed_worker(seed: int):
    """Frozen slow model from ONE seed, and the SF ensemble built on it."""
    b = bench()
    frozen1 = cpc.slow_probability(b.pretest, b.test, seeds=(seed,))
    p, _ = b.run("test", frozen=frozen1)
    bi = b.bi["test"]
    return {"seed": seed,
            "frozen_auc_bar": auc(bi.y_bar, bi.pool(frozen1)),
            "frozen_auc_row": auc(bi.y_row, frozen1),
            "sf_auc_bar": auc(bi.y_bar, bi.pool(p)),
            "sf_auc_row": auc(bi.y_row, p),
            "sf_brier_bar": brier(bi.y_bar, bi.pool(p)),
            "delta_bar": auc(bi.y_bar, bi.pool(p)) - auc(bi.y_bar, bi.pool(frozen1))}


def stage_ablation(args) -> pd.DataFrame:
    log("stage ablation -- component removal + cumulative build-up, per bar")
    b = bench()
    bi = b.bi["test"]
    frozen = b.frozen["test"]
    nb = args.n_boot

    variants: dict[str, dict] = {}

    def add(name, block, note, p, extra=None):
        variants[name] = {"block": block, "note": note, "p": np.asarray(p, float),
                          "extra": extra or {}}

    # ---- reference ---------------------------------------------------------
    p_full, n_members = b.run("test")
    add("R3-TTT-SF (full, 72 members)", "reference",
        "the shipped selection-free ensemble", p_full, {"n_members": n_members})

    # ---- Block A: leave-one-out removals -----------------------------------
    p, n = b.run("test", blends=(1.0,))
    add("- frozen anchor (beta = 1)", "A",
        "retrieval only; the anchor is deleted, 24 distinct members", p, {"n_members": n})

    # corrected random-memory null (the retrieval removal)
    log(f"  destruction controls, {args.null_seeds} seeds x 3 modes")
    draws = pd.DataFrame(Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(_destruction_worker)(s, m) for s in range(args.null_seeds)
        for m in ("random_memory_scrambled", "random_memory", "random_space")))
    save(draws, "ablation_destruction_draws.csv")
    null = draws[draws["mode"] == "random_memory_scrambled"]
    mis = draws[draws["mode"] == "random_memory"]
    route_null = draws[draws["mode"] == "random_space"]

    p, n = b.run("test", temperatures=(BIG_T,))
    add("- distance kernel (uniform weights, T -> inf)", "A",
        "every retrieved neighbour weighted equally", p, {"n_members": n})

    p, n = b.run("test", prior_strengths=(0.0,))
    add("- prior shrinkage (lambda = 0)", "A",
        "unshrunk empirical neighbour frequency; Prop. 1's minimiser need not "
        "exist -- the implementation clips the logit at 1e-6", p, {"n_members": n})

    p, n = b.run("test", gates=(False,))
    add("- reliability gate (always off)", "A", "trust = 1 for every bar", p,
        {"n_members": n})
    p, n = b.run("test", gates=(True,))
    add("+ reliability gate (always on)", "A", "trust = consensus/(1+spread)", p,
        {"n_members": n})

    for k in cpc.SCALES:
        p, n = b.run("test", scales=(k,))
        add(f"- scale averaging (single k = {k})", "A",
            "one neighbourhood size instead of the four-scale average", p,
            {"n_members": n})

    p, n = b.run("test", balanced=True)
    add("+ class-balanced retrieval", "A",
        "k/2 nearest positives + k/2 nearest negatives instead of the k nearest",
        p, {"n_members": n})

    # retrieval only WITHOUT the fusion algebra: sigmoid(u), the pure fast state
    states = collect(b.memory["test"], b.test, b.groups["test"])
    distinct = {}
    for st in states:
        distinct.setdefault((st["space"], st["temperature"], st["prior_strength"]),
                            st["bias_bar"])
    ronly = np.mean([cpc.sigmoid(v[b.row_bar["test"]]) for v in distinct.values()],
                    axis=0)
    add("- fusion entirely (retrieval only, sigmoid(u))", "A",
        f"the fast state alone, averaged over the {len(distinct)} distinct "
        "(space, T, lambda) states; no anchor, no gate, no blend", ronly,
        {"n_members": len(distinct)})

    p, n = b.run("test", memory=b.memory_event["test"])
    add("- bar deduplication (event-level memory)", "A",
        "INVALID: a burst of posts sharing one bar contributes duplicate copies "
        "of one outcome to the memory", p, {"n_members": n, "n_memory": len(b.memory_event['test'])})

    # single member distribution
    preds = np.load(CACHE / "members_test.npy")
    specs = pd.read_csv(CACHE / "members_test.csv")
    m_bar = np.array([auc(bi.y_bar, bi.pool(preds[i])) for i in range(len(preds))])
    m_row = np.array([auc(bi.y_row, preds[i]) for i in range(len(preds))])
    pv = np.load(CACHE / "members_validation.npy")
    biv = b.bi["validation"]
    v_bar = np.array([auc(biv.y_bar, biv.pool(pv[i])) for i in range(len(pv))])
    member_frame = specs.assign(test_auc_bar=m_bar, test_auc_row=m_row,
                                val_auc_bar=v_bar)
    save(member_frame, "ablation_member_grid.csv")
    order = np.argsort(m_bar)
    add("- ensemble (single member, median of 72)", "A",
        "one grid member instead of the uniform average", preds[order[len(order) // 2]])
    add("- ensemble (single member, worst of 72)", "A", "grid minimum", preds[order[0]])
    add("- ensemble (single member, best of 72)", "A",
        "grid maximum; NOT selectable -- chosen on the test panel, shown as the "
        "ceiling of selection only", preds[order[-1]])
    vsel = int(np.argmax(v_bar))
    add("- ensemble (single member, validation-selected)", "A",
        "the only legal single-member arm: argmax of per-bar validation AUC",
        preds[vsel], {"selected": json.dumps(specs.iloc[vsel].to_dict())})

    log(f"  five-seed slow average: refitting {len(cpc.SLOW_SEEDS)} single-seed arms")
    seeds = pd.DataFrame(Parallel(n_jobs=min(args.jobs, 5), verbose=0)(
        delayed(_seed_worker)(s) for s in cpc.SLOW_SEEDS))
    save(seeds, "ablation_slow_seed_draws.csv")

    # ---- Block B: cumulative build-up --------------------------------------
    # Pre-declared order: start from the minimal single adapter and add one
    # averaging axis at a time until the shipped 72-member ensemble is reached.
    cumulative = [
        ("C1  single adapter (joint, k=32, T=1, lam=2, gate off, beta=0.5)",
         dict(spaces={"joint": cpc.RETRIEVAL_SPACES["joint"]}, scales=(32,),
              temperatures=(1.0,), prior_strengths=(2.0,), gates=(False,),
              blends=(0.5,))),
        ("C2  + scale averaging over k in {8,16,32,64}",
         dict(spaces={"joint": cpc.RETRIEVAL_SPACES["joint"]}, scales=cpc.SCALES,
              temperatures=(1.0,), prior_strengths=(2.0,), gates=(False,),
              blends=(0.5,))),
        ("C3  + temperature grid {0.25, 1}",
         dict(spaces={"joint": cpc.RETRIEVAL_SPACES["joint"]}, scales=cpc.SCALES,
              temperatures=cpc.TEMPERATURES, prior_strengths=(2.0,), gates=(False,),
              blends=(0.5,))),
        ("C4  + prior grid {2, 8}",
         dict(spaces={"joint": cpc.RETRIEVAL_SPACES["joint"]}, scales=cpc.SCALES,
              temperatures=cpc.TEMPERATURES, prior_strengths=cpc.PRIOR_STRENGTHS,
              gates=(False,), blends=(0.5,))),
        ("C5  + gate grid {off, on}",
         dict(spaces={"joint": cpc.RETRIEVAL_SPACES["joint"]}, scales=cpc.SCALES,
              temperatures=cpc.TEMPERATURES, prior_strengths=cpc.PRIOR_STRENGTHS,
              gates=cpc.GATES, blends=(0.5,))),
        ("C6  + blend grid {0.3, 0.5, 0.7}",
         dict(spaces={"joint": cpc.RETRIEVAL_SPACES["joint"]}, scales=cpc.SCALES,
              temperatures=cpc.TEMPERATURES, prior_strengths=cpc.PRIOR_STRENGTHS,
              gates=cpc.GATES, blends=cpc.BLENDS)),
        ("C7  + three routing spaces = R3-TTT-SF", dict()),
    ]
    for name, kw in cumulative:
        p, n = b.run("test", **kw)
        add(name, "B", "cumulative build-up, one averaging axis at a time", p,
            {"n_members": n})

    # ---- assemble the table -------------------------------------------------
    rows = []
    ref_scores = bi.scores(frozen)
    rows.append({
        "configuration": "Frozen slow model (5-seed average)", "block": "reference",
        "note": "no adaptation; the anchor every row is measured against",
        "n_members": 0, **ref_scores,
        **{f"vs_frozen_{k}": v for k, v in
           dict(bar_delta=0.0, bar_ci_lo=0.0, bar_ci_hi=0.0, bar_p_le_zero=np.nan,
                row_delta=0.0, row_ci_lo=0.0, row_ci_hi=0.0, row_p_le_zero=np.nan,
                bar_brier_delta=0.0, bar_brier_ci_lo=0.0, bar_brier_ci_hi=0.0,
                bar_brier_p_le_zero=np.nan).items()},
        "fraction_of_effect_retained": 0.0,
    })

    effect = auc(bi.y_bar, bi.pool(p_full)) - ref_scores["auc_bar"]

    def emit(name, spec):
        p = spec["p"]
        s = bi.scores(p)
        cf = contrast(bi, p, frozen, n_boot=nb, prefix="vs_frozen_")
        cs = contrast(bi, p, p_full, n_boot=nb, prefix="vs_full_")
        rec = {"configuration": name, "block": spec["block"], "note": spec["note"],
               **spec["extra"], **s, **cf, **cs,
               "fraction_of_effect_retained": cf["vs_frozen_bar_delta"] / effect,
               "beats_full_method_bar": bool(s["auc_bar"] > bi.scores(p_full)["auc_bar"])}
        return rec

    names = list(variants)
    log(f"  bootstrapping {len(names)} closed-form variants x 2 estimands")
    recs = Parallel(n_jobs=min(args.jobs, len(names)), verbose=0)(
        delayed(emit)(n, variants[n]) for n in names)
    rows.extend(recs)

    # replicate-based rows (nulls and seeds) -> mean +/- sd, no bootstrap P
    for tag, frame, blk, note in (
        ("- routing (routing coordinates replaced by i.i.d. noise)", route_null, "A",
         f"{len(route_null)} replicates; the router is real, the routing space is "
         "noise for both memory and stream"),
        ("- retrieval (k uniformly random memory bars; CORRECTED null)", null, "A",
         f"{len(null)} replicates; distances zeroed AND order permuted, so the k "
         "retrieved slots are k uniformly random memory bars"),
        ("[published mis-specified random-memory control]", mis, "A-note",
         f"{len(mis)} replicates; NOT a retrieval-destruction control -- it sorts "
         "its random 256-item pool by true distance, so it is the memory-size "
         "curve at an 11.4% subsample"),
    ):
        rows.append({
            "configuration": tag, "block": blk,
            "note": note, "n_members": 72,
            "auc_bar": float(frame["auc_bar"].mean()),
            "auc_row": float(frame["auc_row"].mean()),
            "brier_bar": float(frame["brier_bar"].mean()),
            "auc_bar_sd": float(frame["auc_bar"].std(ddof=1)),
            "retrieval_only_auc_bar": float(frame["retrieval_only_auc_bar"].mean()),
            "retrieval_only_auc_row": float(frame["retrieval_only_auc_row"].mean()),
            "vs_frozen_bar_delta": float(frame["auc_bar"].mean() - ref_scores["auc_bar"]),
            "vs_frozen_bar_ci_lo": float(np.percentile(frame["auc_bar"], 2.5) - ref_scores["auc_bar"]),
            "vs_frozen_bar_ci_hi": float(np.percentile(frame["auc_bar"], 97.5) - ref_scores["auc_bar"]),
            "vs_frozen_row_delta": float(frame["auc_row"].mean() - ref_scores["auc_row"]),
            "fraction_of_effect_retained":
                float(frame["auc_bar"].mean() - ref_scores["auc_bar"]) / effect,
            "beats_full_method_bar": False,
        })

    rows.append({
        "configuration": "- five-seed slow average (single seed, mean of 5)",
        "block": "A",
        "note": "the frozen anchor is refit from one seed; spread over the five "
                "seeds is the sd column",
        "n_members": 72,
        "auc_bar": float(seeds["sf_auc_bar"].mean()),
        "auc_row": float(seeds["sf_auc_row"].mean()),
        "brier_bar": float(seeds["sf_brier_bar"].mean()),
        "auc_bar_sd": float(seeds["sf_auc_bar"].std(ddof=1)),
        "vs_frozen_bar_delta": float(seeds["delta_bar"].mean()),
        "vs_frozen_bar_ci_lo": float(seeds["delta_bar"].min()),
        "vs_frozen_bar_ci_hi": float(seeds["delta_bar"].max()),
        "fraction_of_effect_retained": float(seeds["delta_bar"].mean()) / effect,
        "beats_full_method_bar": bool(seeds["sf_auc_bar"].mean() > bi.scores(p_full)["auc_bar"]),
        "extra_frozen_auc_bar_mean": float(seeds["frozen_auc_bar"].mean()),
        "extra_frozen_auc_bar_sd": float(seeds["frozen_auc_bar"].std(ddof=1)),
    })

    frame = pd.DataFrame(rows)
    frame.loc[frame["configuration"] == "- fusion entirely (retrieval only, sigmoid(u))",
              "retrieval_only_auc_bar"] = auc(bi.y_bar, bi.pool(ronly))
    save(frame, "ablation_table.csv")
    cols = ["configuration", "auc_bar", "auc_row", "brier_bar", "vs_frozen_bar_delta",
            "vs_frozen_bar_ci_lo", "vs_frozen_bar_ci_hi", "vs_frozen_bar_p_le_zero",
            "fraction_of_effect_retained", "beats_full_method_bar"]
    log("\n" + frame[[c for c in cols if c in frame.columns]].to_string(index=False))
    return frame


# =============================================================================
# stage: sensitivity
# =============================================================================
def _memory_worker(fraction: float, draw: int):
    b = bench()
    out = {"fraction": fraction, "draw": draw}
    for key in ("test", "validation"):
        mem = b.memory[key]
        n = len(mem)
        take = min(n, max(8, int(round(fraction * n))))
        rng = np.random.default_rng(500_000 + int(fraction * 1000) * 97 + draw)
        keep = np.sort(rng.choice(n, size=take, replace=False))
        sub = cpc.BarMemory(mem.frame.iloc[keep].reset_index(drop=True),
                            mem.labels[keep])
        p, _ = b.run(key, memory=sub)
        bi = b.bi[key]
        out[f"{key}_auc_bar"] = auc(bi.y_bar, bi.pool(p))
        out[f"{key}_auc_row"] = auc(bi.y_row, p)
        if key == "test":
            out["p_test"] = p
            out["n_memory"] = int(take)
    return out


def _axis_point(axis: str, value, kw: dict):
    """One pinned-axis ensemble marginal on both streams."""
    b = bench()
    rec = {"axis": axis, "value": str(value)}
    for key in ("test", "validation"):
        p, n = b.run(key, **kw)
        bi = b.bi[key]
        tag = "test" if key == "test" else "val"
        rec[f"{tag}_auc_bar"] = auc(bi.y_bar, bi.pool(p))
        rec[f"{tag}_auc_row"] = auc(bi.y_row, p)
        rec[f"{tag}_brier_bar"] = brier(bi.y_bar, bi.pool(p))
        rec["n_members"] = n
        if key == "test":
            rec["_p"] = p
    return rec


def stage_sensitivity(args) -> pd.DataFrame:
    log("stage sensitivity -- one axis at a time, ensemble marginals, both streams")
    b = bench()
    bi = b.bi["test"]
    biv = b.bi["validation"]
    frozen = b.frozen["test"]
    nb = args.n_boot

    jobs: list[tuple[str, object, dict]] = []
    for beta in np.round(np.arange(0.0, 1.0001, 0.1), 3):
        jobs.append(("beta", float(beta), dict(blends=(float(beta),))))
    for T in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, BIG_T):
        jobs.append(("temperature", "inf" if T == BIG_T else T, dict(temperatures=(T,))))
    for lam in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        jobs.append(("prior_strength", lam, dict(prior_strengths=(lam,))))
    n_mem = len(b.memory["test"])
    for k in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, n_mem):
        jobs.append(("k", "full" if k == n_mem else k, dict(scales=(k,), max_k=k)))
    for space in list(cpc.RETRIEVAL_SPACES) + ["arrival_only"]:
        if space == "arrival_only":
            sp = {"arrival_only": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]}
        else:
            sp = {space: cpc.RETRIEVAL_SPACES[space]}
        jobs.append(("routing_space", space, dict(spaces=sp)))
    for g in (False, True):
        jobs.append(("gate", g, dict(gates=(g,))))

    log(f"  {len(jobs)} pinned-axis points x 2 streams")
    recs = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(_axis_point)(a, v, kw) for a, v, kw in jobs)

    preds_axis = [rec.pop("_p") for rec in recs]
    cfs = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(contrast)(bi, p, frozen, n_boot=nb) for p in preds_axis)
    frame = pd.DataFrame([{**r, **c} for r, c in zip(recs, cfs)])

    # ---- memory size --------------------------------------------------------
    fracs = (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00)
    log(f"  memory-size axis: {len(fracs)} fractions x {args.mem_draws} draws")
    mem = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(_memory_worker)(f, d) for f in fracs
        for d in range(1 if f == 1.00 else args.mem_draws))
    mem_rows = []
    raw = []
    pooled = {f: np.mean(np.stack([m["p_test"] for m in mem if m["fraction"] == f],
                                  axis=0), axis=0) for f in fracs}
    mem_cfs = dict(zip(fracs, Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(contrast)(bi, pooled[f], frozen, n_boot=nb) for f in fracs)))
    for f in fracs:
        sub = [m for m in mem if m["fraction"] == f]
        cf = mem_cfs[f]
        mem_rows.append({
            "axis": "memory_fraction", "value": f"{f:.2f}",
            "test_auc_bar": float(np.mean([m["test_auc_bar"] for m in sub])),
            "test_auc_bar_sd": float(np.std([m["test_auc_bar"] for m in sub], ddof=1))
                               if len(sub) > 1 else 0.0,
            "test_auc_bar_pooled": auc(bi.y_bar, bi.pool(pooled[f])),
            "test_auc_row": float(np.mean([m["test_auc_row"] for m in sub])),
            "val_auc_bar": float(np.mean([m["validation_auc_bar"] for m in sub])),
            "n_members": 72, "n_draws": len(sub),
            "n_memory": int(np.mean([m["n_memory"] for m in sub])),
            **cf})
        for m in sub:
            raw.append({k: v for k, v in m.items() if k != "p_test"})
    save(pd.DataFrame(raw), "sensitivity_memory_draws.csv")
    frame = pd.concat([frame, pd.DataFrame(mem_rows)], ignore_index=True)

    # ---- ensemble size ------------------------------------------------------
    preds = np.load(CACHE / "members_test.npy")
    predsv = np.load(CACHE / "members_validation.npy")
    sizes = (1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 60, 72)
    log(f"  ensemble-size axis: {len(sizes)} sizes x {args.subset_draws} random subsets")
    ens_rows, ens_raw, ens_stats, ens_pooled = [], [], {}, {}
    for m in sizes:
        rng = np.random.default_rng(700_000 + m)
        n_draw = 1 if m == 72 else args.subset_draws
        a_bar, a_row, a_val = [], [], []
        pm_sum = np.zeros(preds.shape[1])
        for d in range(n_draw):
            pick = rng.choice(preds.shape[0], size=m, replace=False)
            pp = preds[pick].mean(axis=0)
            pv = predsv[pick].mean(axis=0)
            a_bar.append(auc(bi.y_bar, bi.pool(pp)))
            a_row.append(auc(bi.y_row, pp))
            a_val.append(auc(biv.y_bar, biv.pool(pv)))
            pm_sum += pp
            ens_raw.append({"size": m, "draw": d, "test_auc_bar": a_bar[-1],
                            "test_auc_row": a_row[-1], "val_auc_bar": a_val[-1]})
        ens_stats[m] = (a_bar, a_row, a_val, n_draw)
        ens_pooled[m] = pm_sum / n_draw
    ens_cfs = dict(zip(sizes, Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(contrast)(bi, ens_pooled[m], frozen, n_boot=nb) for m in sizes)))
    for m in sizes:
        a_bar, a_row, a_val, n_draw = ens_stats[m]
        cf = ens_cfs[m]
        ens_rows.append({
            "axis": "ensemble_size", "value": str(m), "n_members": m,
            "test_auc_bar": float(np.mean(a_bar)),
            "test_auc_bar_sd": float(np.std(a_bar, ddof=1)) if n_draw > 1 else 0.0,
            "test_auc_bar_p2.5": float(np.percentile(a_bar, 2.5)),
            "test_auc_bar_p97.5": float(np.percentile(a_bar, 97.5)),
            "test_auc_row": float(np.mean(a_row)),
            "val_auc_bar": float(np.mean(a_val)), "n_draws": n_draw, **cf})
    save(pd.DataFrame(ens_raw), "sensitivity_ensemble_draws.csv")
    frame = pd.concat([frame, pd.DataFrame(ens_rows)], ignore_index=True)

    save(frame, "sensitivity_curves.csv")

    # ---- decision threshold -------------------------------------------------
    p_full = np.load(CACHE / "sf_test.npy")
    pb_full, pb_frozen = bi.pool(p_full), bi.pool(frozen)
    th_rows = []
    for tau in np.round(np.arange(0.05, 0.951, 0.025), 4):
        rec = {"threshold": float(tau)}
        for tag, s in (("sf", pb_full), ("frozen", pb_frozen)):
            yhat = (s >= tau).astype(int)
            tp = int(((yhat == 1) & (bi.y_bar == 1)).sum())
            fp = int(((yhat == 1) & (bi.y_bar == 0)).sum())
            fn_ = int(((yhat == 0) & (bi.y_bar == 1)).sum())
            tn = int(((yhat == 0) & (bi.y_bar == 0)).sum())
            tpr = tp / max(tp + fn_, 1)
            tnr = tn / max(tn + fp, 1)
            prec = tp / max(tp + fp, 1)
            rec |= {f"{tag}_balanced_accuracy": 0.5 * (tpr + tnr),
                    f"{tag}_accuracy": (tp + tn) / len(bi.y_bar),
                    f"{tag}_f1": 2 * prec * tpr / max(prec + tpr, 1e-12),
                    f"{tag}_flag_rate": (tp + fp) / len(bi.y_bar)}
        rec["delta_balanced_accuracy"] = (rec["sf_balanced_accuracy"]
                                          - rec["frozen_balanced_accuracy"])
        rec["delta_f1"] = rec["sf_f1"] - rec["frozen_f1"]
        th_rows.append(rec)
    th = pd.DataFrame(th_rows)
    save(th, "sensitivity_threshold.csv")

    # Quantile-MATCHED thresholds: each arm is cut at its own quantile so the
    # two arms flag the same number of bars.  This removes the calibration
    # confound and isolates ranking; the absolute-threshold sweep above keeps it.
    q_rows = []
    for flag in np.round(np.arange(0.05, 0.951, 0.025), 4):
        rec = {"flag_rate": float(flag)}
        for tag, s in (("sf", pb_full), ("frozen", pb_frozen)):
            tau = float(np.quantile(s, 1.0 - flag))
            yhat = (s >= tau).astype(int)
            tp = int(((yhat == 1) & (bi.y_bar == 1)).sum())
            fp = int(((yhat == 1) & (bi.y_bar == 0)).sum())
            fn_ = int(((yhat == 0) & (bi.y_bar == 1)).sum())
            tn = int(((yhat == 0) & (bi.y_bar == 0)).sum())
            tpr = tp / max(tp + fn_, 1)
            tnr = tn / max(tn + fp, 1)
            prec = tp / max(tp + fp, 1)
            rec |= {f"{tag}_threshold": tau, f"{tag}_precision": prec,
                    f"{tag}_recall": tpr,
                    f"{tag}_balanced_accuracy": 0.5 * (tpr + tnr),
                    f"{tag}_f1": 2 * prec * tpr / max(prec + tpr, 1e-12),
                    f"{tag}_realised_flag_rate": (tp + fp) / len(bi.y_bar)}
        rec["delta_balanced_accuracy"] = (rec["sf_balanced_accuracy"]
                                          - rec["frozen_balanced_accuracy"])
        rec["delta_precision"] = rec["sf_precision"] - rec["frozen_precision"]
        q_rows.append(rec)
    save(pd.DataFrame(q_rows), "sensitivity_threshold_matched.csv")
    return frame


# =============================================================================
# stage: figs
# =============================================================================
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.8,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.6,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "figure.dpi": 200,
    })
    return plt


INK, BLUE, RED, GREY, GOLD = "#1b1b1b", "#2f5d8a", "#a83232", "#8f8f8f", "#d8b24a"


def stage_figs(args) -> None:
    log("stage figs")
    plt = _style()
    from matplotlib.ticker import FuncFormatter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    abl = pd.read_csv(OUT / "ablation_table.csv")
    ctx = json.loads((OUT / "context.json").read_text())
    frozen_bar = ctx["test"]["frozen"]["auc_bar"]
    full_bar = ctx["test"]["sf"]["auc_bar"]
    effect = full_bar - frozen_bar

    # ------------------------------------------------------------------ Fig 1
    SHORT = {
        "R3-TTT-SF (full, 72 members)": "R3-TTT-SF (full, 72 members)",
        "- frozen anchor (beta = 1)": r"$-$ frozen anchor ($\beta=1$)",
        "- distance kernel (uniform weights, T -> inf)":
            r"$-$ distance kernel ($T\!\to\!\infty$)",
        "- prior shrinkage (lambda = 0)": r"$-$ prior shrinkage ($\lambda=0$)",
        "- reliability gate (always off)": r"$-$ reliability gate (off)",
        "+ reliability gate (always on)": r"$+$ reliability gate (on)",
        "- scale averaging (single k = 8)": r"$-$ scale averaging ($k=8$)",
        "- scale averaging (single k = 16)": r"$-$ scale averaging ($k=16$)",
        "- scale averaging (single k = 32)": r"$-$ scale averaging ($k=32$)",
        "- scale averaging (single k = 64)": r"$-$ scale averaging ($k=64$)",
        "+ class-balanced retrieval": r"$+$ class-balanced retrieval",
        "- fusion entirely (retrieval only, sigmoid(u))":
            r"$-$ fusion entirely (retrieval only)",
        "- bar deduplication (event-level memory)":
            r"$-$ bar dedup (event memory) [invalid]",
        "- ensemble (single member, median of 72)": r"$-$ ensemble (member, median)",
        "- ensemble (single member, worst of 72)": r"$-$ ensemble (member, worst)",
        "- ensemble (single member, best of 72)":
            r"$-$ ensemble (member, best) [not selectable]",
        "- ensemble (single member, validation-selected)":
            r"$-$ ensemble (member, validation-selected)",
        "- routing (routing coordinates replaced by i.i.d. noise)":
            r"$-$ routing (coordinates replaced by noise)",
        "- retrieval (k uniformly random memory bars; CORRECTED null)":
            r"$-$ retrieval (random memory bars, corrected null)",
        "- five-seed slow average (single seed, mean of 5)":
            r"$-$ five-seed slow average",
        "C1  single adapter (joint, k=32, T=1, lam=2, gate off, beta=0.5)":
            r"C1 single adapter (grid centroid)",
        "C2  + scale averaging over k in {8,16,32,64}": r"C2 $+$ scale averaging",
        "C3  + temperature grid {0.25, 1}": r"C3 $+$ temperature grid",
        "C4  + prior grid {2, 8}": r"C4 $+$ prior grid",
        "C5  + gate grid {off, on}": r"C5 $+$ gate grid",
        "C6  + blend grid {0.3, 0.5, 0.7}": r"C6 $+$ blend grid",
        "C7  + three routing spaces = R3-TTT-SF": r"C7 $+$ routing spaces $=$ full",
    }
    show = abl[abl["block"].isin(["reference", "A", "B"])].copy()
    show = show[show["configuration"] != "Frozen slow model (5-seed average)"]
    show = show.sort_values("vs_frozen_bar_delta")

    fig, ax = plt.subplots(figsize=(6.0, 5.9))
    for i, (_, r) in enumerate(show.iterrows()):
        d = r["vs_frozen_bar_delta"]
        lo, hi = r.get("vs_frozen_bar_ci_lo", np.nan), r.get("vs_frozen_bar_ci_hi", np.nan)
        vlo, vhi = r.get("vs_full_bar_ci_lo", np.nan), r.get("vs_full_bar_ci_hi", np.nan)
        is_full = str(r["configuration"]).startswith("R3-TTT-SF")
        beats = bool(r.get("beats_full_method_bar", False))
        sig = beats and np.isfinite(vlo) and vlo > 0
        color = INK if is_full else (RED if sig else (BLUE if not beats else "#cf8f8f"))
        ax.barh(i, d, height=0.62, color=color, edgecolor="none", zorder=2,
                alpha=1.0 if (is_full or sig) else 0.85)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=INK, lw=0.8, zorder=3)
            for x in (lo, hi):
                ax.plot([x, x], [i - 0.15, i + 0.15], color=INK, lw=0.8, zorder=3)
    ax.axvline(0.0, color=INK, lw=0.8, zorder=1)
    ax.axvline(effect, color=GREY, lw=0.9, ls="--", zorder=1)
    ax.set_yticks(np.arange(len(show)))
    ax.set_yticklabels([SHORT.get(c, str(c)) for c in show["configuration"]])
    ax.set_xlabel(r"$\Delta$ per-bar AUC vs the frozen slow model "
                  f"({frozen_bar:.4f}, {ctx['test']['n_bars']} bars)")
    ax.set_ylim(-0.7, len(show) - 0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.3f}"))
    ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor=INK, label="the shipped method"),
        Patch(facecolor=RED, label="beats it, paired interval excludes zero"),
        Patch(facecolor="#cf8f8f", label="beats it on the point estimate only"),
        Patch(facecolor=BLUE, label="does not beat it"),
    ], frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.002), ncol=2,
        handlelength=1.1, borderpad=0.2, columnspacing=1.2)
    ax.set_title("Component ablation, clean panel, primary per-bar estimand",
                 loc="left", pad=32)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {OUT/'fig_ablation.pdf'}")

    # ------------------------------------------------------------------ Fig 2
    sen = pd.read_csv(OUT / "sensitivity_curves.csv")
    th = pd.read_csv(OUT / "sensitivity_threshold.csv")
    thm = pd.read_csv(OUT / "sensitivity_threshold_matched.csv")
    ens_raw = pd.read_csv(OUT / "sensitivity_ensemble_draws.csv")

    def curve(ax, axis, labels=None, shaded=(), xlabel="", title="", rot=0,
              tick_fs=6.6):
        """Categorical x-position curve: delta band + validation twin."""
        d = sen[sen["axis"] == axis].reset_index(drop=True)
        xs = np.arange(len(d))
        lab = labels if labels is not None else [str(v) for v in d["value"]]
        ax.fill_between(xs, d["bar_ci_lo"], d["bar_ci_hi"], color=BLUE, alpha=0.15,
                        lw=0, zorder=1)
        for j, v in enumerate(lab):
            if v in shaded:
                ax.axvspan(j - 0.3, j + 0.3, color=GOLD, alpha=0.30, lw=0, zorder=0)
        ax.plot(xs, d["bar_delta"], color=BLUE, lw=1.3, marker="o", ms=2.6, zorder=3)
        ax.axhline(0.0, color=INK, lw=0.7, zorder=2)
        ax.axhline(effect, color=GREY, lw=0.8, ls="--", zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(lab, rotation=rot, fontsize=tick_fs,
                           ha="right" if rot else "center")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta$ AUC (bar)")
        ax.set_title(title, loc="left", pad=4)
        ax2 = ax.twinx()
        ax2.plot(xs, d["val_auc_bar"], color=RED, lw=0.9, ls=":", marker="s", ms=1.9,
                 zorder=3)
        ax2.set_ylabel("validation AUC", color=RED, fontsize=6.4)
        ax2.tick_params(axis="y", colors=RED, labelsize=5.8)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(RED)
        ax2.spines["right"].set_linewidth(0.5)
        ax2.spines["top"].set_visible(False)
        return d

    fig, axes = plt.subplots(4, 2, figsize=(6.9, 9.0))

    d = curve(axes[0, 0], "beta", shaded=("0.3", "0.5", "0.7"), tick_fs=6.0,
              xlabel=r"blend $\beta$",
              title=r"(a) blend $\beta$ — strictly monotone on test")
    axes[0, 0].annotate(r"$\beta\!=\!0$ recovers the frozen model exactly",
                        xy=(9.85, 0.0035), fontsize=6.0, color=INK, ha="right")
    axes[0, 0].annotate(r"validation peaks at $\beta\!=\!0.7$, not at $\beta\!=\!1$",
                        xy=(9.85, 0.0012), fontsize=6.0, color=RED, ha="right")

    tl = ["0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10", r"$\infty$"]
    curve(axes[0, 1], "temperature", labels=tl, shaded=("0.25", "1"), rot=0,
          tick_fs=6.0,
          xlabel="kernel temperature $T$",
          title=r"(b) temperature $T$ — flat, incl. $T\!\to\!\infty$")

    curve(axes[1, 0], "prior_strength", shaded=("2.0", "8.0"), tick_fs=6.0,
          xlabel=r"prior strength $\lambda$",
          title=r"(c) prior strength $\lambda$ — monotone down")

    kl = [str(v) for v in sen.loc[sen["axis"] == "k", "value"]]
    curve(axes[1, 1], "k", labels=kl, shaded=("8", "16", "32", "64"), rot=45,
          tick_fs=5.8, xlabel="neighbourhood size $k$",
          title="(d) neighbourhood size $k$ — interior optimum")

    ml = [f"{100*float(v):.0f}%" for v in sen.loc[sen["axis"] == "memory_fraction", "value"]]
    curve(axes[2, 0], "memory_fraction", labels=ml, shaded=("100%",), rot=45,
          tick_fs=6.0, xlabel="fraction of the memory retained",
          title="(e) memory size — monotone; about half the gain\nfrom a tenth of the memory")

    # (f) ensemble size: the informative band is the SPREAD OVER SUBSETS
    ax = axes[2, 1]
    g = ens_raw.groupby("size")["test_auc_bar"]
    sizes = np.array(sorted(ens_raw["size"].unique()))
    xs = np.arange(len(sizes))
    mean = g.mean().reindex(sizes).to_numpy() - frozen_bar
    lo = g.quantile(0.025).reindex(sizes).to_numpy() - frozen_bar
    hi = g.quantile(0.975).reindex(sizes).to_numpy() - frozen_bar
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.15, lw=0, zorder=1)
    ax.plot(xs, mean, color=BLUE, lw=1.3, marker="o", ms=2.6, zorder=3)
    ax.axhline(0.0, color=INK, lw=0.7)
    ax.axhline(effect, color=GREY, lw=0.8, ls="--")
    ax.axvspan(len(sizes) - 1.3, len(sizes) - 0.7, color=GOLD, alpha=0.30, lw=0, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in sizes], rotation=0, fontsize=5.8)
    ax.set_xlabel("members averaged (random subsets of 72)")
    ax.set_ylabel(r"$\Delta$ AUC (bar)")
    ax.set_title("(f) ensemble size — the mean is flat from 4 members;\n"
                 "only the spread shrinks", loc="left", pad=4)
    ax2 = ax.twinx()
    ax2.plot(xs, g.std().reindex(sizes).to_numpy(), color=RED, lw=0.9, ls=":",
             marker="s", ms=1.9)
    ax2.set_ylabel("sd over subsets", color=RED, fontsize=6.4)
    ax2.tick_params(axis="y", colors=RED, labelsize=5.8)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(RED)
    ax2.spines["top"].set_visible(False)

    # (g) routing space + gate
    ax = axes[3, 0]
    bars = sen[sen["axis"].isin(["routing_space", "gate"])].sort_values("bar_delta")
    yy = np.arange(len(bars))
    ax.barh(yy, bars["bar_delta"], color=BLUE, alpha=0.85, height=0.6, zorder=2)
    for i, (_, r) in enumerate(bars.iterrows()):
        ax.plot([r["bar_ci_lo"], r["bar_ci_hi"]], [i, i], color=INK, lw=0.8, zorder=3)
    ax.axvline(0, color=INK, lw=0.7)
    ax.axvline(effect, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([("gate " + ("on" if v == "True" else "off")) if a == "gate"
                        else str(v).replace("_", " ")
                        for a, v in zip(bars["axis"], bars["value"])])
    ax.set_xlabel(r"$\Delta$ AUC (bar)")
    ax.set_title("(g) routing space and reliability gate", loc="left", pad=4)

    # (h) decision threshold
    ax = axes[3, 1]
    ax.plot(th["sf_flag_rate"], th["delta_balanced_accuracy"], color=GREY, lw=1.2,
            ls="--", label="fixed absolute cut-off")
    ax.plot(thm["flag_rate"], thm["delta_balanced_accuracy"], color=BLUE, lw=1.2,
            label="quantile-matched cut-off")
    ax.axhline(0.0, color=INK, lw=0.7)
    ax.set_xlabel("share of the 525 bars flagged")
    ax.set_ylabel(r"$\Delta$ balanced accuracy (bar)")
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)
    ax.set_title("(h) decision threshold — the ranking gain does not\n"
                 "survive a fixed absolute cut-off", loc="left", pad=4)

    fig.tight_layout(h_pad=1.6, w_pad=2.2)
    fig.savefig(OUT / "fig_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {OUT/'fig_sensitivity.pdf'}")


# =============================================================================
# stage: report
# =============================================================================
def _fmt(v, n=4, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:+.{n}f}" if sign else f"{v:.{n}f}"


def stage_report(args) -> None:
    log("stage report")
    ctx = json.loads((OUT / "context.json").read_text())
    abl = pd.read_csv(OUT / "ablation_table.csv")
    sen = pd.read_csv(OUT / "sensitivity_curves.csv")
    th = pd.read_csv(OUT / "sensitivity_threshold.csv")
    frozen_bar = ctx["test"]["frozen"]["auc_bar"]
    full_bar = ctx["test"]["sf"]["auc_bar"]
    effect = full_bar - frozen_bar

    proto = cpc.protocol_json(
        track="ablation_sensitivity",
        purpose=("component ablation and one-axis sensitivity for the two "
                 "sections the paper lacks; CLEAN panel, PRIMARY per-bar estimand"),
        primary_estimand={
            "unit": "aligned 15-minute outcome bar",
            "bar_score": "mean of the arm's per-post probabilities inside the bar",
            "n_test_bars": ctx["test"]["n_bars"],
            "n_test_posts": ctx["test"]["n_posts"],
            "n_test_days": ctx["test"]["n_days"],
            "frozen_auc_bar": frozen_bar, "sf_auc_bar": full_bar,
            "delta_auc_bar": effect,
        },
        secondary_estimand={"unit": "post", "frozen_auc_row": ctx["test"]["frozen"]["auc_row"],
                            "sf_auc_row": ctx["test"]["sf"]["auc_row"]},
        validation_stream={
            "purpose": "reported only to show validation and test disagree; "
                       "NOTHING in the headline is selected on it",
            "window": "2025-10-01 .. 2025-12-31",
            "n_bars": ctx["validation"]["n_bars"],
            "slow_model_and_memory": "train only (datetime < 2025-10-01)",
            "frozen_auc_bar": ctx["validation"]["frozen"]["auc_bar"],
            "sf_auc_bar": ctx["validation"]["sf"]["auc_bar"],
        },
        uncertainty={"estimator": "paired 5-day moving-block bootstrap",
                     "clustered_by": "calendar day", "n_boot": args.n_boot,
                     "seed": SEED, "reported": "95% percentile interval and the "
                                               "bootstrap tail P(delta <= 0)"},
        replicate_controls={"random_memory_null_seeds": args.null_seeds,
                            "memory_draws_per_fraction": args.mem_draws,
                            "member_subset_draws_per_size": args.subset_draws},
        corrected_null=("the published random-memory control is mis-specified: it "
                        "sorts a random 256-item pool by TRUE distance and then "
                        "takes the nearest k, so it is a memory-SUBSAMPLING "
                        "control.  This track uses random_memory_scrambled, in "
                        "which the k retrieved slots are k uniformly random "
                        "memory bars."),
        blas_threads=1,
        never_selected_on_test=True,
        environment_map=("CLEAN.  Do not difference these levels against the "
                         "canonical (0.8273/0.8362), regenerated common-protocol "
                         "(0.8324/0.8422) or matched-backbone environments."),
    )
    jdump(proto, "protocol.json")

    # ---------------- LaTeX --------------------------------------------------
    TEX = {
        "- routing (routing coordinates replaced by i.i.d. noise)":
            r"$-$ routing (routing coordinates replaced by i.i.d.\ noise)",
        "- retrieval (k uniformly random memory bars; CORRECTED null)":
            r"$-$ retrieval (the $k$ slots filled with random memory bars)",
        "- distance kernel (uniform weights, T -> inf)":
            r"$-$ distance kernel ($T\!\to\!\infty$, uniform neighbour weights)",
        "- prior shrinkage (lambda = 0)": r"$-$ prior shrinkage ($\lambda=0$)",
        "- reliability gate (always off)": r"$-$ reliability gate (forced off)",
        "+ reliability gate (always on)": r"$+$ reliability gate (forced on)",
        "+ class-balanced retrieval": r"$+$ class-balanced retrieval",
        "- scale averaging (single k = 8)": r"$-$ scale averaging (single $k=8$)",
        "- scale averaging (single k = 16)": r"$-$ scale averaging (single $k=16$)",
        "- scale averaging (single k = 32)": r"$-$ scale averaging (single $k=32$)",
        "- scale averaging (single k = 64)": r"$-$ scale averaging (single $k=64$)",
        "- ensemble (single member, median of 72)":
            r"$-$ ensemble (single member, grid median)",
        "- ensemble (single member, worst of 72)":
            r"$-$ ensemble (single member, grid worst)",
        "- ensemble (single member, best of 72)":
            r"$-$ ensemble (single member, grid best; \emph{not selectable})",
        "- ensemble (single member, validation-selected)":
            r"$-$ ensemble (single member, validation-selected)",
        "- five-seed slow average (single seed, mean of 5)":
            r"$-$ five-seed slow average (one seed, mean of five)",
        "- frozen anchor (beta = 1)":
            r"$-$ frozen anchor ($\beta=1$)",
        "- fusion entirely (retrieval only, sigmoid(u))":
            r"$-$ fusion entirely (retrieval only, $\sigma(u)$)",
        "- bar deduplication (event-level memory)":
            r"$-$ bar deduplication (event-level memory; \emph{invalid})",
        "C1  single adapter (joint, k=32, T=1, lam=2, gate off, beta=0.5)":
            r"C1\quad single adapter at the grid centroid",
        "C2  + scale averaging over k in {8,16,32,64}":
            r"C2\quad $+$ scale averaging, $k\in\{8,16,32,64\}$",
        "C3  + temperature grid {0.25, 1}":
            r"C3\quad $+$ temperature grid $\{0.25,1\}$",
        "C4  + prior grid {2, 8}": r"C4\quad $+$ prior grid $\{2,8\}$",
        "C5  + gate grid {off, on}": r"C5\quad $+$ gate grid $\{$off, on$\}$",
        "C6  + blend grid {0.3, 0.5, 0.7}":
            r"C6\quad $+$ blend grid $\{0.3,0.5,0.7\}$",
        "C7  + three routing spaces = R3-TTT-SF":
            r"C7\quad $+$ three routing spaces $\;=\;$ R3-TTT-SF",
        "[published mis-specified random-memory control]":
            r"\emph{$-$ retrieval, as published (mis-specified; see text)}",
    }
    ORDER_A = [
        "- routing (routing coordinates replaced by i.i.d. noise)",
        "- retrieval (k uniformly random memory bars; CORRECTED null)",
        "[published mis-specified random-memory control]",
        "- distance kernel (uniform weights, T -> inf)",
        "- five-seed slow average (single seed, mean of 5)",
        "+ class-balanced retrieval",
        "+ reliability gate (always on)",
        "- ensemble (single member, worst of 72)",
        "- ensemble (single member, median of 72)",
        "- scale averaging (single k = 16)",
        "- scale averaging (single k = 64)",
        "- scale averaging (single k = 8)",
        "- scale averaging (single k = 32)",
        "- bar deduplication (event-level memory)",
        "- ensemble (single member, best of 72)",
        "- ensemble (single member, validation-selected)",
        "- fusion entirely (retrieval only, sigmoid(u))",
        "- prior shrinkage (lambda = 0)",
        "- reliability gate (always off)",
        "- frozen anchor (beta = 1)",
    ]

    def row(cfg):
        r = abl[abl["configuration"] == cfg]
        if r.empty:
            return None
        r = r.iloc[0]

        def g(k):
            return float(r[k]) if k in r and pd.notna(r[k]) else np.nan

        beats = bool(r.get("beats_full_method_bar", False))
        vlo = g("vs_full_bar_ci_lo")
        sig = beats and np.isfinite(vlo) and vlo > 0
        name = TEX.get(cfg, str(cfg))
        if beats:
            name = r"\textcolor{BrickRed}{" + name + "}"
        if sig:
            name += r"$^{\dagger}$"
        d, lo, hi = g("vs_frozen_bar_delta"), g("vs_frozen_bar_ci_lo"), g("vs_frozen_bar_ci_hi")
        p = g("vs_frozen_bar_p_le_zero")
        ci = (f"${_fmt(d)}$ $[{_fmt(lo)},{_fmt(hi)}]$" if np.isfinite(lo)
              else f"${_fmt(d)}$")
        pcell = ("---" if not np.isfinite(p)
                 else (r"$<\!0.0002$" if p < 2e-4 else f"${p:.4f}$"))
        frac = g("fraction_of_effect_retained")
        fcell = "--" if not np.isfinite(frac) else "$" + f"{100 * frac:.0f}" + r"\%$"
        cells = [name, f"${g('auc_bar'):.4f}$", f"${g('auc_row'):.4f}$",
                 f"${g('brier_bar'):.4f}$", ci, pcell, fcell]
        return " & ".join(cells) + r" \\"

    nboot_tex = f"{args.n_boot:,}".replace(",", r"{,}")
    caption = (
        r"\textbf{Component ablation of R3-TTT-SF, clean panel, primary per-bar "
        r"estimand} (frozen $" + f"{frozen_bar:.4f}" + r"$, "
        + f"{ctx['test']['n_bars']}" + r" outcome bars, "
        + f"{ctx['test']['n_days']}" + r" test days, "
        + f"{ctx['test']['n_posts']}" + r" posts). Every row takes zero gradient "
        r"steps and adapts zero parameters. Block~A removes one ingredient at a "
        r"time from the shipped ensemble, ordered by what it costs; Block~B "
        r"rebuilds the method from a single adapter at the centroid of the "
        r"pre-specified grid, adding one averaging axis per rung. $\Delta$ and its "
        r"paired five-day moving-block bootstrap interval are against the frozen "
        r"slow model, clustered by calendar day, " + nboot_tex + r" replicates, "
        r"seed~42; $P$ is the bootstrap tail $P(\Delta\!\le\!0)$, not a "
        r"frequentist $p$-value; lower Brier is better; \emph{retained} is "
        r"$\Delta$ as a share of the shipped method's $+" + f"{effect:.4f}" + r"$. "
        r"Replicate-based rows (the two random-memory controls, "
        + f"{args.null_seeds}" + r" draws each, and the single-seed anchor, five "
        r"seeds) report the replicate mean and the 2.5--97.5 percentile of the "
        r"replicate distribution instead of a bootstrap interval, and their $P$ "
        r"cell reads \texttt{---}. \emph{The two AUC columns are the same rows "
        r"scored at the two estimands}; rows where they disagree are rows whose "
        r"published reading depended on burst weighting. Rows in "
        r"\textcolor{BrickRed}{red} \textbf{beat} the shipped method at the "
        r"primary estimand; $\dagger$ marks the ones whose \emph{paired} interval "
        r"against the shipped method also excludes zero. We report them and do "
        r"not claim them: every one of these comparisons was made on the 2026 "
        r"panel. Do not compare these levels with "
        r"Table~\ref{tab:r3v4_main} (canonical environment) or "
        r"Table~\ref{tab:gradient_tta} (matched-backbone environment); "
        r"Appendix~\ref{app:environments} is the map.")

    lines = [r"\begin{table}[t]", r"\centering\footnotesize",
             r"\caption{" + caption + "}", r"\label{tab:ablation}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Configuration & AUC (bar) & AUC (post) & Brier $\downarrow$ & "
             r"$\Delta$AUC vs frozen [95\% CI] & $P(\Delta\!\le\!0)$ & retained \\",
             r"\midrule",
             (f"Frozen slow model, 5-seed average & ${frozen_bar:.4f}$ & "
              f"${ctx['test']['frozen']['auc_row']:.4f}$ & "
              f"${ctx['test']['frozen']['brier_bar']:.4f}$ & --- & --- & --- \\\\"),
             r"\midrule",
             r"\multicolumn{7}{l}{\emph{Block A --- remove or force one "
             r"ingredient, holding the rest of the grid fixed}} \\"]
    for cfg in ORDER_A:
        s_ = row(cfg)
        if s_:
            lines.append(s_)
    lines += [r"\midrule",
              r"\multicolumn{7}{l}{\emph{Block B --- cumulative build-up from a "
              r"single adapter}} \\"]
    for cfg in [c for c in abl["configuration"] if str(c).startswith("C")
                and str(c)[1:2].isdigit()]:
        s_ = row(cfg)
        if s_:
            lines.append(s_)
    fr = abl[abl["configuration"] == "R3-TTT-SF (full, 72 members)"].iloc[0]
    lines += [r"\midrule",
              (r"\textbf{R3-TTT-SF (full, 72 members)} & "
               f"$\\mathbf{{{full_bar:.4f}}}$ & "
               f"${float(fr['auc_row']):.4f}$ & ${float(fr['brier_bar']):.4f}$ & "
               f"$+{effect:.4f}$ "
               f"$[{float(fr['vs_frozen_bar_ci_lo']):+.4f},"
               f"{float(fr['vs_frozen_bar_ci_hi']):+.4f}]$ & "
               r"$<\!0.0001$ & $100\%$ \\"),
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_ablation.tex").write_text("\n".join(lines) + "\n")
    log(f"  wrote {OUT/'table_ablation.tex'}")

    # a machine-readable digest for the drafter
    digest = {
        "frozen_auc_bar": frozen_bar, "full_auc_bar": full_bar, "effect_bar": effect,
        "rows_beating_full_method": abl.loc[
            abl["beats_full_method_bar"].fillna(False).astype(bool),
            "configuration"].tolist(),
    }
    for axis in sen["axis"].unique():
        d = sen[sen["axis"] == axis]
        best = d.loc[d["bar_delta"].idxmax()]
        digest[f"axis_{axis}"] = {
            "best_value": best["value"], "best_delta": float(best["bar_delta"]),
            "n_points": int(len(d)),
            "val_best_value": str(d.loc[d["val_auc_bar"].idxmax(), "value"]),
        }
    thm = pd.read_csv(OUT / "sensitivity_threshold_matched.csv")
    digest["threshold_sf_best_balacc"] = float(th["sf_balanced_accuracy"].max())
    digest["threshold_frozen_best_balacc"] = float(th["frozen_balanced_accuracy"].max())
    digest["threshold_share_sf_ahead_absolute"] = float((th["delta_balanced_accuracy"] > 0).mean())
    digest["threshold_share_sf_ahead_matched"] = float((thm["delta_balanced_accuracy"] > 0).mean())
    jdump(digest, "digest.json")
    log("  digest: " + json.dumps({k: v for k, v in digest.items()
                                   if not k.startswith("axis_")}, default=str))

    _write_sections(args, ctx, abl, sen, th, thm, digest)
    _write_readme(args, ctx, abl, sen, th, thm, digest)


def _ax(sen, axis):
    d = sen[sen["axis"] == axis].copy()
    return d.set_index(d["value"].astype(str))


def _write_sections(args, ctx, abl, sen, th, thm, digest) -> None:
    """Drop-in LaTeX for the two missing sections, in the paper's voice."""
    fb = ctx["test"]["frozen"]["auc_bar"]
    full = ctx["test"]["sf"]["auc_bar"]
    eff = full - fb
    A = abl.set_index("configuration")

    full = ctx["test"]["sf"]["auc_bar"]

    def q(cfg, col="auc_bar"):
        return float(A.loc[cfg, col])

    def d(cfg):
        r = A.loc[cfg]
        return (float(r["vs_frozen_bar_delta"]), float(r["vs_frozen_bar_ci_lo"]),
                float(r["vs_frozen_bar_ci_hi"]))

    def ci(cfg):
        v, lo, hi = d(cfg)
        return f"${v:+.4f}$ $[{lo:+.4f},{hi:+.4f}]$"

    def dr(cfg):
        return float(A.loc[cfg, "vs_frozen_bar_delta"])

    def fr(cfg):
        return float(A.loc[cfg, "fraction_of_effect_retained"])

    def ro(cfg):
        return float(A.loc[cfg, "retrieval_only_auc_bar"])

    def vf(cfg):
        r = A.loc[cfg]
        return (f"$+{float(r['vs_full_bar_delta']):.4f}$ "
                f"$[{float(r['vs_full_bar_ci_lo']):+.4f},"
                f"{float(r['vs_full_bar_ci_hi']):+.4f}]$")

    members = pd.read_csv(OUT / "ablation_member_grid.csv")
    ens = pd.read_csv(OUT / "sensitivity_ensemble_draws.csv")
    sd1 = float(ens[ens["size"] == 1]["test_auc_bar"].std(ddof=1))
    sd48 = float(ens[ens["size"] == 48]["test_auc_bar"].std(ddof=1))
    n_above = int((members["test_auc_bar"] > full).sum())
    rho = float(members[["val_auc_bar", "test_auc_bar"]].corr(method="spearman").iloc[0, 1])
    n_beat = int(abl["beats_full_method_bar"].fillna(False).astype(bool).sum())

    beta = _ax(sen, "beta")
    T = _ax(sen, "temperature")
    lam = _ax(sen, "prior_strength")
    kk = _ax(sen, "k")
    mem = _ax(sen, "memory_fraction")
    ens = _ax(sen, "ensemble_size")
    gate = _ax(sen, "gate")
    beta_val_best = beta["val_auc_bar"].idxmax()
    beta_val_rank = int((beta["val_auc_bar"] > beta.loc["1.0", "val_auc_bar"]).sum()) + 1
    beta_word = ["", "first", "second", "third", "fourth", "fifth", "sixth",
                 "seventh", "eighth", "ninth", "tenth", "eleventh"][beta_val_rank]
    n_rows = int(len(abl))
    k_best = kk["test_auc_bar"].idxmax()
    ens_best = ens["test_auc_bar"].idxmax()
    first_mem = mem[mem["bar_ci_lo"] > 0].index.min()

    abl_tex = rf"""% ---------------------------------------------------------------------------
% Component ablation.  Generated by code/exp_ablation_sensitivity.py --
% results/ablation_sensitivity/.  Clean panel, primary per-bar estimand.
% ---------------------------------------------------------------------------
\subsection{{Component ablation}}
\label{{sec:ablation}}

\input{{tables/table_ablation}}

Table~\ref{{tab:ablation}} removes one ingredient at a time from the shipped
ensemble and then rebuilds it from a single adapter. Every row is scored at both
estimands, because several of them reverse between the two.

\paragraph{{Lead with the two rows that fail.}}
Replacing the routing coordinates with i.i.d.\ noise---the router is real, the
space is not---returns ${dr('- routing (routing coordinates replaced by i.i.d. noise)'):+.5f}$ per bar, and the
retrieval component alone falls to {ro('- routing (routing coordinates replaced by i.i.d. noise)'):.4f}.
Filling the $k$ retrieved slots with $k$ uniformly random memory bars---the
correctly specified null, in which the distances are zeroed \emph{{and}} the
candidate order is permuted---returns ${dr('- retrieval (k uniformly random memory bars; CORRECTED null)'):+.5f}$,
${100*fr('- retrieval (k uniformly random memory bars; CORRECTED null)'):.0f}\%$ of the effect, with the retrieval component at
{ro('- retrieval (k uniformly random memory bars; CORRECTED null)'):.4f}---chance. Destroy either the router or the
neighbours and nothing is left. That is the mechanism claim, and it is the only
one in this table that is unqualified.

\paragraph{{One of our own published controls is mis-specified, and we say so.}}
The random-memory control reported in earlier drafts of this work does not
destroy retrieval: because the router sorts its random $256$-item candidate pool
by true distance before the top-$k$ cut, it performs ordinary nearest-neighbour
retrieval inside an $11.4\%$ subsample of the memory. It returns
${dr('[published mis-specified random-memory control]'):+.5f}$, about half the effect, with a retrieval-only AUC of
{ro('[published mis-specified random-memory control]'):.4f}---nowhere near chance. Its number is a memory-size result, not a
retrieval-destruction result: it is what nearest-neighbour retrieval delivers
from an $11.4\%$ random subsample. The global memory-size curve
(Figure~\ref{{fig:sensitivity}}e) is the independent check and agrees within its
interval---${float(mem.loc['0.10','bar_delta']):+.4f}$ $[{float(mem.loc['0.10','bar_ci_lo']):+.4f},{float(mem.loc['0.10','bar_ci_hi']):+.4f}]$ at $10\%$, against the control's
${dr('[published mis-specified random-memory control]'):+.4f}$; the control resamples its pool per query bar rather than
once globally, which gives it more diversity. The half it retains is a
\emph{{positive}} result we had not claimed: \textbf{{retrieval still delivers
roughly half the per-bar gain from about a tenth of the memory}}.

\paragraph{{What does not earn its place.}}
The distance kernel is decorative: switching it off entirely
($T\!\to\!\infty$, uniform weights over the retrieved neighbours) costs
${full - q('- distance kernel (uniform weights, T -> inf)'):.4f}$ AUC and the arm still delivers
{100*float(A.loc['- distance kernel (uniform weights, T -> inf)','fraction_of_effect_retained']):.0f}\% of the effect; the sensitivity sweep agrees
(Figure~\ref{{fig:sensitivity}}b). The five-seed average of the frozen anchor is
likewise inert (${q('- five-seed slow average (single seed, mean of 5)') - full:+.5f}$ against a single seed, seed-to-seed
sd ${float(A.loc['- five-seed slow average (single seed, mean of 5)','auc_bar_sd']):.5f}$); we keep it for determinism, not for AUC.
Class balancing the retrieval---an obvious alternative---removes almost the whole
effect (${q('+ class-balanced retrieval'):.4f}$, {100*float(A.loc['+ class-balanced retrieval','fraction_of_effect_retained']):.0f}\% retained), which is why it is off.

\paragraph{{{n_beat} of the {n_rows} rows \emph{{beat}} the shipped method, and we
report them rather than a ladder in which every rung degrades.}}
Deleting the frozen anchor ($\beta=1$) reaches ${q('- frozen anchor (beta = 1)'):.4f}$,
{ci('- frozen anchor (beta = 1)')}, {100*fr('- frozen anchor (beta = 1)'):.0f}\% of the shipped effect, and its
\emph{{paired}} interval against the shipped method also excludes zero
({vf('- frozen anchor (beta = 1)')}); so does forcing the reliability gate off
({vf('- reliability gate (always off)')}). Deleting the prior shrinkage reaches
${q('- prior shrinkage (lambda = 0)'):.4f}$, a single neighbourhood size $k=32$ reaches
${q('- scale averaging (single k = 32)'):.4f}$, removing bar deduplication reaches
${q('- bar deduplication (event-level memory)'):.4f}$, and the member validation would have selected reaches
${q('- ensemble (single member, validation-selected)'):.4f}$. Only bar deduplication is declined on grounds of
\emph{{construction}}---a burst of posts sharing one outcome bar contributes
duplicate copies of one label to the memory. The rest are live findings, and
Section~\ref{{sec:sensitivity}} prices them. \textbf{{We adopt none of them}}, because
every one of these comparisons was made on the 2026 panel, and adopting a
configuration on the strength of that panel is precisely the selection this paper
exists to price. One of them does not survive contact with a second metric in any
case: pure retrieval without the fusion algebra scores ${q('- fusion entirely (retrieval only, sigmoid(u))'):.4f}$, but
its interval against the frozen model spans zero
({ci('- fusion entirely (retrieval only, sigmoid(u))')}) and its Brier is the worst in the table
(${q('- fusion entirely (retrieval only, sigmoid(u))', 'brier_bar'):.4f}$ against ${float(A.loc['R3-TTT-SF (full, 72 members)','brier_bar']):.4f}$). The rest do
survive it.

\paragraph{{The ensemble is a variance device, not a performance device.}}
Block~B rebuilds the method from the centroid of the pre-specified grid. A
\emph{{single}} adapter already reaches ${q('C1  single adapter (joint, k=32, T=1, lam=2, gate off, beta=0.5)'):.4f}$; every averaging axis added
thereafter moves the point estimate slightly \emph{{down}}, and the $72$-member
average lands at ${full:.4f}$. Drawing random subsets of the grid
(Figure~\ref{{fig:sensitivity}}f) makes the mechanism explicit: the mean is flat
from four members onwards while the standard deviation across subsets falls from
{sd1:.4f} at one member to {sd48:.4f} at forty-eight. The worst single member is
${q('- ensemble (single member, worst of 72)'):.4f}$---below the frozen model---and {n_above} of the $72$ beat
the average. The uniform average therefore buys a floor, not a ceiling: it is
what makes the arm reportable without a selection step, and at the per-bar
estimand it costs ${full - q('- ensemble (single member, validation-selected)'):+.4f}$ against the member validation would
have chosen, whose per-bar rank correlation with test across the grid is
$\rho={rho:+.3f}$. That is the honest state of contribution~A1 at this estimand.
"""

    sen_tex = rf"""% ---------------------------------------------------------------------------
% Sensitivity.  Generated by code/exp_ablation_sensitivity.py.
% ---------------------------------------------------------------------------
\subsection{{Sensitivity}}
\label{{sec:sensitivity}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig_sensitivity}}
\caption{{\textbf{{One-axis-at-a-time sensitivity of the selection-free ensemble,
clean panel, per-bar estimand.}} Each point pins one axis of the $72$-member grid
and averages the remaining axes, so a curve shows what an axis contributes
\emph{{to the deployed method}} rather than to an isolated adapter. Dark line and
band: $\Delta$AUC against the frozen slow model with its paired five-day
moving-block $95\%$ interval, clustered by calendar day, {args.n_boot} replicates.
Light dotted line, right axis: per-bar AUC on the validation window
(2025-10-01--12-31), whose slow model and memory come from training data only;
nothing in the paper is selected on it. Gold bands mark the values in the
pre-specified grid. Grey dashed line: the shipped method's effect. Panel~(h)
reports the decision threshold two ways---at a fixed absolute cut-off and at a
quantile-matched cut-off that makes both arms flag the same number of bars.}}
\label{{fig:sensitivity}}
\end{{figure}}

Of the eight axes we swept, three are flat, four are not, and one---the decision
threshold---reverses the conclusion. We report all of them.

\paragraph{{Flat.}}
The kernel temperature is a genuine plateau: $\Delta$ moves from
${float(T.loc['0.05','bar_delta']):+.4f}$ at $T=0.05$ to ${float(T.loc['inf','bar_delta']):+.4f}$ at $T\to\infty$, a range of
${float(T.loc['0.05','bar_delta']) - float(T.loc['inf','bar_delta']):.4f}$ AUC across a $200\times$ finite sweep \emph{{plus}} the limit in
which the kernel is deleted outright. The ensemble size saturates immediately:
the mean over random subsets is ${float(ens.loc['4','test_auc_bar']):.4f}$ at four members against
${float(ens.loc['72','test_auc_bar']):.4f}$ at $72$, and what changes with size is not the mean but the
spread, whose standard deviation falls from {sd1:.4f} at one member to
{sd48:.4f} at forty-eight. The routing space is flat over three of its four
settings (${float(_ax(sen,'routing_space').loc['timing','bar_delta']):+.4f}$, ${float(_ax(sen,'routing_space').loc['joint','bar_delta']):+.4f}$, ${float(_ax(sen,'routing_space').loc['arrival_only','bar_delta']):+.4f}$); only
\texttt{{market\_state}} alone is materially worse
(${float(_ax(sen,'routing_space').loc['market_state','bar_delta']):+.4f}$, interval spanning zero). Contrary to an earlier draft
written at the per-post estimand, the best routing space \emph{{is}} in the
pre-specified grid.

\paragraph{{Not flat, and the shipped value is not the optimum on any of them.}}
The blend $\beta$ is \emph{{strictly monotone}} on test from
${float(beta.loc['0.0','test_auc_bar']):.4f}$ at $\beta=0$---which recovers the frozen model exactly---to
${float(beta.loc['1.0','test_auc_bar']):.4f}$ at $\beta=1$; the prior strength $\lambda$ is monotone
\emph{{down}}, so the unshrunk estimator ($\lambda=0$, ${float(lam.loc['0.0','bar_delta']):+.4f}$) beats every
value in the shipped grid $\{{2,8\}}$; the neighbourhood size has an interior
optimum at $k={k_best}$ and decays to ${float(kk.loc['full','bar_delta']):+.4f}$ when the whole memory is
retrieved; and the memory-size curve rises with
memory (non-monotonically at the small end, where the draw-to-draw spread
dominates), its interval first excluding zero at {100*float(first_mem):.0f}\% of the memory. The shipped configuration is not the optimum on any
of the four: it is ${float(beta.loc['1.0','test_auc_bar']) - float(beta.loc['0.5','test_auc_bar']):.4f}$ below the best blend, ${float(lam.loc['0.0','test_auc_bar']) - float(lam.loc['2.0','test_auc_bar']):.4f}$ below the
best prior strength (${float(lam.loc['0.0','test_auc_bar']) - float(lam.loc['8.0','test_auc_bar']):.4f}$ below at the grid's upper end), and
${float(kk.loc[k_best,'test_auc_bar']) - full:.4f}$ below the best single neighbourhood size.
\textbf{{We say so rather than move.}}

\paragraph{{Why we do not move.}}
Three reasons, and the first is the only one that survives every check.
(i)~Changing the shipped configuration on the strength of the 2026 panel is
selection on an exhausted confirmatory set---the failure this paper prices, at
$1.88\times$ the adaptation effect for grid support alone
(Table~\ref{{tab:price_list}}, C1). (ii)~$\beta=0$ recovers the frozen model
exactly, so the blend is the only parameterisation with a safe limit at both
ends. (iii)~On never-selected-on pre-2026 folds $\beta=1$ is worth a further
$+0.0170$ $[+0.0114,+0.0228]$, so the direction is corroborated off-panel; the
question is deferred to a one-shot pre-registered forward test that has not been
run. We must also withdraw an argument made in earlier drafts: at the per-bar
estimand \emph{{validation does not rank $\beta=1$ last}}. The validation curve
peaks at $\beta={beta_val_best}$ and ranks $\beta=1$ {beta_word} of eleven, ahead of the shipped $\beta=0.5$. The
validation--test disagreement on this axis is real, but it is weaker than the
per-post metric made it look, and it is not the argument we thought it was.

\paragraph{{The gate is the most expensive commitment inside the grid.}}
Forcing the reliability gate off gives ${float(gate.loc['False','bar_delta']):+.4f}$ against ${float(gate.loc['True','bar_delta']):+.4f}$ with it
on---a spread of ${float(gate.loc['False','bar_delta']) - float(gate.loc['True','bar_delta']):.4f}$, {100*(float(gate.loc['False','bar_delta']) - float(gate.loc['True','bar_delta']))/eff:.0f}\% of the adaptation effect,
inside a grid we average over rather than choose from.

\paragraph{{The AUC gain does not survive a fixed decision threshold.}}
AUC is threshold-free and the arm's Brier is better
(${float(A.loc['R3-TTT-SF (full, 72 members)','brier_bar']):.4f}$ against ${ctx['test']['frozen']['brier_bar']:.4f}$), but the fused
probability is compressed towards the base rate, so at a fixed absolute cut-off
the shipped arm has \emph{{higher}} balanced accuracy than the frozen model at only
{100*digest['threshold_share_sf_ahead_absolute']:.0f}\% of the thresholds we swept, in a narrow band around the base
rate. Cutting each arm at its own quantile, so that both flag the same number of
bars, the shipped arm is ahead at {100*digest['threshold_share_sf_ahead_matched']:.0f}\% of operating points. The ranking
improvement is real; converting it into a fixed-threshold decision rule requires
a recalibration step this paper does not supply, and we do not claim one.
"""
    (OUT / "section_ablation.tex").write_text(abl_tex)
    (OUT / "section_sensitivity.tex").write_text(sen_tex)
    log(f"  wrote {OUT/'section_ablation.tex'} and {OUT/'section_sensitivity.tex'}")


def _md(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Markdown table without the tabulate dependency."""
    def cell(v):
        if isinstance(v, float) and np.isfinite(v):
            return floatfmt.format(v)
        return "" if (isinstance(v, float) and not np.isfinite(v)) else str(v)
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    body = ["| " + " | ".join(cell(v) for v in r) + " |"
            for r in frame.itertuples(index=False, name=None)]
    return "\n".join([head, rule] + body)


def _write_readme(args, ctx, abl, sen, th, thm, digest) -> None:
    fb = ctx["test"]["frozen"]["auc_bar"]
    full = ctx["test"]["sf"]["auc_bar"]
    eff = full - fb
    cols = ["configuration", "block", "auc_bar", "auc_row", "brier_bar",
            "vs_frozen_bar_delta", "vs_frozen_bar_ci_lo", "vs_frozen_bar_ci_hi",
            "vs_frozen_bar_p_le_zero", "fraction_of_effect_retained",
            "beats_full_method_bar"]
    tbl = _md(abl[[c for c in cols if c in abl.columns]])
    sen_tbl = _md(sen[["axis", "value", "n_members", "test_auc_bar", "val_auc_bar",
                       "bar_delta", "bar_ci_lo", "bar_ci_hi", "bar_p_le_zero"]])
    beats = "\n".join(f"- `{c}`" for c in digest["rows_beating_full_method"])
    text = f"""# Ablation and sensitivity — clean panel, primary per-bar estimand

Generated by `code/exp_ablation_sensitivity.py` (deterministic; `--help` for the
stages).  Environment: **CLEAN**, frozen `{fb:.4f}` / R3-TTT-SF `{full:.4f}`,
effect `{eff:+.4f}` over {ctx['test']['n_bars']} outcome bars
({ctx['test']['n_posts']} posts, {ctx['test']['n_days']} test days).
Never difference these levels against the canonical (0.8273/0.8362), regenerated
common-protocol (0.8324/0.8422) or matched-backbone environments.

Validation stream (reported only to show validation and test disagree; nothing is
selected on it): {ctx['validation']['n_bars']} bars, frozen
`{ctx['validation']['frozen']['auc_bar']:.4f}` / SF `{ctx['validation']['sf']['auc_bar']:.4f}`.

## Headline of this track

**{len(digest['rows_beating_full_method'])} of the ablation rows beat the full
method at the primary estimand.** An ablation ladder in which every rung degrades
would be false here, so the table marks them and the text explains each.

{beats}

## What the per-bar switch invalidated in `refnotes/V_ablation_sensitivity_design.md`

V was written in the regenerated common-protocol environment at the per-post
estimand.  Structure retained; the following substantive claims do **not**
survive and must not be copied out of V:

1. **"Validation ranks $\\beta=1$ last."** False here. The per-bar validation
   curve peaks at $\\beta=0.7$ and ranks $\\beta=1$ fifth of eleven.
2. **"The best routing space (`arrival_only`) is not in the grid."** False here:
   `timing` (in the grid) is the best space at the per-bar estimand and
   `arrival_only` is third of four.
3. **"A random Gaussian projection is a placebo that lands on the frozen
   model."** Retired; on the clean panel it is a near-isometry.  The corrected
   random-memory null is used as the floor instead.
4. **All levels.** Every number in V's Part IV tables is CP; none may be quoted.

## Ablation table

{tbl}

## Sensitivity curves

{sen_tbl}

## Files

| file | what |
|---|---|
| `protocol.json` | frozen protocol record for this track |
| `context.json` | panel census, both streams, frozen and SF references |
| `ablation_table.csv` | the ablation table, both estimands, paired intervals |
| `ablation_member_grid.csv` | all 72 members with test and validation per-bar AUC |
| `ablation_random_memory_null_draws.csv` | corrected null, {args.null_seeds} replicates |
| `ablation_random_memory_published_draws.csv` | the mis-specified published control |
| `ablation_slow_seed_draws.csv` | five single-seed frozen anchors |
| `sensitivity_curves.csv` | every axis point, both streams, paired intervals |
| `sensitivity_memory_draws.csv` | raw memory-subsample draws |
| `sensitivity_ensemble_draws.csv` | raw random-member-subset draws |
| `sensitivity_threshold.csv` | absolute decision-threshold sweep |
| `sensitivity_threshold_matched.csv` | quantile-matched decision-threshold sweep |
| `fig_ablation.pdf`, `fig_sensitivity.pdf` | the two figures |
| `table_ablation.tex`, `section_ablation.tex`, `section_sensitivity.tex` | drop-in LaTeX |
| `digest.json` | machine-readable summary for the drafter |
"""
    (OUT / "REPORT.md").write_text(text)
    log(f"  wrote {OUT/'REPORT.md'}")


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+",
                    choices=["context", "ablation", "sensitivity", "figs",
                             "report", "all"],
                    help="stages to run, in order")
    ap.add_argument("--jobs", type=int, default=64, help="joblib workers")
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help="paired block-bootstrap replicates (default 5000)")
    ap.add_argument("--null-seeds", type=int, default=N_NULL_SEEDS,
                    help="replicates of the corrected random-memory null")
    ap.add_argument("--mem-draws", type=int, default=N_MEM_DRAWS,
                    help="memory subsample draws per fraction")
    ap.add_argument("--subset-draws", type=int, default=N_SUBSET_DRAWS,
                    help="random member subsets per ensemble size")
    args = ap.parse_args()

    stages = ["context", "ablation", "sensitivity", "figs", "report"] \
        if "all" in args.stages else args.stages
    t0 = time.time()
    for s in stages:
        globals()[f"stage_{s}"](args)
    log(f"done in {time.time() - t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
