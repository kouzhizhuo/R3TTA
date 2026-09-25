#!/usr/bin/env python
"""R3-TTT: the strongest MISSING baselines, gradient-free first.

Five families, in the priority order the baseline survey set (not the brief's):

  1. k-NN-LM interpolation on the routing space   (Khandelwal et al., ICLR 2020)
     -- plus plain k-NN vote as the floor, Nadaraya-Watson (1964) as the
     bandwidth-free strong form, and a nearest-centroid prototype.
  2. AdaNPC                                        (Zhang et al., ICML 2023)
     -- (a) faithful: the k-NN vote REPLACES the classifier;
        (b) growth: AdaNPC's pseudo-label memory growth applied to OUR memory.
        (b) is F_b-measurable and therefore LEGAL under our own filtration.
  3. T3A                                           (Iwasawa & Matsuo, NeurIPS 2021)
     -- on the routing space (the pseudo-label twin of R3-TTT) and on both
        neural backbones.
  4. LAME                                          (Boudiaf et al., CVPR 2022)
     -- faithful (the test batch is a bar) and a sliding-window variant.
  5. NOTE                                          (Gong et al., NeurIPS 2022)
     -- IABN alone (zero gradient steps) and IABN+PBRS (the full method).

Plus two the survey ranked #2 overall and which nothing in the paper answers:
  * Hedge / exponentially-weighted aggregation over the published 72-member
    grid with 1-hour-delayed matured feedback -- the regret-optimal comparator
    to contribution A1's uniform average.
  * alpha-BN and ETA (EATA without the Fisher regulariser) in the neural stage.

Environments are NEVER mixed.
  * ``retrieval`` / ``hedge``: CLEAN panel, real GBT slow model.
        frozen 0.7679 per bar / 0.8138 per post; R3-TTT-SF 0.7800 / 0.8282.
  * ``neural``   : CLEAN panel, matched neural backbones, re-fitted here.
        This is a NEW environment.  Its levels may not be compared with the
        contaminated-panel table ``tab:gradient_tta``.

Primary estimand: per bar (525 outcome bars).  Row level is a secondary and is
always printed beside it.  Every hyperparameter of every baseline is selected on
the VALIDATION split (2025-10-01..2026-01-01) at the validation per-bar AUC.
Nothing is selected on the 2026 test panel.  R3-TTT is not tuned here at all.

Usage
-----
    python exp_new_baselines.py --stage all
    python exp_new_baselines.py --stage retrieval --n-boot 5000
    python exp_new_baselines.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# BLAS threads pinned BEFORE numpy is imported.  Documented 7.7e-5 slow-model
# nondeterminism otherwise.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common_protocol_clean as cpc          # noqa: E402  NEVER common_protocol

SEED = 42
N_BOOT_DEFAULT = 5000
BLOCK_DAYS = 5
T0 = time.time()
_OUT: Path | None = None


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def out() -> Path:
    assert _OUT is not None
    return _OUT


def save(frame: pd.DataFrame, name: str) -> Path:
    p = out() / name
    frame.to_csv(p, index=False)
    log(f"  wrote {p.name}  ({len(frame)} rows)")
    return p


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    return v


def jdump(obj, name: str) -> Path:
    p = out() / name
    p.write_text(json.dumps(obj, indent=2, default=_jsonable))
    log(f"  wrote {p.name}")
    return p


# =============================================================================
# metrics + the bar index (identical definitions to exp_prebar_spine.py)
# =============================================================================
def auc(y, s) -> float:
    y = np.asarray(y, dtype=int)
    if np.unique(y).size != 2:
        return float("nan")
    return float(cpc.fast_binary_auc(y, np.asarray(s, dtype=float)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def logloss(y, p, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class BarIndex:
    """The decision-unit index: 525 bars on test, 407 on validation."""

    def __init__(self, stream: pd.DataFrame):
        self.stream = stream
        self.times = stream["datetime"].to_numpy()
        self.groups = cpc.bar_groups(stream)
        self.n_rows = len(stream)
        self.n_bars = len(self.groups)
        self.row_bar = np.empty(self.n_rows, dtype=int)
        for pos, locs in enumerate(self.groups):
            self.row_bar[locs] = pos
        self.sizes = np.array([len(g) for g in self.groups], dtype=float)
        self.y_row = stream["target_hi_vol"].to_numpy(dtype=int)
        for pos, g in enumerate(self.groups):
            if np.unique(self.y_row[g]).size != 1:
                raise AssertionError(f"bar {pos} carries more than one label")
        self.y_bar = np.array([self.y_row[g][0] for g in self.groups], dtype=int)
        self.bar_times = np.array([self.times[g[0]] for g in self.groups])

    def pool(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.bincount(self.row_bar, weights=p, minlength=self.n_bars) / self.sizes

    def spread(self, bar_values) -> np.ndarray:
        return np.asarray(bar_values, dtype=float)[self.row_bar]

    def auc_bar(self, p) -> float:
        return auc(self.y_bar, self.pool(p))

    def auc_row(self, p) -> float:
        return auc(self.y_row, p)

    def scores(self, p) -> dict:
        pb = self.pool(p)
        return {
            "auc_bar": auc(self.y_bar, pb),
            "brier_bar": brier(self.y_bar, pb),
            "logloss_bar": logloss(self.y_bar, pb),
            "auc_row": auc(self.y_row, p),
            "brier_row": brier(self.y_row, p),
            "logloss_row": logloss(self.y_row, p),
        }


# =============================================================================
# paired 5-day moving-block bootstrap, clustered by CALENDAR DAY
# =============================================================================
def _block_rows(days, n_boot, block_days, seed):
    d = pd.to_datetime(pd.Series(np.asarray(days))).dt.normalize().to_numpy()
    uniq = np.array(sorted(set(d)))
    index = {u: np.flatnonzero(d == u) for u in uniq}
    n_blocks = max(1, int(np.ceil(len(uniq) / block_days)))
    high = max(1, len(uniq) - block_days + 1)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        starts = rng.integers(0, high, size=n_blocks)
        chosen = [u for s in starts for u in uniq[s:s + block_days]]
        yield np.concatenate([index[u] for u in chosen])


def paired_boot(y, a, b, days, *, metric="auc", n_boot=N_BOOT_DEFAULT,
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
    for rows in _block_rows(days, n_boot, block_days, seed):
        va, vb = fn(y[rows], a[rows]), fn(y[rows], b[rows])
        if np.isfinite(va) and np.isfinite(vb):
            draws.append(sign * (va - vb))
    draws = np.asarray(draws, dtype=float)
    return {
        "metric": metric,
        "point_a": float(fn(y, a)), "point_b": float(fn(y, b)),
        "delta": float(point),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "p_delta_le_zero": float(np.mean(draws <= 0.0)),
        "n_boot": int(len(draws)), "block_days": int(block_days),
        "clustered_by": "calendar day", "seed": int(seed),
    }


def compare(bi: BarIndex, pa, pb, *, n_boot: int, prefix: str) -> dict:
    """Delta of arm a over arm b at both estimands, AUC + Brier + log loss."""
    ba, bb = bi.pool(pa), bi.pool(pb)
    rec = {}
    for tag, (y, x, z, days) in {
        "bar": (bi.y_bar, ba, bb, bi.bar_times),
        "row": (bi.y_row, np.asarray(pa), np.asarray(pb), bi.times),
    }.items():
        for metric in ("auc", "brier", "logloss"):
            r = paired_boot(y, x, z, days, metric=metric, n_boot=n_boot)
            rec[f"{prefix}{tag}_{metric}_delta"] = r["delta"]
            rec[f"{prefix}{tag}_{metric}_ci_lo"] = r["ci_lo"]
            rec[f"{prefix}{tag}_{metric}_ci_hi"] = r["ci_hi"]
            rec[f"{prefix}{tag}_{metric}_p_le_zero"] = r["p_delta_le_zero"]
    return rec


# =============================================================================
# HAC Diebold-Mariano on Brier and log loss (Newey-West, hand-rolled:
# statsmodels is absent on this machine and we do not fight pip)
# =============================================================================
def hac_dm(loss_a, loss_b, *, lag: int | None = None) -> dict:
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if lag is None:
        lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    dbar = float(d.mean())
    e = d - dbar
    g0 = float(np.dot(e, e) / n)
    s = g0
    for h in range(1, lag + 1):
        gh = float(np.dot(e[h:], e[:-h]) / n)
        s += 2.0 * (1.0 - h / (lag + 1.0)) * gh
    s = max(s, 1e-24)
    stat = dbar / np.sqrt(s / n)
    # two-sided normal p
    from math import erfc, sqrt
    p = float(erfc(abs(stat) / sqrt(2.0)))
    return {"dm_stat": float(stat), "dm_p_two_sided": p, "nw_lag": int(lag),
            "mean_loss_diff": dbar, "n": int(n)}


# =============================================================================
# routing: an exact re-implementation of common_protocol_clean._route with the
# StandardScaler injected, so that a GROWING memory does not silently re-fit the
# normalisation statistics (the price list's single largest axis, +0.1077).
# Verified byte-identical against cpc._route in stage `retrieval`.
# =============================================================================
from sklearn.preprocessing import StandardScaler          # noqa: E402


def fit_space(memory_frame: pd.DataFrame, features):
    raw = memory_frame[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler().fit(raw)
    return scaler, scaler.transform(raw)


def transform_rows(scaler, frame: pd.DataFrame, features) -> np.ndarray:
    return scaler.transform(
        frame[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0))


def bar_queries(stream_x: np.ndarray, groups) -> np.ndarray:
    return np.stack([stream_x[locs].mean(axis=0) for locs in groups], axis=0)


def distance_matrix(memory_x: np.ndarray, queries: np.ndarray, n_feat: int):
    """The (n_query, n_memory) squared distance matrix cpc._route builds.

    The whole query block is multiplied at once, exactly as cpc._route does.
    This matters: in the 5-d ``timing`` space 1,973 of 2,249 memory bars are
    exact duplicates, and a gemv-vs-gemm rounding difference of 9e-16 flips 22%
    of the retrieved slots.  Any streaming variant must reuse this block rather
    than recompute a single query's row.
    """
    d = (
        np.einsum("ij,ij->i", queries, queries)[:, None]
        - 2.0 * (queries @ memory_x.T)
        + np.einsum("ij,ij->i", memory_x, memory_x)[None, :]
    ) / float(n_feat)
    np.maximum(d, 0.0, out=d)
    return d


def sort_topk(d: np.ndarray, keep: int):
    keep = min(keep, d.shape[1])
    local = np.argpartition(d, keep - 1, axis=1)[:, :keep]
    local_d = np.take_along_axis(d, local, axis=1)
    order = np.argsort(local_d, axis=1, kind="stable")
    return (np.take_along_axis(local, order, axis=1),
            np.take_along_axis(local_d, order, axis=1))


def route_matrix(memory_x: np.ndarray, queries: np.ndarray, n_feat: int, keep: int):
    """(idx, dist) exactly as cpc._route computes them, scaler injected."""
    return sort_topk(distance_matrix(memory_x, queries, n_feat), keep)


# =============================================================================
# environments
# =============================================================================
class Env:
    """One (memory, stream, frozen slow probability) environment.

    ``validation``: slow model and memory from TRAIN only, scored on the
    validation split.  This is where every baseline hyperparameter is chosen.
    ``test``: slow model and memory from TRAIN+VALIDATION, scored once on test.
    """

    def __init__(self, name, memory_source, stream, deploy_start):
        self.name = name
        self.memory_source = memory_source.reset_index(drop=True)
        self._mem_pseudo = None
        self.stream = stream.reset_index(drop=True)
        self.bi = BarIndex(self.stream)
        self.memory = cpc.build_bar_memory(memory_source, deploy_start=deploy_start)
        self.frozen = cpc.slow_probability(memory_source, self.stream)
        self.spaces = {}
        for sp, feats in cpc.RETRIEVAL_SPACES.items():
            scaler, mem_x = fit_space(self.memory.frame, feats)
            stream_x = transform_rows(scaler, self.stream, feats)
            q = bar_queries(stream_x, self.bi.groups)
            dfull = distance_matrix(mem_x, q, len(feats))
            # ``idx``/``dist`` are the top-MAX_K neighbours retrieved EXACTLY as
            # cpc._route retrieves them (argpartition then stable sort), so every
            # retrieval baseline sees the same neighbours R3-TTT sees, including
            # the load-bearing tie-break.  ``idx_all`` is the full ordering,
            # which only Nadaraya-Watson (bandwidth over the whole memory) uses.
            idx, dist = sort_topk(dfull, cpc.MAX_K)
            idx_all, dist_all = sort_topk(dfull, len(mem_x))
            self.spaces[sp] = dict(features=list(feats), scaler=scaler,
                                   mem_x=mem_x, queries=q, idx=idx, dist=dist,
                                   idx_all=idx_all, dist_all=dist_all,
                                   dist_full=dfull)
        self.sf, _, _, self.n_members = cpc.r3ttt_selection_free(
            self.memory, self.stream, self.frozen)
        # frozen slow model applied to the memory bars themselves -- the
        # pseudo-label source for the T3A port.  Cached: refitting five GBTs per
        # call would dominate the run.
        self._mem_pseudo = cpc.slow_probability(self.memory_source,
                                                self.memory.frame)
        log(f"env {name}: {len(self.memory)} memory bars, {self.bi.n_rows} rows / "
            f"{self.bi.n_bars} bars | frozen bar {self.bi.auc_bar(self.frozen):.4f} "
            f"row {self.bi.auc_row(self.frozen):.4f} | SF bar "
            f"{self.bi.auc_bar(self.sf):.4f} row {self.bi.auc_row(self.sf):.4f}")


def build_envs():
    panel = cpc.load_event_panel()
    cpc.assert_clean(panel, where="clean event panel")
    train = panel[panel.datetime < cpc.TRAIN_END].reset_index(drop=True)
    val = panel[(panel.datetime >= cpc.TRAIN_END)
                & (panel.datetime < cpc.TEST_START)].reset_index(drop=True)
    pretest = panel[panel.datetime < cpc.TEST_START].reset_index(drop=True)
    test = panel[panel.datetime >= cpc.TEST_START].reset_index(drop=True)
    ev_val = Env("validation", train, val, cpc.TRAIN_END)
    ev_test = Env("test", pretest, test, cpc.TEST_START)
    return panel, ev_val, ev_test


# =============================================================================
# the gradient-free retrieval family
# =============================================================================
KNN_K = (1, 2, 4, 8, 16, 32, 64, 128, 256)
KNN_T = ("unif", 1.0, 0.25)                       # kernel temperature; unif = flat
LAMBDAS = tuple(np.round(np.arange(0.0, 1.0001, 0.05), 3))
NW_H = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
PROTO_TAU = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
T3A_M = (1, 5, 20, 50, 100, -1)                   # -1 = unbounded support
ADANPC_TAU = (0.25, 1.0, 4.0)
ADANPC_MARGIN = (2.0, 0.9, 0.7, 0.55)             # 2.0 = growth disabled
GROWTH_MARGIN = (2.0, 0.9, 0.8, 0.7, 0.6, 0.55)   # 2.0 = growth disabled


def knn_bar_scores(env: Env, space: str, k: int, temp) -> np.ndarray:
    """Distance-weighted neighbour-label mean, per bar.  No prior shrinkage."""
    s = env.spaces[space]
    idx = s["idx"][:, :k]
    dist = s["dist"][:, :k]
    lab = env.memory.labels[idx]
    if temp == "unif":
        w = np.ones_like(dist)
    else:
        gap = dist - dist[:, :1]
        w = np.exp(-np.clip(gap / float(temp), 0.0, 700.0))
    mass = w.sum(axis=1)
    return (w * lab).sum(axis=1) / np.maximum(mass, 1e-300)


def nw_bar_scores(env: Env, space: str, h: float) -> np.ndarray:
    """Nadaraya-Watson over the FULL memory: no k, one bandwidth."""
    s = env.spaces[space]
    dist = s["dist_all"]
    lab = env.memory.labels[s["idx_all"]]
    gap = dist - dist[:, :1]
    w = np.exp(-np.clip(gap / float(h), 0.0, 700.0))
    mass = w.sum(axis=1)
    return (w * lab).sum(axis=1) / np.maximum(mass, 1e-300), float(
        np.mean(mass ** 2 / np.maximum((w ** 2).sum(axis=1), 1e-300)))


def prototype_bar_scores(env: Env, space: str, tau: float) -> np.ndarray:
    """Two class centroids in the routing space (Snell et al. reduced to C=2)."""
    s = env.spaces[space]
    y = env.memory.labels
    c1 = s["mem_x"][y == 1].mean(axis=0)
    c0 = s["mem_x"][y == 0].mean(axis=0)
    q = s["queries"]
    d1 = ((q - c1) ** 2).sum(axis=1) / q.shape[1]
    d0 = ((q - c0) ** 2).sum(axis=1) / q.shape[1]
    return cpc.sigmoid((d0 - d1) / float(tau))


def interpolate(env: Env, bar_score: np.ndarray, lam: float, *, space_kind="prob"):
    """p_row = (1-lam) p_slow_row + lam p_retrieval_bar  (k-NN-LM's rule)."""
    r = env.bi.spread(bar_score)
    if space_kind == "prob":
        return (1.0 - lam) * env.frozen + lam * r
    rc = np.clip(r, 1e-6, 1 - 1e-6)
    return cpc.sigmoid((1.0 - lam) * cpc.logit(env.frozen) + lam * cpc.logit(rc))


def t3a_bar_scores(env: Env, space: str, m: int) -> np.ndarray:
    """T3A ported to the routing space: PSEUDO-labelled support prototypes.

    Support sets are initialised from the memory bars carrying the frozen slow
    model's own predicted labels -- the analogue of T3A initialising from the
    linear classifier's weight vectors, and the exact pseudo-label twin of
    R3-TTT, which uses the same bars with their REAL matured labels.  Each test
    bar is pseudo-labelled by the frozen model, appended to its support set, and
    the M lowest-entropy members per class are retained.  Zero gradients.
    """
    s = env.spaces[space]
    mem_p = env._mem_pseudo
    mem_x = s["mem_x"]
    mem_lab = (mem_p >= 0.5).astype(int)
    ent = -(mem_p * np.log(np.clip(mem_p, 1e-12, 1))
            + (1 - mem_p) * np.log(np.clip(1 - mem_p, 1e-12, 1)))
    sup = {c: [(float(ent[i]), mem_x[i]) for i in np.flatnonzero(mem_lab == c)]
           for c in (0, 1)}
    q = s["queries"]
    pbar = env.bi.pool(env.frozen)
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
            if sup[cc]:
                cent[cc] = np.mean([v for _, v in sup[cc]], axis=0)
            else:
                cent[cc] = np.zeros(q.shape[1])
        z = q[t]
        nz = np.linalg.norm(z) + 1e-12
        s1 = float(z @ cent[1]) / (nz * (np.linalg.norm(cent[1]) + 1e-12))
        s0 = float(z @ cent[0]) / (nz * (np.linalg.norm(cent[0]) + 1e-12))
        scores[t] = cpc.sigmoid((s1 - s0) * 4.0)
    return scores


def adanpc_bar_scores(env: Env, space: str, k: int, tau: float, margin: float):
    """AdaNPC: cosine k-NN vote over a real-labelled memory that GROWS with
    confidently pseudo-labelled test points.  The vote REPLACES the classifier.

    Growth uses only PREDICTED labels, so it is F_b-measurable and legal under
    our filtration.  ``margin >= 1`` disables growth (the static variant).
    """
    s = env.spaces[space]
    mem_x = s["mem_x"]
    mem_y = env.memory.labels.copy()
    mem_n = mem_x / (np.linalg.norm(mem_x, axis=1, keepdims=True) + 1e-12)
    q = s["queries"]
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    grown_x = [mem_n]
    grown_y = [mem_y]
    scores = np.empty(len(q), dtype=float)
    n_added = 0
    X = mem_n
    Y = mem_y
    for t in range(len(q)):
        sim = X @ qn[t]
        kk = min(k, len(sim))
        top = np.argpartition(-sim, kk - 1)[:kk]
        top = top[np.argsort(-sim[top], kind="stable")]
        w = sim[top] / float(tau)
        w = np.exp(w - w.max())
        w /= w.sum()
        p = float((w * Y[top]).sum())
        scores[t] = p
        if margin < 1.0 and max(p, 1.0 - p) >= margin:
            X = np.vstack([X, qn[t][None, :]])
            Y = np.append(Y, float(p >= 0.5))
            n_added += 1
    del grown_x, grown_y
    return scores, n_added


def r3ttt_growth(env: Env, margin: float, *, verify_against=None):
    """R3-TTT-SF with AdaNPC-style pseudo-label memory GROWTH.

    Prices our immutability commitment.  The routing scalers are frozen at
    deployment (a grown memory must not move the normalisation statistics -- the
    price list's largest axis).  ``margin >= 1`` disables growth and MUST
    reproduce ``cpc.r3ttt_selection_free`` exactly; that identity is asserted.
    """
    bi = env.bi
    n_rows = bi.n_rows
    spaces = {}
    for sp, s in env.spaces.items():
        spaces[sp] = dict(n_feat=len(s["features"]), queries=s["queries"],
                          dist_full=s["dist_full"], extra=[])
    labels = env.memory.labels.copy()
    p_row = np.zeros(n_rows, dtype=float)
    n_added = 0
    for t, locs in enumerate(bi.groups):
        acc = np.zeros(len(locs), dtype=float)
        count = 0
        for sp, s in spaces.items():
            # the pre-deployment block is the byte-identical row of the batched
            # distance matrix; only appended pseudo-labelled bars are new
            row = s["dist_full"][t]
            if s["extra"]:
                e = np.stack(s["extra"], axis=0)
                q = s["queries"][t]
                de = ((e - q[None, :]) ** 2).sum(axis=1) / float(s["n_feat"])
                row = np.concatenate([row, np.maximum(de, 0.0)])
            idx, dist = sort_topk(row[None, :], cpc.MAX_K)
            states = cpc._fast_states(                      # noqa: SLF001
                idx, dist, labels,
                balanced=cpc.BALANCED_DEFAULT, scales=cpc.SCALES,
                temperatures=cpc.TEMPERATURES,
                prior_strengths=cpc.PRIOR_STRENGTHS, gates=cpc.GATES)
            for (_, _, _), (bias_bar, trust_bar) in states.items():
                for beta in cpc.BLENDS:
                    acc += cpc.fuse(env.frozen[locs],
                                    np.repeat(bias_bar, len(locs)),
                                    np.repeat(trust_bar, len(locs)), beta)
                    count += 1
        p = acc / count
        p_row[locs] = p
        if margin < 1.0:
            conf = float(np.mean(p))
            if max(conf, 1.0 - conf) >= margin:
                for sp, s in spaces.items():
                    s["extra"].append(s["queries"][t].copy())
                labels = np.append(labels, float(conf >= 0.5))
                n_added += 1
    if verify_against is not None:
        d = float(np.max(np.abs(p_row - verify_against)))
        log(f"  growth-loop identity check (margin={margin}): max|diff| = {d:.3e}")
        if d > 1e-9:
            raise AssertionError(
                f"streaming growth loop does not reproduce r3ttt_selection_free "
                f"at margin={margin}: max abs diff {d:.3e}")
    return p_row, n_added


# =============================================================================
# stage: retrieval
# =============================================================================
def stage_retrieval(args, ev_val: Env, ev_test: Env) -> pd.DataFrame:
    log("stage retrieval -- gradient-free retrieval family, CLEAN GBT environment")
    sweeps: list[dict] = []
    arms: dict[str, dict] = {}

    def sweep_lambda(arm, base_key, bar_val, bar_test, extra, kinds=("prob",)):
        """Choose lambda (and the arm's own knobs) on VALIDATION per-bar AUC."""
        for kind in kinds:
            for lam in LAMBDAS:
                pv = interpolate(ev_val, bar_val, lam, space_kind=kind)
                sweeps.append(dict(arm=arm, **extra, lam=float(lam), logit_space=(kind == "logit"),
                                   val_auc_bar=ev_val.bi.auc_bar(pv),
                                   val_auc_row=ev_val.bi.auc_row(pv)))
                key = (arm, kind)
                cand = sweeps[-1]["val_auc_bar"]
                if key not in arms or cand > arms[key]["val_auc_bar"]:
                    arms[key] = dict(arm=arm, logit_space=(kind == "logit"),
                                     val_auc_bar=cand,
                                     val_auc_row=sweeps[-1]["val_auc_row"],
                                     hp={**extra, "lam": float(lam)},
                                     bar_test=bar_test, kind=kind)
        del base_key

    # ---- arm A: plain k-NN vote (the floor: retrieval alone, no slow model) --
    best_a = None
    for sp in cpc.RETRIEVAL_SPACES:
        for k in KNN_K:
            bv = knn_bar_scores(ev_val, sp, k, "unif")
            a = ev_val.bi.auc_bar(ev_val.bi.spread(bv))
            sweeps.append(dict(arm="knn_vote", space=sp, k=k, temp="unif", lam=1.0,
                               logit_space=False, val_auc_bar=a,
                               val_auc_row=ev_val.bi.auc_row(ev_val.bi.spread(bv))))
            if best_a is None or a > best_a[0]:
                best_a = (a, sp, k)
    _, sp_a, k_a = best_a
    arms[("knn_vote", "prob")] = dict(
        arm="knn_vote", logit_space=False, val_auc_bar=best_a[0], val_auc_row=np.nan,
        hp=dict(space=sp_a, k=k_a, temp="unif", lam=1.0),
        bar_test=knn_bar_scores(ev_test, sp_a, k_a, "unif"), kind="retrieval_only")
    log(f"  knn_vote selected space={sp_a} k={k_a} (val bar {best_a[0]:.4f})")

    # ---- arm B: k-NN-LM interpolation --------------------------------------
    log("  k-NN-LM: 3 spaces x 9 k x 3 kernels x 21 lambda x 2 spaces = 3,402 configs")
    for sp in cpc.RETRIEVAL_SPACES:
        for k in KNN_K:
            for tp in KNN_T:
                bv = knn_bar_scores(ev_val, sp, k, tp)
                bt = knn_bar_scores(ev_test, sp, k, tp)
                sweep_lambda("knnlm", None, bv, bt,
                             dict(space=sp, k=k, temp=str(tp)),
                             kinds=("prob", "logit"))

    # ---- arm C: Nadaraya-Watson (full memory, one bandwidth) ---------------
    log("  Nadaraya-Watson: 3 spaces x 10 bandwidths x 21 lambda = 630 configs")
    for sp in cpc.RETRIEVAL_SPACES:
        for h in NW_H:
            bv, neff = nw_bar_scores(ev_val, sp, h)
            bt, _ = nw_bar_scores(ev_test, sp, h)
            sweep_lambda("nadaraya_watson", None, bv, bt,
                         dict(space=sp, bandwidth=h, eff_neighbours=round(neff, 2)))

    # ---- arm D: nearest-centroid prototype ---------------------------------
    log("  prototype: 3 spaces x 6 tau x 21 lambda = 378 configs")
    for sp in cpc.RETRIEVAL_SPACES:
        for tau in PROTO_TAU:
            sweep_lambda("prototype", None,
                         prototype_bar_scores(ev_val, sp, tau),
                         prototype_bar_scores(ev_test, sp, tau),
                         dict(space=sp, tau=tau))

    # ---- arm E: T3A on the routing space -----------------------------------
    log("  T3A (routing space): 3 spaces x 6 M x 21 lambda = 378 configs")
    for sp in cpc.RETRIEVAL_SPACES:
        for m in T3A_M:
            sweep_lambda("t3a_routing", None,
                         t3a_bar_scores(ev_val, sp, m),
                         t3a_bar_scores(ev_test, sp, m),
                         dict(space=sp, support_M=m))

    # ---- arm F: AdaNPC ------------------------------------------------------
    log("  AdaNPC: 3 spaces x 6 k x 3 tau x 4 margins = 216 configs (vote-only), "
        "x 21 lambda for the fused variant")
    best_f = None
    for sp in cpc.RETRIEVAL_SPACES:
        for k in (8, 16, 32, 64, 128, 256):
            for tau in ADANPC_TAU:
                for mg in ADANPC_MARGIN:
                    bv, nadd = adanpc_bar_scores(ev_val, sp, k, tau, mg)
                    a = ev_val.bi.auc_bar(ev_val.bi.spread(bv))
                    sweeps.append(dict(arm="adanpc_vote", space=sp, k=k, tau=tau,
                                       margin=mg, lam=1.0, logit_space=False,
                                       n_pseudo_added=nadd, val_auc_bar=a,
                                       val_auc_row=ev_val.bi.auc_row(ev_val.bi.spread(bv))))
                    if best_f is None or a > best_f[0]:
                        best_f = (a, sp, k, tau, mg)
                    # fused variant, our extension, declared
                    bt, _ = adanpc_bar_scores(ev_test, sp, k, tau, mg)
                    sweep_lambda("adanpc_fused", None, bv, bt,
                                 dict(space=sp, k=k, tau=tau, margin=mg))
    _, sp_f, k_f, tau_f, mg_f = best_f
    bt_f, nadd_f = adanpc_bar_scores(ev_test, sp_f, k_f, tau_f, mg_f)
    arms[("adanpc_vote", "prob")] = dict(
        arm="adanpc_vote", logit_space=False, val_auc_bar=best_f[0], val_auc_row=np.nan,
        hp=dict(space=sp_f, k=k_f, tau=tau_f, margin=mg_f, lam=1.0,
                n_pseudo_added_test=nadd_f),
        bar_test=bt_f, kind="retrieval_only")
    log(f"  adanpc_vote selected space={sp_f} k={k_f} tau={tau_f} margin={mg_f} "
        f"(val bar {best_f[0]:.4f}, {nadd_f} pseudo-labelled bars appended on test)")

    pd.DataFrame(sweeps).to_csv(out() / "retrieval_validation_sweep.csv", index=False)
    log(f"  wrote retrieval_validation_sweep.csv ({len(sweeps)} configs)")

    # ---- test, once ---------------------------------------------------------
    rows = []
    preds = {"frozen": ev_test.frozen, "r3ttt_sf": ev_test.sf}
    for (arm, kind), rec in sorted(arms.items()):
        if rec["kind"] == "retrieval_only":
            p = ev_test.bi.spread(rec["bar_test"])
        else:
            p = interpolate(ev_test, rec["bar_test"], rec["hp"]["lam"],
                            space_kind=rec["kind"])
        name = arm if kind == "prob" else f"{arm}_logit"
        preds[name] = p
        rows.append(dict(arm=name, selected=json.dumps(rec["hp"], default=str),
                         val_auc_bar=rec["val_auc_bar"], val_auc_row=rec["val_auc_row"]))

    # ---- arm B': SELECTION-FREE k-NN-LM, 72 members, nothing selected -------
    # The apples-to-apples comparator for R3-TTT-SF: the same 72-member budget
    # (3 routing spaces x 4 k x 2 kernel temperatures x 3 interpolation weights,
    # the same axis VALUES the published grid uses), uniformly averaged, no
    # validation consulted for anything.  This arm removes the selection
    # confound that the swept k-NN-LM arm above carries.
    def knnlm_sf72(env: Env, kind="prob") -> np.ndarray:
        total = np.zeros(env.bi.n_rows, dtype=float)
        count = 0
        for sp in cpc.RETRIEVAL_SPACES:
            for k in cpc.SCALES:
                for tp in cpc.TEMPERATURES:
                    bar = knn_bar_scores(env, sp, k, tp)
                    for lam in cpc.BLENDS:
                        total += interpolate(env, bar, lam, space_kind=kind)
                        count += 1
        assert count == cpc.N_ENSEMBLE_MEMBERS, count
        return total / count

    for kind in ("prob", "logit"):
        name = "knnlm_sf72" if kind == "prob" else "knnlm_sf72_logit"
        pv = knnlm_sf72(ev_val, kind)
        preds_extra = knnlm_sf72(ev_test, kind)
        rows.append(dict(arm=name,
                         selected=json.dumps({"selection": "NONE -- uniform average "
                                              "over 72 members", "kind": kind}),
                         val_auc_bar=ev_val.bi.auc_bar(pv),
                         val_auc_row=ev_val.bi.auc_row(pv)))
        preds[name] = preds_extra
        log(f"  {name}: val bar {ev_val.bi.auc_bar(pv):.4f} -> test bar "
            f"{ev_test.bi.auc_bar(preds_extra):.4f}")

    # ---- the R3-TTT -> k-NN-LM decomposition ladder -------------------------
    # If the machinery-free ensemble wins, the useful question is WHICH piece of
    # the machinery costs.  Every rung is selection-free: nothing below consults
    # validation, and the axis VALUES are the published ones.
    def sf_variant(env: Env, *, scales=cpc.SCALES, k_as_member=False,
                   blends=cpc.BLENDS, temps=cpc.TEMPERATURES,
                   priors=cpc.PRIOR_STRENGTHS, gates=cpc.GATES):
        total = np.zeros(env.bi.n_rows, dtype=float)
        count = 0
        for sp in cpc.RETRIEVAL_SPACES:
            s = env.spaces[sp]
            idx, dist = s["idx"], s["dist"]
            scale_sets = [(k,) for k in scales] if k_as_member else [tuple(scales)]
            for sc in scale_sets:
                st = cpc._fast_states(                      # noqa: SLF001
                    idx, dist, env.memory.labels,
                    balanced=cpc.BALANCED_DEFAULT, scales=sc, temperatures=temps,
                    prior_strengths=priors, gates=gates)
                for (_, _, _), (bias_bar, trust_bar) in st.items():
                    bias = env.bi.spread(bias_bar)
                    trust = env.bi.spread(trust_bar)
                    for beta in blends:
                        total += cpc.fuse(env.frozen, bias, trust, beta)
                        count += 1
        return total / count, count

    chk, nchk = sf_variant(ev_test)
    d_chk = float(np.max(np.abs(chk - ev_test.sf)))
    log(f"  sf_variant identity check: {nchk} members, max|diff| vs "
        f"r3ttt_selection_free = {d_chk:.3e}")
    if d_chk > 1e-12 or nchk != cpc.N_ENSEMBLE_MEMBERS:
        raise AssertionError("sf_variant does not reproduce the published SF arm")

    LADDER = [
        ("R0  R3-TTT-SF as published", dict()),
        ("R1  - trust gate", dict(gates=(False,))),
        ("R2  - Beta-Bernoulli prior shrinkage", dict(priors=(1e-6,))),
        ("R3  k as a member axis (no internal k-average)", dict(k_as_member=True)),
        ("R4  - gate - prior", dict(gates=(False,), priors=(1e-6,))),
        ("R5  - gate - prior + k as a member axis  (= k-NN-LM, 72 members)",
         dict(gates=(False,), priors=(1e-6,), k_as_member=True)),
    ]
    ladder_rows = []
    ladder_preds = {}
    for label, kw in LADDER:
        pv, _ = sf_variant(ev_val, **kw)
        pt, nm = sf_variant(ev_test, **kw)
        rec = dict(rung=label, n_members=nm,
                   val_auc_bar=ev_val.bi.auc_bar(pv),
                   val_auc_row=ev_val.bi.auc_row(pv),
                   **ev_test.bi.scores(pt),
                   **compare(ev_test.bi, pt, ev_test.sf, n_boot=args.n_boot,
                             prefix="vs_sf__"),
                   **compare(ev_test.bi, pt, ev_test.frozen, n_boot=args.n_boot,
                             prefix="vs_frozen__"))
        ladder_rows.append(rec)
        ladder_preds[f"ladder_{label.split()[0]}"] = pt
        log(f"  {label:58s} n={nm:3d}  val bar {rec['val_auc_bar']:.4f}  "
            f"test bar {rec['auc_bar']:.4f}  vs SF "
            f"{rec['vs_sf__bar_auc_delta']:+.4f} "
            f"[{rec['vs_sf__bar_auc_ci_lo']:+.4f},{rec['vs_sf__bar_auc_ci_hi']:+.4f}] "
            f"P={rec['vs_sf__bar_auc_p_le_zero']:.4f}")
    save(pd.DataFrame(ladder_rows), "retrieval_decomposition_ladder.csv")

    # ---- the lambda curve at the selected (space, k, kernel), both estimands
    sel_knnlm = json.loads([r for r in rows if r["arm"] == "knnlm"][0]["selected"])
    tp_sel = sel_knnlm["temp"]
    tp_sel = tp_sel if tp_sel == "unif" else float(tp_sel)
    bar_v = knn_bar_scores(ev_val, sel_knnlm["space"], sel_knnlm["k"], tp_sel)
    bar_t = knn_bar_scores(ev_test, sel_knnlm["space"], sel_knnlm["k"], tp_sel)
    curve = []
    for lam in LAMBDAS:
        pv = interpolate(ev_val, bar_v, lam)
        pt = interpolate(ev_test, bar_t, lam)
        curve.append(dict(lam=float(lam), val_auc_bar=ev_val.bi.auc_bar(pv),
                          val_auc_row=ev_val.bi.auc_row(pv),
                          test_auc_bar=ev_test.bi.auc_bar(pt),
                          test_auc_row=ev_test.bi.auc_row(pt),
                          test_brier_bar=brier(ev_test.bi.y_bar, ev_test.bi.pool(pt))))
    save(pd.DataFrame(curve), "retrieval_knnlm_lambda_curve.csv")

    # ---- arm G: AdaNPC-style growth applied to OUR memory -------------------
    log("  R3-TTT + AdaNPC pseudo-label memory growth: 6 margins")
    p_check, _ = r3ttt_growth(ev_val, 2.0, verify_against=ev_val.sf)
    del p_check
    best_g = None
    for mg in GROWTH_MARGIN:
        pv, nadd = r3ttt_growth(ev_val, mg)
        a = ev_val.bi.auc_bar(pv)
        sweeps.append(dict(arm="r3ttt_growth", margin=mg, lam=np.nan,
                           logit_space=False, n_pseudo_added=nadd,
                           val_auc_bar=a, val_auc_row=ev_val.bi.auc_row(pv)))
        log(f"    growth margin {mg}: {nadd} bars appended, val bar AUC {a:.4f}")
        if best_g is None or a > best_g[0]:
            best_g = (a, mg)
    _, mg_g = best_g
    p_growth, nadd_g = r3ttt_growth(ev_test, mg_g, verify_against=(
        ev_test.sf if mg_g >= 1.0 else None))
    preds["r3ttt_growth"] = p_growth
    rows.append(dict(arm="r3ttt_growth",
                     selected=json.dumps({"margin": mg_g,
                                          "n_pseudo_added_test": nadd_g}),
                     val_auc_bar=best_g[0], val_auc_row=np.nan))
    # every margin scored on test too, so the immutability price is a curve
    growth_curve = []
    for mg in GROWTH_MARGIN:
        pg, na = r3ttt_growth(ev_test, mg)
        growth_curve.append(dict(
            margin=mg, n_pseudo_added=na, **ev_test.bi.scores(pg),
            **compare(ev_test.bi, pg, ev_test.sf, n_boot=args.n_boot,
                      prefix="vs_sf__")))
        log(f"    TEST growth margin {mg}: +{na} bars, bar AUC "
            f"{growth_curve[-1]['auc_bar']:.4f} (SF {ev_test.bi.auc_bar(ev_test.sf):.4f})")
    save(pd.DataFrame(growth_curve), "retrieval_growth_curve.csv")

    pd.DataFrame(sweeps).to_csv(out() / "retrieval_validation_sweep.csv", index=False)

    # ---- score everything at both estimands, with paired intervals ----------
    table = []
    t_cost = {}
    for name, p in preds.items():
        rec = {"arm": name, **ev_test.bi.scores(p)}
        if name != "frozen":
            rec.update(compare(ev_test.bi, p, ev_test.frozen,
                               n_boot=args.n_boot, prefix="vs_frozen__"))
        if name != "r3ttt_sf":
            rec.update(compare(ev_test.bi, p, ev_test.sf,
                               n_boot=args.n_boot, prefix="vs_sf__"))
        sel = [r for r in rows if r["arm"] == name]
        rec["selected"] = sel[0]["selected"] if sel else ""
        rec["val_auc_bar"] = sel[0]["val_auc_bar"] if sel else np.nan
        table.append(rec)
        log(f"  {name:22s} bar {rec['auc_bar']:.4f}  row {rec['auc_row']:.4f}"
            + (f"  vs SF {rec.get('vs_sf__bar_auc_delta', float('nan')):+.4f} "
               f"[{rec.get('vs_sf__bar_auc_ci_lo', float('nan')):+.4f},"
               f"{rec.get('vs_sf__bar_auc_ci_hi', float('nan')):+.4f}] "
               f"P={rec.get('vs_sf__bar_auc_p_le_zero', float('nan')):.4f}"
               if name != "r3ttt_sf" else ""))
    frame = pd.DataFrame(table)
    save(frame, "retrieval_main.csv")

    # ---- cost accounting ----------------------------------------------------
    cost_rows = []
    n_bars = ev_test.bi.n_bars
    for name, fn in (
        ("knnlm", lambda: interpolate(ev_test, knn_bar_scores(ev_test, sp_a, k_a, 1.0), 0.5)),
        ("knnlm_sf72", lambda: knnlm_sf72(ev_test, "prob")),
        ("r3ttt_sf", lambda: cpc.r3ttt_selection_free(ev_test.memory, ev_test.stream,
                                                      ev_test.frozen)[0]),
        ("r3ttt_growth", lambda: r3ttt_growth(ev_test, mg_g)[0]),
    ):
        t0 = time.perf_counter()
        fn()
        w = time.perf_counter() - t0
        cost_rows.append(dict(arm=name, backward_per_bar=0, adapted_parameters=0,
                              wall_clock_total_s=round(w, 4),
                              wall_clock_ms_per_bar=round(1000 * w / n_bars, 4),
                              n_bars=n_bars))
    t_cost.update({r["arm"]: r for r in cost_rows})
    save(pd.DataFrame(cost_rows), "retrieval_cost.csv")

    # ---- HAC Diebold-Mariano, per bar, vs SF -------------------------------
    dm = []
    yb = ev_test.bi.y_bar
    sfb = ev_test.bi.pool(ev_test.sf)
    for name, p in preds.items():
        if name == "r3ttt_sf":
            continue
        pb = ev_test.bi.pool(p)
        for metric, lf in (("brier", lambda y, q: (q - y) ** 2),
                           ("logloss", lambda y, q: -(y * np.log(np.clip(q, 1e-12, 1))
                                                      + (1 - y) * np.log(np.clip(1 - q, 1e-12, 1))))):
            r = hac_dm(lf(yb, sfb), lf(yb, pb))
            dm.append(dict(arm=name, metric=metric, direction="SF minus arm", **r))
    save(pd.DataFrame(dm), "retrieval_hac_dm.csv")

    np.savez_compressed(out() / "retrieval_test_predictions.npz",
                        datetime=ev_test.stream["datetime"].astype(str).to_numpy(),
                        y=ev_test.bi.y_row, **preds, **ladder_preds)
    return frame


# =============================================================================
# stage: hedge -- exponentially weighted aggregation over the 72-member grid
# =============================================================================
def stage_hedge(args, ev_val: Env, ev_test: Env) -> pd.DataFrame:
    log("stage hedge -- EWA/Hedge over the published 72-member grid, delayed feedback")
    rows = []

    def members(env: Env):
        _, _, _, n, mem = cpc.r3ttt_selection_free(
            env.memory, env.stream, env.frozen, return_members=True)
        M = np.stack([m["probability"] for m in mem], axis=0)
        assert M.shape[0] == n
        return M, mem

    def run_hedge(env: Env, M: np.ndarray, eta: float, *, loss="logloss"):
        """w_m <- w_m exp(-eta * L_m) on MATURED rows only; predict the weighted
        average.  Labels arrive with the protocol's 1 h delay, so the update is
        F_b-measurable at the next bar.  This arm consumes deployment outcomes;
        R3-TTT does not."""
        bi = env.bi
        avail = pd.to_datetime(env.stream["available_time"]).to_numpy()
        order = np.argsort(avail, kind="stable")
        y = bi.y_row.astype(float)
        n_m = M.shape[0]
        logw = np.zeros(n_m)
        p = np.zeros(bi.n_rows)
        cursor = 0
        n_updates = 0
        for t, locs in enumerate(bi.groups):
            now = bi.bar_times[t]
            fresh = []
            while cursor < bi.n_rows and avail[order[cursor]] <= now:
                fresh.append(order[cursor])
                cursor += 1
            if fresh:
                f = np.asarray(fresh, dtype=int)
                q = np.clip(M[:, f], 1e-6, 1 - 1e-6)
                if loss == "logloss":
                    L = -(y[f] * np.log(q) + (1 - y[f]) * np.log(1 - q)).sum(axis=1)
                else:
                    L = ((q - y[f]) ** 2).sum(axis=1)
                logw -= eta * L
                logw -= logw.max()
                n_updates += len(f)
            w = np.exp(logw - logw.max())
            w /= w.sum()
            p[locs] = w @ M[:, locs]
        return p, int(n_updates), float(np.exp(logw - logw.max()).max()
                                        / np.exp(logw - logw.max()).sum())

    Mv, _ = members(ev_val)
    Mt, grid = members(ev_test)
    ETAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    best = None
    for loss in ("logloss", "brier"):
        for eta in ETAS:
            pv, nup, wmax = run_hedge(ev_val, Mv, eta, loss=loss)
            a = ev_val.bi.auc_bar(pv)
            rows.append(dict(arm="hedge", eta=eta, loss=loss, n_label_updates=nup,
                             max_weight=wmax, val_auc_bar=a,
                             val_auc_row=ev_val.bi.auc_row(pv)))
            if best is None or a > best[0]:
                best = (a, eta, loss)
    _, eta_b, loss_b = best
    log(f"  hedge selected eta={eta_b} loss={loss_b} (val bar {best[0]:.4f}); "
        f"eta=0 is exactly the uniform 72-member average")
    save(pd.DataFrame(rows), "hedge_validation_sweep.csv")

    p_hedge, nup, wmax = run_hedge(ev_test, Mt, eta_b, loss=loss_b)
    # validation-argmax member, the other selection comparator
    val_member_auc = np.array([ev_val.bi.auc_bar(Mv[i]) for i in range(Mv.shape[0])])
    j = int(np.argmax(val_member_auc))
    p_argmax = Mt[j]
    out_rows = []
    for name, p in (("hedge_ewa", p_hedge), ("validation_argmax_member", p_argmax),
                    ("r3ttt_sf", ev_test.sf), ("frozen", ev_test.frozen)):
        rec = dict(arm=name, **ev_test.bi.scores(p))
        if name != "r3ttt_sf":
            rec.update(compare(ev_test.bi, p, ev_test.sf, n_boot=args.n_boot,
                               prefix="vs_sf__"))
        if name != "frozen":
            rec.update(compare(ev_test.bi, p, ev_test.frozen, n_boot=args.n_boot,
                               prefix="vs_frozen__"))
        out_rows.append(rec)
        log(f"  {name:26s} bar {rec['auc_bar']:.4f} row {rec['auc_row']:.4f}")
    frame = pd.DataFrame(out_rows)
    frame["n_label_updates"] = [nup, np.nan, np.nan, np.nan]
    frame["max_member_weight"] = [wmax, np.nan, np.nan, np.nan]
    frame["selected_member"] = ["", json.dumps(
        {k: str(v) for k, v in grid[j].items() if k != "probability"}), "", ""]
    save(frame, "hedge_main.csv")
    return frame


# =============================================================================
# stage: neural (matched backbones on the CLEAN panel)
# =============================================================================
def stage_neural(args, ev_val: Env, ev_test: Env) -> pd.DataFrame:
    import copy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import QuantileTransformer
    import exp_gradient_tta_lib as L

    torch.set_num_threads(args.torch_threads)
    log(f"stage neural -- matched backbones, CLEAN panel, torch threads="
        f"{args.torch_threads}")

    # ---------------------------------------------------------------- methods
    class IABN1d(nn.Module):
        """NOTE's Instance-Aware Batch Norm, ported to a bar of tabular rows.

        NOTE computes instance statistics over each sample's spatial/temporal
        axis and corrects the SOURCE statistic only where the deviation exceeds
        k standard errors.  Here the analogous axis is the bar: a bar is a set of
        correlated rows sharing one outcome, which is the object NOTE's IABN
        exists to normalise.  Declared as a port, not a faithful re-run.
        Zero gradient steps; the affine parameters are the only trainable state
        and they are touched only by the PBRS half.
        """

        def __init__(self, bn: nn.BatchNorm1d, k: float = 4.0):
            super().__init__()
            self.k = float(k)
            self.eps = bn.eps
            self.weight = nn.Parameter(bn.weight.detach().clone())
            self.bias = nn.Parameter(bn.bias.detach().clone())
            self.register_buffer("src_mean", bn.running_mean.detach().clone())
            self.register_buffer("src_var", bn.running_var.detach().clone())

        def forward(self, x):
            n = x.shape[0]
            mu_s, var_s = self.src_mean, self.src_var
            mu_i = x.mean(0)
            se_mu = torch.sqrt(var_s / max(n, 1))
            if n < 2:
                # the instance variance is undefined on a singleton bar; NOTE's
                # correction then reduces to the source variance, which is the
                # only well-defined behaviour and is applied uniformly.
                var_i, se_var = var_s, torch.ones_like(var_s)
            else:
                var_i = x.var(0, unbiased=True)
                se_var = var_s * (2.0 / (n - 1)) ** 0.5
            d_mu = mu_i - mu_s
            d_var = var_i - var_s
            mu = mu_s + torch.sign(d_mu) * torch.clamp(d_mu.abs() - self.k * se_mu, min=0.0)
            var = var_s + torch.sign(d_var) * torch.clamp(d_var.abs() - self.k * se_var, min=0.0)
            var = torch.clamp(var, min=self.eps)
            return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias

    def convert_iabn(model: nn.Module, k: float) -> nn.Module:
        for name, child in model.named_children():
            if isinstance(child, nn.BatchNorm1d):
                setattr(model, name, IABN1d(child, k))
            else:
                convert_iabn(child, k)
        return model

    class NOTE_IABN(L.BaseTTA):
        """NOTE minus PBRS: IABN only.  ZERO gradient steps, single forward."""
        name = "note_iabn"

        def __init__(self, model, k=4.0):
            super().__init__(model)
            self.model.eval()
            self.model.requires_grad_(False)
            convert_iabn(self.model, k)
            self.model.eval()
            self.model.requires_grad_(False)
            self.cost.adapted_params = 0

        @torch.no_grad()
        def __call__(self, x):
            self.cost.forward += 1
            return F.softmax(self.model(x), dim=1)[:, 1].numpy()

    class NOTE(L.BaseTTA):
        """NOTE = IABN + prediction-balanced reservoir sampling + entropy step
        on the IABN affine parameters.  This is the gradient half."""
        name = "note"

        def __init__(self, model, k=4.0, lr=1e-3, capacity=64):
            super().__init__(model)
            self.model.eval()
            self.model.requires_grad_(False)
            convert_iabn(self.model, k)
            params = []
            for m in self.model.modules():
                if isinstance(m, IABN1d):
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
                    params += [m.weight, m.bias]
            self.params = params
            self.opt = torch.optim.Adam(params, lr=lr) if params else None
            self.cost.adapted_params = sum(p.numel() for p in params)
            self.capacity = capacity
            self.bank = {0: [], 1: []}
            self.seen = {0: 0, 1: 0}
            self.rng = np.random.default_rng(SEED)

        def _push(self, row, c):
            self.seen[c] += 1
            half = self.capacity // 2
            if len(self.bank[c]) < half:
                self.bank[c].append(row)
            else:
                j = int(self.rng.integers(0, self.seen[c]))
                if j < half:
                    self.bank[c][j] = row

        @torch.no_grad()
        def _predict(self, x):
            return F.softmax(self.model(x), dim=1)[:, 1]

        def __call__(self, x):
            self.cost.forward += 1
            p = self._predict(x)
            for i in range(x.shape[0]):
                self._push(x[i].detach().clone(), int(p[i] >= 0.5))
            pool = self.bank[0] + self.bank[1]
            if self.opt is not None and len(pool) >= 2:
                xb = torch.stack(pool)
                out = self.model(xb)
                self.cost.forward += 1
                loss = L.softmax_entropy(out).mean(0)
                loss.backward()
                self.cost.backward += 1
                self.opt.step()
                self.opt.zero_grad()
                self.cost.opt_steps += 1
            return p.numpy()

    # ---- LAME ---------------------------------------------------------------
    def _knn_affinity(X, knn):
        N = X.size(0)
        dist = torch.cdist(X, X)
        n_neighbors = min(knn + 1, N)
        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]
        W = torch.zeros(N, N)
        if knn_index.numel():
            W.scatter_(dim=-1, index=knn_index, value=1.0)
        return W

    def _rbf_affinity(X, knn):
        N = X.size(0)
        dist = torch.cdist(X, X)
        n_neighbors = min(knn, N)
        kth = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]
        sigma = kth.mean().clamp_min(1e-8)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))

    def _linear_affinity(X, knn=None):
        return X @ X.t()

    def laplacian_optimization(unary, kernel, bound_lambda=1.0, max_steps=100):
        """Verbatim from code_ref/roid/classification/methods/lame.py."""
        oldE = float("inf")
        Y = (-unary).softmax(-1)
        for i in range(max_steps):
            pairwise = bound_lambda * kernel.matmul(Y)
            Y = (-unary + pairwise).softmax(-1)
            E = ((unary * Y - bound_lambda * pairwise * Y
                  + Y * torch.log(Y.clip(1e-20))).sum()).item()
            if i > 1 and abs(E - oldE) <= 1e-8 * abs(oldE):
                break
            oldE = E
        return Y

    def split_model(model, kind):
        if kind.startswith("mlp"):
            return (lambda x: model.body(x)), model.head
        def feat(x):
            tok = x.unsqueeze(-1) * model.feat_w + model.feat_b
            tok = torch.cat([model.cls.expand(x.size(0), -1, -1), tok], dim=1)
            return model.norm(model.enc(tok)[:, 0])
        return feat, model.head

    class LAME(L.BaseTTA):
        """Faithful LAME: the affinity graph is built WITHIN the test batch, and
        our test batch is one bar (mean 3.5 rows; 242/407 validation bars are
        singletons, where laplacian_optimization returns softmax(-unary) exactly
        and LAME is the identity)."""
        name = "lame"

        def __init__(self, model, kind, affinity="rbf", knn=5, window=0,
                     force_symmetry=False):
            super().__init__(model)
            self.model.eval()
            self.model.requires_grad_(False)
            self.feat, self.clf = split_model(self.model, kind)
            self.aff = {"knn": _knn_affinity, "rbf": _rbf_affinity,
                        "linear": _linear_affinity}[affinity]
            self.affinity_name = affinity
            self.knn = knn
            self.window = int(window)
            self.force_symmetry = force_symmetry
            self.buf = []
            self.cost.adapted_params = 0
            self.n_singleton = 0

        @torch.no_grad()
        def __call__(self, x):
            self.cost.forward += 1
            cur = x
            if self.window > 0 and self.buf:
                past = torch.cat(self.buf[-self.window:], dim=0)
                xin = torch.cat([cur, past], dim=0)
            else:
                xin = cur
            f = self.feat(xin)
            logits = self.clf(f)
            unary = -torch.log(logits.softmax(dim=1) + 1e-10)
            fn = F.normalize(f, p=2, dim=-1)
            kernel = self.aff(fn, self.knn) if self.affinity_name != "linear" \
                else _linear_affinity(fn)
            if self.force_symmetry:
                kernel = 0.5 * (kernel + kernel.t())
            if xin.shape[0] == 1:
                self.n_singleton += 1
            Y = laplacian_optimization(unary, kernel)
            if self.window > 0:
                self.buf.append(cur)
                if len(self.buf) > 512:
                    self.buf = self.buf[-512:]
            self.cost.extra["n_singleton_batches"] = self.n_singleton
            return Y[:cur.shape[0], 1].numpy()

    class T3A(L.BaseTTA):
        """T3A: pseudo-labelled support sets over the penultimate features, the
        linear head's weight vectors as the initial prototypes, M lowest-entropy
        members retained per class.  Zero backward passes."""
        name = "t3a"

        def __init__(self, model, kind, M=20):
            super().__init__(model)
            self.model.eval()
            self.model.requires_grad_(False)
            self.feat, self.clf = split_model(self.model, kind)
            W = self.clf.weight.detach().clone()               # (2, d)
            self.sup = {c: [(0.0, W[c])] for c in (0, 1)}
            self.M = M
            self.cost.adapted_params = 0

        @torch.no_grad()
        def __call__(self, x):
            self.cost.forward += 1
            f = self.feat(x)
            logits = self.clf(f)
            p = logits.softmax(1)
            ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1)
            for i in range(x.shape[0]):
                c = int(p[i, 1] >= 0.5)
                self.sup[c].append((float(ent[i]), f[i]))
                if self.M > 0 and len(self.sup[c]) > self.M:
                    self.sup[c] = sorted(self.sup[c], key=lambda z: z[0])[:self.M]
            cent = torch.stack([torch.stack([v for _, v in self.sup[c]]).mean(0)
                                for c in (0, 1)])              # (2, d)
            s = F.normalize(f, dim=-1) @ F.normalize(cent, dim=-1).t()
            return s.softmax(1)[:, 1].numpy()

    class AlphaBN(L.BaseTTA):
        """alpha-BN: running = (1-a)*source + a*test-batch.  a=1 is bn_stats,
        a=0 is frozen.  The competent practitioner's minimal fix for the 3-row
        BatchNorm collapse; no gradients."""
        name = "alpha_bn"

        def __init__(self, model, alpha=0.1):
            super().__init__(model)
            self.model.eval()
            self.model.requires_grad_(False)
            self.alpha = float(alpha)
            self.bns = [m for m in self.model.modules() if isinstance(m, nn.BatchNorm1d)]
            self.src = [(m.running_mean.clone(), m.running_var.clone()) for m in self.bns]
            self.cost.adapted_params = 0
            self.hooks = []

        @torch.no_grad()
        def __call__(self, x):
            self.cost.forward += 1
            a = self.alpha
            handles = []
            for m, (sm, sv) in zip(self.bns, self.src):
                def pre(mod, inp, sm=sm, sv=sv, mod_ref=m):
                    z = inp[0]
                    if z.shape[0] >= 2:
                        bm, bv = z.mean(0), z.var(0, unbiased=True)
                    else:
                        bm, bv = z.mean(0), sv
                    mod_ref.running_mean = (1 - a) * sm + a * bm
                    mod_ref.running_var = (1 - a) * sv + a * bv
                    return None
                handles.append(m.register_forward_pre_hook(pre))
            self.model.eval()
            out = self.model(x)
            for h in handles:
                h.remove()
            for m, (sm, sv) in zip(self.bns, self.src):
                m.running_mean, m.running_var = sm.clone(), sv.clone()
            return F.softmax(out, dim=1)[:, 1].numpy()

    # ---------------------------------------------------------------- data
    panel = args._panel
    tr = panel[panel.datetime < cpc.TRAIN_END].reset_index(drop=True)
    va = panel[(panel.datetime >= cpc.TRAIN_END)
               & (panel.datetime < cpc.TEST_START)].reset_index(drop=True)
    pretest = panel[panel.datetime < cpc.TEST_START].reset_index(drop=True)
    te = panel[panel.datetime >= cpc.TEST_START].reset_index(drop=True)
    FEATS = cpc.MODEL_FEATURES
    y_tr, y_va, y_te = (d.target_hi_vol.to_numpy(int) for d in (tr, va, te))
    y_pre = pretest.target_hi_vol.to_numpy(int)
    g_va, g_te = cpc.bar_groups(va), cpc.bar_groups(te)
    bt_va = va.groupby("datetime", sort=True).size().index.to_numpy()
    bt_te = te.groupby("datetime", sort=True).size().index.to_numpy()
    av_va, av_te = va.available_time.to_numpy(), te.available_time.to_numpy()
    x_tr_r, x_va_r = L.frame_matrix(tr, FEATS), L.frame_matrix(va, FEATS)
    x_pre_r, x_te_r = L.frame_matrix(pretest, FEATS), L.frame_matrix(te, FEATS)
    bi_va, bi_te = BarIndex(va), BarIndex(te)

    def make_prep(kind, xfit):
        if kind == "zscore":
            s = L.Standardizer.fit(xfit)
            return lambda x: s(x)
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                 subsample=10 ** 9, random_state=0).fit(xfit)
        return lambda x: qt.transform(x)

    sweep_file = ROOT / "config" / "arch_sweep2.json"
    arch_sweep = json.load(open(sweep_file))
    BACKBONES = {}
    for kind in ("mlp_bn", "ft_trans"):
        best = max([r for r in arch_sweep if r["kind"] == kind],
                   key=lambda r: r["val_auc"])
        BACKBONES[kind] = dict(
            prep=best["prep"],
            arch={k: v for k, v in best.items()
                  if k not in ("prep", "kind", "epochs", "lr", "wd", "val_auc")},
            epochs=best["epochs"], lr=best["lr"], wd=best["wd"])
    log(f"  backbone architectures reused from the contaminated-panel sweep "
        f"(declared deviation): {json.dumps(BACKBONES)}")

    SEEDS = tuple(range(args.n_seeds))

    def fit_models(kind, xfit, yfit):
        c = BACKBONES[kind]
        return [L.train_backbone(kind, xfit, yfit, seed=s, epochs=c["epochs"],
                                 lr=c["lr"], weight_decay=c["wd"], arch=c["arch"])
                for s in SEEDS]

    def make_method(name, model, hp, kind, d_in):
        if name == "frozen":
            return L.Frozen(model)
        if name == "bn_stats":
            return L.BNStatsOnly(model)
        if name == "alpha_bn":
            return AlphaBN(model, alpha=hp["alpha"])
        if name == "tent":
            return L.Tent(model, lr=hp["lr"])
        if name == "eata":
            return L.EATA(model, lr=hp["lr"], fishers=hp.get("fishers"),
                          d_margin=hp.get("d_margin", 0.05))
        if name == "eta":
            return L.EATA(model, lr=hp["lr"], fishers=None,
                          d_margin=hp.get("d_margin", 0.05))
        if name == "sar":
            return L.SAR(model, lr=hp["lr"], reset_constant=hp.get("reset_constant", 0.2))
        if name == "cotta":
            return L.CoTTA(model, lr=hp["lr"], aug_sigma=hp["aug_sigma"])
        if name == "roid":
            return L.ROID(model, lr=hp["lr"], prior_mode=hp.get("prior_mode", "logits"))
        if name == "tafas":
            return L.TAFAS(model, d_in, lr=hp["lr"], steps=hp.get("steps", 1))
        if name == "note_iabn":
            return NOTE_IABN(model, k=hp["k_sigma"])
        if name == "note":
            return NOTE(model, k=hp["k_sigma"], lr=hp["lr"], capacity=hp.get("capacity", 64))
        if name == "lame":
            return LAME(model, kind, affinity=hp["affinity"], knn=hp["knn"],
                        window=hp.get("window", 0))
        if name == "t3a":
            return T3A(model, kind, M=hp["M"])
        raise ValueError(name)

    def run_method(name, models, x_stream, groups, hp, kind, *, bar_times, avail,
                   y_stream, buffer_init=None, fishers=None):
        ps, costs = [], []
        for i, m in enumerate(models):
            h = dict(hp)
            if name == "eata" and fishers is not None:
                h["fishers"] = fishers[i]
            meth = make_method(name, copy.deepcopy(m), h, kind, x_stream.shape[1])
            p = L.run_stream(meth, x_stream, groups, buffer_rows=h.get("buffer", 0),
                             bar_times=bar_times, available_times=avail,
                             y_stream=y_stream, buffer_init=buffer_init)
            ps.append(p)
            costs.append(meth.cost)
        return np.mean(ps, axis=0), costs

    # -------------------------------------------------- validation grids
    SWEEPS = {
        # existing (grids identical to results/gradient_tta_baselines)
        "bn_stats": [dict(buffer=b) for b in (0, 64)],
        "tent": [dict(lr=lr, buffer=b) for lr in (1e-5, 1e-4, 1e-3, 1e-2) for b in (0, 64)],
        "eata": [dict(lr=lr, buffer=b, d_margin=dm) for lr in (1e-4, 1e-3, 1e-2)
                 for b in (0, 64) for dm in (0.05, 0.5, 2.0)],
        "sar": [dict(lr=lr, buffer=b, reset_constant=rc) for lr in (2.5e-5, 2.5e-4, 2.5e-3)
                for b in (0, 64) for rc in (0.2, 0.02)],
        "cotta": [dict(lr=lr, aug_sigma=sg, buffer=64) for lr in (1e-4, 1e-3)
                  for sg in (0.05, 0.1, 0.3)],
        "roid": [dict(lr=lr, buffer=64, prior_mode=pm) for lr in (1e-5, 1e-4, 1e-3, 1e-2)
                 for pm in ("logits", "posterior")],
        "tafas": [dict(lr=lr, buffer=0, steps=st) for lr in (1e-3, 1e-2) for st in (1, 5, 20)],
        # new
        "eta": [dict(lr=lr, buffer=b, d_margin=dm) for lr in (1e-4, 1e-3, 1e-2)
                for b in (0, 64) for dm in (0.05, 2.0)],
        "alpha_bn": [dict(alpha=a, buffer=b) for a in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
                     for b in (0, 64)],
        "note_iabn": [dict(k_sigma=k, buffer=0) for k in (0.5, 1.0, 2.0, 3.0, 4.0, 8.0)],
        "note": [dict(k_sigma=k, lr=lr, capacity=c, buffer=0)
                 for k in (1.0, 2.0, 4.0) for lr in (1e-4, 1e-3, 1e-2) for c in (64, 256)],
        "lame": [dict(affinity=a, knn=k, window=0, buffer=0)
                 for a in ("knn", "rbf", "linear") for k in (1, 3, 5)],
        "lame_window": [dict(affinity=a, knn=k, window=w, buffer=0)
                        for a in ("knn", "rbf") for k in (5, 10)
                        for w in (1, 8, 32, 128)],
        "t3a": [dict(M=m, buffer=0) for m in (1, 5, 20, 50, 100, -1)],
    }
    BN_ONLY = {"bn_stats", "alpha_bn", "note_iabn", "note"}

    selected, val_records = {}, []
    for kind in BACKBONES:
        prep = make_prep(BACKBONES[kind]["prep"], x_tr_r)
        xtr, xva = prep(x_tr_r), prep(x_va_r)
        models_v = fit_models(kind, xtr, y_tr)
        p_frozen_va = np.mean([L.predict_frozen(m, xva) for m in models_v], axis=0)
        log(f"  [{kind}] frozen neural VAL bar {bi_va.auc_bar(p_frozen_va):.4f} "
            f"row {bi_va.auc_row(p_frozen_va):.4f}")
        fishers = [L.compute_fishers(m, xtr) for m in models_v]
        selected[kind] = {}
        for name, grid in SWEEPS.items():
            if name in BN_ONLY and kind != "mlp_bn":
                continue
            base = "lame" if name == "lame_window" else name
            best = None
            for hp in grid:
                p, _ = run_method(base, models_v, xva, g_va, hp, kind,
                                  bar_times=bt_va, avail=av_va, y_stream=y_va,
                                  buffer_init=xtr,
                                  fishers=fishers if name == "eata" else None)
                a_bar = bi_va.auc_bar(p)
                val_records.append(dict(backbone=kind, method=name,
                                        **{k: v for k, v in hp.items()},
                                        val_auc_bar=a_bar,
                                        val_auc_row=bi_va.auc_row(p)))
                if best is None or a_bar > best[0]:
                    best = (a_bar, hp)
            selected[kind][name] = dict(best[1])
            log(f"  [{kind}] SELECTED {name:12s} {json.dumps(best[1])} "
                f"(val bar {best[0]:.4f})")
    pd.DataFrame(val_records).to_csv(out() / "neural_validation_sweep.csv", index=False)
    jdump(selected, "neural_selected_hyperparameters.json")

    # -------------------------------------------------- test, once
    mem_te = cpc.build_bar_memory(pretest, deploy_start=cpc.TEST_START)
    rows, cost_rows, preds = [], [], {}
    for kind in BACKBONES:
        prep = make_prep(BACKBONES[kind]["prep"], x_pre_r)
        xpre, xte = prep(x_pre_r), prep(x_te_r)
        models_t = fit_models(kind, xpre, y_pre)
        fishers_t = [L.compute_fishers(m, xpre) for m in models_t]
        p_frozen, costs = run_method("frozen", models_t, xte, g_te, {}, kind,
                                     bar_times=bt_te, avail=av_te, y_stream=y_te,
                                     buffer_init=xpre)
        preds[f"{kind}__frozen"] = p_frozen
        cost_rows.append(L.cost_row("frozen", kind, costs[0]))
        log(f"  [{kind}] frozen TEST bar {bi_te.auc_bar(p_frozen):.4f} "
            f"row {bi_te.auc_row(p_frozen):.4f}")
        for name in SWEEPS:
            if name not in selected[kind]:
                continue
            base = "lame" if name == "lame_window" else name
            hp = dict(selected[kind][name])
            p, costs = run_method(base, models_t, xte, g_te, hp, kind,
                                  bar_times=bt_te, avail=av_te, y_stream=y_te,
                                  buffer_init=xpre,
                                  fishers=fishers_t if name == "eata" else None)
            preds[f"{kind}__{name}"] = p
            cr = L.cost_row(name, kind, costs[0])
            cr["selected"] = json.dumps(hp)
            cost_rows.append(cr)
            log(f"  [{kind}] {name:12s} TEST bar {bi_te.auc_bar(p):.4f} "
                f"row {bi_te.auc_row(p):.4f}")
        # R3-TTT on the SAME neural slow model, zero gradient steps
        t0 = time.perf_counter()
        p_r3, _, _, nmm = cpc.r3ttt_selection_free(mem_te, te, p_frozen)
        wall = time.perf_counter() - t0
        preds[f"{kind}__r3ttt"] = p_r3
        c = L.Cost(forward=costs[0].n_bars, backward=0, opt_steps=0, wall=wall,
                   n_bars=len(g_te), n_rows=len(te), adapted_params=0,
                   extra={"n_members": nmm, "memory_items": len(mem_te)})
        cost_rows.append(L.cost_row("r3ttt", kind, c))
        log(f"  [{kind}] r3ttt        TEST bar {bi_te.auc_bar(p_r3):.4f} "
            f"row {bi_te.auc_row(p_r3):.4f}")

    for key, p in preds.items():
        kind, name = key.split("__", 1)
        ref = preds[f"{kind}__frozen"]
        r3 = preds[f"{kind}__r3ttt"]
        rec = dict(backbone=kind, method=name, **bi_te.scores(p))
        if name != "frozen":
            rec.update(compare(bi_te, p, ref, n_boot=args.n_boot, prefix="vs_frozen__"))
        if name != "r3ttt":
            rec.update(compare(bi_te, p, r3, n_boot=args.n_boot, prefix="vs_r3ttt__"))
        if f"{kind}__bn_stats" in preds and name not in ("bn_stats",):
            rec.update(compare(bi_te, p, preds[f"{kind}__bn_stats"],
                               n_boot=args.n_boot, prefix="vs_bnstats__"))
        rec["selected"] = json.dumps(selected[kind].get(name, {}))
        rows.append(rec)
    frame = pd.DataFrame(rows)
    save(frame, "neural_main.csv")
    save(pd.DataFrame(cost_rows), "neural_cost.csv")
    dfp = pd.DataFrame({k: v for k, v in preds.items()})
    dfp.insert(0, "y", bi_te.y_row)
    dfp.insert(0, "datetime", te["datetime"].astype(str).to_numpy())
    save(dfp, "neural_test_predictions.csv")
    # the GBT reference line, printed but never differenced against the neural rows
    jdump({
        "environment": "CLEAN panel, matched neural backbones, refitted in this run",
        "warning": ("this is a NEW environment; its levels may NOT be compared "
                    "with results/gradient_tta_baselines (contaminated panel) nor "
                    "with the GBT clean headline"),
        "gbt_reference_same_panel": {
            "frozen_auc_bar": bi_te.auc_bar(ev_test.frozen),
            "frozen_auc_row": bi_te.auc_row(ev_test.frozen),
            "r3ttt_sf_auc_bar": bi_te.auc_bar(ev_test.sf),
            "r3ttt_sf_auc_row": bi_te.auc_row(ev_test.sf),
        },
    }, "neural_environment_note.json")
    return frame


# =============================================================================
# stage: loqo -- the pre-2026 leave-one-quarter-out confirmation
#
# The 2026 panel is exhausted as a confirmatory set, so any arm that beats
# R3-TTT-SF there has to be checked on the exploration folds before it can be
# believed.  Fold construction is copied from exp_ladder_stage1.run_fold: the
# fold's causal target is re-thresholded on the history median, history is
# everything before the quarter, evaluation is the quarter.  A DISTINCT
# environment; never difference its levels against the clean test panel.
# =============================================================================
LOQO_QUARTERS = ("2025Q1", "2025Q2", "2025Q3", "2025Q4")

LADDER_SPEC = [
    ("R0  R3-TTT-SF as published", dict()),
    ("R1  - trust gate", dict(gates=(False,))),
    ("R2  - Beta-Bernoulli prior shrinkage", dict(priors=(1e-6,))),
    ("R3  k as a member axis (no internal k-average)", dict(k_as_member=True)),
    ("R4  - gate - prior", dict(gates=(False,), priors=(1e-6,))),
    ("R5  - gate - prior + k as a member axis  (= k-NN-LM, 72 members)",
     dict(gates=(False,), priors=(1e-6,), k_as_member=True)),
]


def _sf_variant_generic(frozen, bi, spaces, mem_labels, *, scales=cpc.SCALES,
                        k_as_member=False, blends=cpc.BLENDS,
                        temps=cpc.TEMPERATURES, priors=cpc.PRIOR_STRENGTHS,
                        gates=cpc.GATES):
    total = np.zeros(bi.n_rows, dtype=float)
    count = 0
    for sp in cpc.RETRIEVAL_SPACES:
        idx, dist = spaces[sp]["idx"], spaces[sp]["dist"]
        scale_sets = [(k,) for k in scales] if k_as_member else [tuple(scales)]
        for sc in scale_sets:
            st = cpc._fast_states(                              # noqa: SLF001
                idx, dist, mem_labels, balanced=cpc.BALANCED_DEFAULT,
                scales=sc, temperatures=temps, prior_strengths=priors,
                gates=gates)
            for (_, _, _), (bias_bar, trust_bar) in st.items():
                bias, trust = bi.spread(bias_bar), bi.spread(trust_bar)
                for beta in blends:
                    total += cpc.fuse(frozen, bias, trust, beta)
                    count += 1
    return total / count, count


def stage_loqo(args) -> pd.DataFrame:
    log("stage loqo -- pre-2026 leave-one-quarter-out confirmation of the ladder")
    panel = args._panel.copy()
    folds = []
    for q in LOQO_QUARTERS:
        period = pd.Period(q, freq="Q")
        start = period.start_time
        end = period.end_time + pd.Timedelta(nanoseconds=1)
        # the clean event panel carries tz-naive datetimes; match them
        if getattr(args._panel["datetime"].dtype, "tz", None) is not None:
            start = start.tz_localize("UTC")
            end = end.tz_localize("UTC")
        folds.append((q, start, end))

    per_fold, fold_payload = [], []
    for fi, (q, start, end) in enumerate(folds):
        p = panel.copy()
        thr = float(p.loc[p["datetime"] < start, "abs_ret_fwd_1h"].median())
        p["target_hi_vol"] = (p["abs_ret_fwd_1h"] > thr).astype(int)
        history = p[p["datetime"] < start].reset_index(drop=True)
        ev = p[(p["datetime"] >= start) & (p["datetime"] < end)].reset_index(drop=True)
        if len(ev) < 50 or ev["target_hi_vol"].nunique() < 2:
            log(f"  fold {q}: skipped ({len(ev)} rows)")
            continue
        bi = BarIndex(ev)
        frozen = cpc.slow_probability(history, ev)
        memory = cpc.build_bar_memory(history, deploy_start=start)
        spaces = {}
        for sp, feats in cpc.RETRIEVAL_SPACES.items():
            scaler, mem_x = fit_space(memory.frame, feats)
            qx = bar_queries(transform_rows(scaler, ev, feats), bi.groups)
            idx, dist = sort_topk(distance_matrix(mem_x, qx, len(feats)), cpc.MAX_K)
            spaces[sp] = dict(idx=idx, dist=dist)
        preds = {"frozen": frozen}
        for label, kw in LADDER_SPEC:
            pt, nm = _sf_variant_generic(frozen, bi, spaces, memory.labels, **kw)
            preds[label.split()[0]] = pt
            del nm
        ref, _, _, _ = cpc.r3ttt_selection_free(memory, ev, frozen, groups=bi.groups)
        d0 = float(np.max(np.abs(preds["R0"] - ref)))
        if d0 > 1e-12:
            raise AssertionError(f"fold {q}: R0 != r3ttt_selection_free ({d0:.2e})")
        for name, pp in preds.items():
            per_fold.append(dict(quarter=q, arm=name, n_bars=bi.n_bars,
                                 n_rows=bi.n_rows, n_memory=len(memory),
                                 threshold_bps=thr * 1e4,
                                 auc_bar=bi.auc_bar(pp), auc_row=bi.auc_row(pp),
                                 brier_bar=brier(bi.y_bar, bi.pool(pp))))
        fold_payload.append(dict(quarter=q, y=bi.y_bar, days=bi.bar_times,
                                 scores={k: bi.pool(v) for k, v in preds.items()}))
        log(f"  fold {q}: {bi.n_bars} bars, memory {len(memory)}, frozen "
            f"{bi.auc_bar(frozen):.4f}, R0 {bi.auc_bar(preds['R0']):.4f}, "
            f"R5 {bi.auc_bar(preds['R5']):.4f}")
    save(pd.DataFrame(per_fold), "loqo_per_fold.csv")

    # macro-over-folds paired block bootstrap on MATCHED replicate indices
    arms = [a for a in ("frozen", "R0", "R1", "R2", "R3", "R4", "R5")]
    rng_streams = [np.random.default_rng(SEED + 1000 * i)
                   for i in range(len(fold_payload))]
    draws = {a: np.zeros(args.n_boot) for a in arms}
    for b in range(args.n_boot):
        acc = {a: [] for a in arms}
        for fi, fp in enumerate(fold_payload):
            d = pd.to_datetime(pd.Series(fp["days"])).dt.normalize().to_numpy()
            uniq = np.array(sorted(set(d)))
            index = {u: np.flatnonzero(d == u) for u in uniq}
            n_blocks = max(1, int(np.ceil(len(uniq) / BLOCK_DAYS)))
            high = max(1, len(uniq) - BLOCK_DAYS + 1)
            starts = rng_streams[fi].integers(0, high, size=n_blocks)
            chosen = [u for s in starts for u in uniq[s:s + BLOCK_DAYS]]
            rows = np.concatenate([index[u] for u in chosen])
            for a in arms:
                acc[a].append(auc(fp["y"][rows], fp["scores"][a][rows]))
        for a in arms:
            v = np.asarray(acc[a], dtype=float)
            draws[a][b] = np.nan if np.any(~np.isfinite(v)) else float(np.mean(v))
    macro = {a: float(np.mean([f["scores"][a] is not None for f in fold_payload]))
             for a in arms}
    del macro
    pf = pd.DataFrame(per_fold)
    rows = []
    for a in arms:
        point = float(pf[pf.arm == a]["auc_bar"].mean())
        rows.append(dict(arm=a, macro_auc_bar=point,
                         macro_auc_row=float(pf[pf.arm == a]["auc_row"].mean())))
    base = draws["R0"]
    for r in rows:
        a = r["arm"]
        d = draws[a] - base
        d = d[np.isfinite(d)]
        r["vs_R0_delta"] = (float(pf[pf.arm == a]["auc_bar"].mean())
                            - float(pf[pf.arm == "R0"]["auc_bar"].mean()))
        r["vs_R0_ci_lo"] = float(np.percentile(d, 2.5))
        r["vs_R0_ci_hi"] = float(np.percentile(d, 97.5))
        r["vs_R0_p_le_zero"] = float(np.mean(d <= 0.0))
        df_ = draws[a] - draws["frozen"]
        df_ = df_[np.isfinite(df_)]
        r["vs_frozen_delta"] = (float(pf[pf.arm == a]["auc_bar"].mean())
                                - float(pf[pf.arm == "frozen"]["auc_bar"].mean()))
        r["vs_frozen_ci_lo"] = float(np.percentile(df_, 2.5))
        r["vs_frozen_ci_hi"] = float(np.percentile(df_, 97.5))
        r["vs_frozen_p_le_zero"] = float(np.mean(df_ <= 0.0))
        r["n_folds"] = int((pf.arm == a).sum())
        r["wins_over_R0_in_folds"] = int(
            (pf[pf.arm == a].set_index("quarter")["auc_bar"]
             > pf[pf.arm == "R0"].set_index("quarter")["auc_bar"]).sum())
        log(f"  MACRO {a:8s} bar {r['macro_auc_bar']:.4f}  vs R0 "
            f"{r['vs_R0_delta']:+.4f} [{r['vs_R0_ci_lo']:+.4f},{r['vs_R0_ci_hi']:+.4f}] "
            f"P={r['vs_R0_p_le_zero']:.4f}  ({r['wins_over_R0_in_folds']}/"
            f"{r['n_folds']} folds)")
    frame = pd.DataFrame(rows)
    save(frame, "loqo_macro.csv")
    return frame


# =============================================================================
# stage: figures (reads the CSVs this script wrote; safe to re-run alone)
# =============================================================================
def stage_figs(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200})
    log("stage figs")

    def fin(fig, name):
        fig.tight_layout()
        fig.savefig(out() / name, bbox_inches="tight")
        plt.close(fig)
        log(f"  wrote {name}")

    # (1) the decomposition ladder
    f = out() / "retrieval_decomposition_ladder.csv"
    if f.exists():
        d = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(5.4, 2.4))
        yy = np.arange(len(d))[::-1]
        ax.errorbar(d["vs_sf__bar_auc_delta"], yy,
                    xerr=[d["vs_sf__bar_auc_delta"] - d["vs_sf__bar_auc_ci_lo"],
                          d["vs_sf__bar_auc_ci_hi"] - d["vs_sf__bar_auc_delta"]],
                    fmt="o", ms=3.5, lw=1.0, color="#22343f", capsize=2)
        ax.axvline(0, color="#999999", lw=0.8)
        main = pd.read_csv(out() / "retrieval_main.csv")
        eff = float(main.loc[main.arm == "r3ttt_sf", "auc_bar"].iloc[0]
                    - main.loc[main.arm == "frozen", "auc_bar"].iloc[0])
        ax.axvline(eff, color="#b23a48", lw=0.8, ls="--")
        ax.text(eff, len(d) - 0.4, " published adaptation effect", color="#b23a48",
                fontsize=6.5, va="top")
        ax.set_yticks(yy)
        ax.set_yticklabels([r.split("  ", 1)[1] if "  " in r else r
                            for r in d["rung"]], fontsize=6.5)
        ax.set_xlabel(r"$\Delta$ per-bar AUC vs R3-TTT-SF (paired 5-day block bootstrap)")
        fin(fig, "fig_decomposition_ladder.pdf")

    # (2) lambda curve, validation against test
    f = out() / "retrieval_knnlm_lambda_curve.csv"
    if f.exists():
        d = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        ax.plot(d["lam"], d["test_auc_bar"], "-o", ms=2.5, lw=1.1,
                color="#22343f", label="test, per bar")
        ax2 = ax.twinx()
        ax2.plot(d["lam"], d["val_auc_bar"], "-s", ms=2.5, lw=1.1,
                 color="#b23a48", label="validation, per bar")
        ax.set_xlabel(r"interpolation weight $\lambda$  (0 = frozen, 1 = retrieval only)")
        ax.set_ylabel("test AUC (bar)", color="#22343f")
        ax2.set_ylabel("validation AUC (bar)", color="#b23a48")
        ax2.spines["top"].set_visible(False)
        fin(fig, "fig_knnlm_lambda_curve.pdf")

    # (3) the price of an immutable memory
    f = out() / "retrieval_growth_curve.csv"
    if f.exists():
        d = pd.read_csv(f).sort_values("n_pseudo_added")
        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        ax.errorbar(d["n_pseudo_added"], d["vs_sf__bar_auc_delta"],
                    yerr=[d["vs_sf__bar_auc_delta"] - d["vs_sf__bar_auc_ci_lo"],
                          d["vs_sf__bar_auc_ci_hi"] - d["vs_sf__bar_auc_delta"]],
                    fmt="o-", ms=3, lw=1.0, color="#22343f", capsize=2)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlabel("pseudo-labelled deployment bars appended to the memory")
        ax.set_ylabel(r"$\Delta$ per-bar AUC vs sealed memory")
        fin(fig, "fig_memory_growth.pdf")

    # (4) matched-backbone neural table
    f = out() / "neural_main.csv"
    if f.exists():
        d = pd.read_csv(f)
        for kind in sorted(d.backbone.unique()):
            g = d[d.backbone == kind].sort_values("auc_bar")
            fig, ax = plt.subplots(figsize=(4.2, 0.22 * len(g) + 1.0))
            yy = np.arange(len(g))
            ax.barh(yy, g["auc_bar"], color="#8fa9bb", height=0.6)
            ax.barh(yy, g["auc_row"], color="none", edgecolor="#22343f",
                    height=0.6, lw=0.7)
            fr = float(g.loc[g.method == "frozen", "auc_bar"].iloc[0])
            ax.axvline(fr, color="#b23a48", lw=0.8, ls="--")
            ax.set_yticks(yy)
            ax.set_yticklabels(g["method"], fontsize=6.5)
            ax.set_xlim(0.45, 0.85)
            ax.set_xlabel("AUC  (filled = per bar, outline = per post); "
                          "dashed = frozen")
            ax.set_title(kind, fontsize=8)
            fin(fig, f"fig_neural_{kind}.pdf")
    return None


# =============================================================================
# protocol + report
# =============================================================================
def write_protocol(args, panel, ev_val: Env, ev_test: Env):
    proto = cpc.protocol_json(
        track="new gradient-free and half-gradient baselines (survey priority order)",
        primary_estimand=("per bar: probabilities mean-pooled to the aligned "
                          "outcome bar, AUC over 525 test bars"),
        secondary_estimand="per post (row level), 1,835 posts",
        selection_rule=("EVERY baseline hyperparameter selected on the VALIDATION "
                        "split (2025-10-01..2026-01-01) at validation PER-BAR AUC; "
                        "the 2026 test panel is touched once. R3-TTT-SF selects "
                        "nothing and was not retuned for this study."),
        environments={
            "retrieval_and_hedge": "CLEAN panel, real GBT slow model",
            "neural": ("CLEAN panel, matched neural backbones refitted here -- a "
                       "NEW environment, never to be differenced against "
                       "results/gradient_tta_baselines"),
        },
        hyperparameter_budgets={
            "r3ttt_sf_published": "72 members, NOTHING selected (uniform average)",
            "knn_vote": "3 spaces x 9 k = 27",
            "knnlm": "3 spaces x 9 k x 3 kernels x 21 lambda x 2 fusion spaces = 3,402",
            "nadaraya_watson": "3 spaces x 10 bandwidths x 21 lambda = 630",
            "prototype": "3 spaces x 6 tau x 21 lambda = 378",
            "t3a_routing": "3 spaces x 6 M x 21 lambda = 378",
            "adanpc_vote": "3 spaces x 6 k x 3 tau x 4 growth margins = 216",
            "adanpc_fused": "the same 216 x 21 lambda = 4,536",
            "r3ttt_growth": "6 confidence margins",
            "hedge_ewa": "10 learning rates x 2 losses = 20",
            "neural": "per-method grids listed in neural_validation_sweep.csv",
        },
        cost_note=("wall clock and backward passes per bar are reported beside AUC "
                   "because zero-backward operation is part of the claim"),
        uncertainty={"estimator": "paired 5-day moving-block bootstrap",
                     "clustered_by": "calendar day", "n_boot": args.n_boot,
                     "seed": SEED,
                     "hac_dm": "Newey-West hand-rolled (statsmodels absent)"},
        blas_threads=1,
        panel_census={
            "n_test_rows": ev_test.bi.n_rows, "n_test_bars": ev_test.bi.n_bars,
            "n_val_rows": ev_val.bi.n_rows, "n_val_bars": ev_val.bi.n_bars,
            "n_val_singleton_bars": int((ev_val.bi.sizes == 1).sum()),
            "n_test_singleton_bars": int((ev_test.bi.sizes == 1).sum()),
            "n_memory_bars_test": len(ev_test.memory),
            "n_memory_bars_val": len(ev_val.memory),
        },
        reference_values={
            "frozen_auc_bar": ev_test.bi.auc_bar(ev_test.frozen),
            "frozen_auc_row": ev_test.bi.auc_row(ev_test.frozen),
            "r3ttt_sf_auc_bar": ev_test.bi.auc_bar(ev_test.sf),
            "r3ttt_sf_auc_row": ev_test.bi.auc_row(ev_test.sf),
        },
        code=str(Path(__file__).resolve()),
    )
    jdump(proto, "protocol.json")
    return proto


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "retrieval", "hedge", "neural", "loqo",
                             "protocol", "figs"],
                    help="which block to run (default: all)")
    ap.add_argument("--out", default=str(ROOT / "results" / "new_baselines"),
                    help="results directory (created; never an existing one)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT,
                    help="bootstrap replicates (default 5000)")
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="neural backbone seeds (default 5)")
    ap.add_argument("--torch-threads", type=int, default=8)
    args = ap.parse_args()

    global _OUT
    _OUT = Path(args.out)
    _OUT.mkdir(parents=True, exist_ok=True)
    log(f"output -> {_OUT}")
    log(f"BLAS threads pinned to {os.environ.get('OMP_NUM_THREADS')}")

    panel, ev_val, ev_test = build_envs()
    args._panel = panel
    args._train = panel[panel.datetime < cpc.TRAIN_END].reset_index(drop=True)
    args._pretest = panel[panel.datetime < cpc.TEST_START].reset_index(drop=True)

    if args.stage in ("all", "protocol"):
        write_protocol(args, panel, ev_val, ev_test)
    if args.stage in ("all", "retrieval"):
        stage_retrieval(args, ev_val, ev_test)
    if args.stage in ("all", "hedge"):
        stage_hedge(args, ev_val, ev_test)
    if args.stage in ("all", "neural"):
        stage_neural(args, ev_val, ev_test)
    if args.stage in ("all", "loqo"):
        stage_loqo(args)
    if args.stage in ("all", "figs"):
        stage_figs(args)
    log("done")


if __name__ == "__main__":
    main()
