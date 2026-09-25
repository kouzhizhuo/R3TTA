#!/usr/bin/env python3
"""Shared machinery for the 2026-09-23 method track (T1).  Reads no test row
by itself; which panel is scored is decided by the calling script.

One `Fold` = one deployment: a fit window, a frozen anchor, an immutable
bar memory matured at the boundary, and a stream scored bar by bar.  Every
arm below is a closed-form function of (frozen anchor, retrieved neighbour
labels, kernel weights) -- zero backward passes, memory frozen at the
boundary, one fast state per outcome bar.

Reference arms are re-derived here from the same primitives and are
provenance-gated against the saved 0921 / 0911 numbers by the callers.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/r3ttt-0923-mpl")

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common_protocol_clean as cpc                   # noqa: E402
import exp_ablation_sensitivity as eas                # noqa: E402
import exp_residual_20260921 as base                  # noqa: E402

OUT = ROOT / "results" / "r3ttt_0923"
OUT.mkdir(parents=True, exist_ok=True)
base.OUT = OUT

N_BOOT = 5000
SCALES = tuple(cpc.SCALES)                 # 8,16,32,64
SPACES = dict(cpc.RETRIEVAL_SPACES)        # timing, market_state, joint
TEMPS = (0.25, 1.0)
LAMS = (2.0, 8.0)
BETAS = (0.3, 0.5, 0.7)
BETA = 0.5
CLIP = 1e-6                                # kNN-LM's clip on the raw vote

TEST_END = pd.Timestamp("2026-04-11")
FORWARD_START = pd.Timestamp("2026-04-16")

# AdaNPC-fused configurations as selected on each backbone's own validation
# anchor in results/knn_neural_0911/metrics.csv (column `selected`).  Carried
# as FIXED comparators here; nothing is reselected.
ADANPC_CFG = {
    "gbt": dict(space="timing", k=16, tau=1.0, margin=0.55, lam=0.4),
    "mlp_bn": dict(space="timing", k=16, tau=4.0, margin=0.9, lam=0.7),
    "ft_trans": dict(space="timing", k=16, tau=4.0, margin=0.9, lam=0.75),
}

auc, brier, logloss = eas.auc, eas.brier, eas.logloss


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def guard(df: pd.DataFrame, *, allow_test: bool) -> None:
    """Hard guard: nothing at or after 2026-04-16 is ever read, and the test
    panel (>= 2026-01-01) is read only by a script that declares it."""
    mx = pd.Timestamp(df["datetime"].max())
    assert mx < FORWARD_START, f"forward window read: {mx}"
    assert mx < TEST_END, f"row at/after TEST_END read: {mx}"
    if not allow_test:
        assert mx < cpc.TEST_START, f"test row read by a test-free script: {mx}"


def load_panel(*, allow_test: bool) -> pd.DataFrame:
    df = cpc.load_event_panel()
    cpc.assert_clean(df, where="event panel")
    df = df[df["datetime"] < TEST_END].reset_index(drop=True)
    if not allow_test:
        df = df[df["datetime"] < cpc.TEST_START].reset_index(drop=True)
    guard(df, allow_test=allow_test)
    log(f"panel {len(df)} rows, max {df['datetime'].max()} "
        f"(allow_test={allow_test})")
    return df


# ===========================================================================
# one deployment
# ===========================================================================
class Fold:
    def __init__(self, fit: pd.DataFrame, stream: pd.DataFrame,
                 boundary: pd.Timestamp, frozen: np.ndarray | None = None,
                 memory=None):
        self.fit = fit.reset_index(drop=True)
        self.stream = stream.reset_index(drop=True)
        self.boundary = boundary
        self.frozen = (cpc.slow_probability(self.fit, self.stream)
                       if frozen is None else np.asarray(frozen, float))
        self.memory = (cpc.build_bar_memory(self.fit, deploy_start=boundary)
                       if memory is None else memory)
        self.groups = cpc.bar_groups(self.stream)
        self.bi = eas.BarIndex(self.stream)
        self.row_bar = self.bi.row_bar
        self.prior = float(self.memory.labels.mean())
        self.ret = {s: base.Retrieved(self.memory, self.stream, self.groups, f)
                    for s, f in SPACES.items()}
        self._x = {}
        self._mom = {}

    def with_anchor(self, frozen: np.ndarray) -> "Fold":
        """Same memory/retrieval, a different frozen anchor (neural backbones)."""
        g = object.__new__(Fold)
        g.__dict__.update(self.__dict__)
        g.frozen = np.asarray(frozen, float)
        return g

    # standardized coordinates (same scaler cpc._route fits: memory only)
    def coords(self, space):
        if space not in self._x:
            feats = list(SPACES[space])
            raw = self.memory.frame[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            sc = StandardScaler().fit(raw)
            X = sc.transform(raw)
            sx = sc.transform(self.stream[feats].replace([np.inf, -np.inf], np.nan)
                              .fillna(0.0))
            Q = np.stack([sx[g].mean(axis=0) for g in self.groups], axis=0)
            self._x[space] = (X, Q)
        return self._x[space]

    def moments(self, space, T):
        """{k: (W, S, W2)} kernel mass, positive mass, sum of squared weights."""
        key = (space, T)
        if key not in self._mom:
            r = self.ret[space]
            w = r.kernel(T)
            out = {}
            for k in SCALES:
                sel = w * r.mask(k)
                out[k] = (sel.sum(1), (sel * r.y).sum(1), (sel ** 2).sum(1))
            self._mom[key] = out
        return self._mom[key]


# ===========================================================================
# primitive local estimates
# ===========================================================================
def shrunk_q(W, S, prior, lam):
    return (S + lam * prior) / (W + lam)


def raw_vote(W, S):
    return S / np.maximum(W, 1e-300)


def local_linear_u(fold: Fold, space, T, lam):
    """Per-k local-linear logit: one Newton step of kernel-weighted logistic
    regression on the routing coordinates at the query (as 0921
    `tie.arm_local_linear`, but returned per k instead of k-averaged)."""
    X, Q = fold.coords(space)
    r = fold.ret[space]
    w = r.kernel(T)
    out = {}
    for k in SCALES:
        sel = w * r.mask(k)
        W, S = sel.sum(1), (sel * r.y).sum(1)
        u0 = cpc.logit(shrunk_q(W, S, fold.prior, lam))
        u = u0.copy()
        for b in range(r.n_query):
            cols = np.flatnonzero(sel[b] > 0)
            if cols.size < 2 * (Q.shape[1] + 1):
                continue
            j = r.idx[b, cols]
            wb = sel[b, cols]
            Z = np.hstack([np.ones((cols.size, 1)), X[j] - Q[b][None, :]])
            p0 = 1.0 / (1.0 + np.exp(-u0[b]))
            s = wb * p0 * (1.0 - p0)
            g = Z.T @ (wb * (r.y[b, cols] - p0))
            H = Z.T @ (Z * s[:, None])
            H[0, 0] += lam * 0.25
            try:
                step = np.linalg.solve(H + 1e-8 * np.eye(H.shape[0]), g)
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(step).all() and abs(step[0]) < 10.0:
                u[b] = u0[b] + step[0]
        out[k] = u
    return out


# ===========================================================================
# AdaNPC (fused), verbatim port of exp_new_baselines.adanpc_bar_scores
# ===========================================================================
def adanpc_fused(fold: Fold, cfg: dict) -> np.ndarray:
    import exp_new_baselines as nb
    spaces = {}
    for s in SPACES:
        X, Q = fold.coords(s)
        spaces[s] = dict(mem_x=X, queries=Q)
    ns = SimpleNamespace(spaces=spaces, memory=fold.memory)
    bar, _ = nb.adanpc_bar_scores(ns, cfg["space"], cfg["k"], cfg["tau"],
                                  cfg["margin"])
    lam = cfg["lam"]
    return (1.0 - lam) * fold.frozen + lam * bar[fold.row_bar]


# ===========================================================================
# arms
# ===========================================================================
def sig(x):
    return cpc.sigmoid(x)


def beta_quadrature(a, b, lf_rows, row_bar, beta=BETA, n=48):
    """E_theta[ sigmoid((1-beta) l_e + beta logit theta) ], theta ~ Beta(a,b)
    per bar, by fixed Gauss-Legendre nodes on (0,1).  Closed-form quadrature,
    no optimisation."""
    x, wq = np.polynomial.legendre.leggauss(n)
    th = 0.5 * (x + 1.0)
    wq = 0.5 * wq
    la, lb = np.log(th)[None, :], np.log1p(-th)[None, :]
    logpdf = (a[:, None] - 1.0) * la + (b[:, None] - 1.0) * lb
    logpdf -= logpdf.max(axis=1, keepdims=True)
    dens = np.exp(logpdf) * wq[None, :]
    dens /= dens.sum(axis=1, keepdims=True)              # (bars, nodes)
    lt = cpc.logit(th)[None, :]
    pr = sig((1.0 - beta) * lf_rows[:, None] + beta * lt)  # (rows, nodes)
    return (pr * dens[row_bar]).sum(axis=1)


def all_arms(fold: Fold, *, which: set[str] | None = None,
             backbone: str = "gbt", spaces=None, temps=TEMPS, lams=LAMS,
             scales=SCALES, beta=BETA) -> dict[str, np.ndarray]:
    """Every reference arm and every 0923 candidate on one fold.

    `which` restricts the computation (None = everything).  The keyword grid
    arguments exist for the ablation / sensitivity scripts; the candidates and
    references are defined at the defaults.
    """
    spaces = list(SPACES) if spaces is None else list(spaces)
    want = (lambda n: True) if which is None else (lambda n: n in which)
    fz, lf, rb, prior = fold.frozen, cpc.logit(fold.frozen), fold.row_bar, fold.prior
    acc: dict[str, list] = {}

    def add(tag, p, w=1.0):
        if want(tag):
            acc.setdefault(tag, []).append((w, p))

    need_ll = any(want(t) for t in ("local_linear", "c03_ll_kmem", "c04_ll_mix",
                                    "c09", "c10", "c11", "c12"))
    for s in spaces:
        w_space = 2.0 if s == "joint" else 1.0
        for T in temps:
            mom = fold.moments(s, T)
            # --- kNN-LM fixed grid (unshrunk, clipped raw vote), k as members
            for k in scales:
                W, S, W2 = mom[k]
                r = np.clip(raw_vote(W, S), CLIP, 1 - CLIP)
                for b in BETAS:
                    add("knnlm72", sig((1 - b) * lf + b * cpc.logit(r)[rb]))
                    add("knnlm72_prob", (1 - b) * fz + b * r[rb])
                add("knnlm24_b05", sig((1 - beta) * lf + beta * cpc.logit(r)[rb]))
                # C6: Beta-binomial posterior with the FROZEN model as prior,
                # pseudo-count k (the anchor is worth one flat neighbourhood)
                add("c06_evid_prob", (S[rb] + k * fz) / (W[rb] + k))
                # round 2 (declared after round 1, combinations of its parts)
                for c in (0.5, 1.0, 2.0):              # C9: anchor strength c*k
                    add("c09_evid_prob_wide", (S[rb] + c * k * fz) / (W[rb] + c * k))
                add("c10_evid_prob_joint2", (S[rb] + k * fz) / (W[rb] + k), w_space)
                neff = W ** 2 / np.maximum(W2, 1e-300)  # C11: Kish count as evidence
                a11 = (neff / (neff + k))[rb]
                add("c11_evid_prob_neff", (1 - a11) * fz + a11 * raw_vote(W, S)[rb])
            if want("c12_evid_prob_lepski"):
                LEPSKI_Z = 2.0
                n_q = len(mom[scales[0]][0])
                kstar = np.full(n_q, scales[0])
                ok = np.ones(n_q, bool)
                hs, sds = [], []
                for k in scales:
                    W, S, W2 = mom[k]
                    h = (S + 0.5) / (W + 1.0)
                    neff = W ** 2 / np.maximum(W2, 1e-300)
                    for hp, sp in zip(hs, sds):
                        ok &= np.abs(h - hp) <= LEPSKI_Z * sp
                    kstar = np.where(ok, k, kstar)
                    hs.append(h)
                    sds.append(np.sqrt(h * (1 - h) / np.maximum(neff, 1e-12)))
                Wk = np.choose(np.searchsorted(scales, kstar),
                               [mom[k][0] for k in scales])
                Sk = np.choose(np.searchsorted(scales, kstar),
                               [mom[k][1] for k in scales])
                add("c12_evid_prob_lepski",
                    (Sk[rb] + kstar[rb] * fz) / (Wk[rb] + kstar[rb]))
            lam_list = list(lams) + ([0.5] if want("c02_wide72") else [])
            for lam in lam_list:
                core = lam in lams
                us, qs, agrees = [], [], []
                for k in scales:
                    W, S, W2 = mom[k]
                    q = shrunk_q(W, S, prior, lam)
                    u = cpc.logit(q)
                    us.append(u)
                    qs.append(q)
                    share = np.where(W > 0, S / np.maximum(W, 1e-12), prior)
                    agrees.append(np.abs(share - 0.5) * 2.0)
                    pk = sig((1 - beta) * lf + beta * u[rb])
                    add("c02_wide72", pk)
                    if not core:
                        continue
                    add("c01_kmem", pk)
                    # C5: evidence-discounted logit weight, beta_b = neff/(neff+k)
                    neff = W ** 2 / np.maximum(W2, 1e-300)
                    bb = (neff / (neff + k))[rb]
                    add("c05_evid_logit", sig((1 - bb) * lf + bb * u[rb]))
                    # C8: posterior-predictive integration over theta
                    if want("c08_postpred"):
                        a = S + lam * prior
                        bpar = (W - S) + lam * (1 - prior)
                        add("c08_postpred", beta_quadrature(a, bpar, lf, rb, beta))
                if not core:
                    continue
                u_bar = np.stack(us, 0).mean(0)
                p_respec = sig((1 - beta) * lf + beta * u_bar[rb])
                add("respec", p_respec)
                add("c07_joint2", p_respec, w_space)
                add("knnlm", (1 - beta) * fz + beta * np.stack(qs, 0).mean(0)[rb])
                cons = np.mean(np.stack(agrees, 0), 0)
                rho = cons / (1.0 + np.stack(us, 0).std(0))
                for b in BETAS:
                    for g in (np.ones_like(rho), rho):
                        e = b * g[rb]
                        add("shipped72", sig((1 - e) * lf + e * u_bar[rb]))
                if need_ll:
                    ull = local_linear_u(fold, s, T, lam)
                    ull_bar = np.stack([ull[k] for k in scales], 0).mean(0)
                    p_ll = sig((1 - beta) * lf + beta * ull_bar[rb])
                    add("local_linear", p_ll)
                    add("c04_ll_mix", p_ll)
                    add("c04_ll_mix", p_respec)
                    for k in scales:
                        add("c03_ll_kmem", sig((1 - beta) * lf + beta * ull[k][rb]))
    out = {"frozen": fz} if want("frozen") or which is None else {}
    for tag, lst in acc.items():
        wts = np.array([w for w, _ in lst])
        out[tag] = np.tensordot(wts / wts.sum(), np.stack([p for _, p in lst], 0), 1)
    if want("adanpc_fused"):
        out["adanpc_fused"] = adanpc_fused(fold, ADANPC_CFG[backbone])
    out["_n_members"] = {t: len(v) for t, v in acc.items()}
    return out


# ===========================================================================
# scoring
# ===========================================================================
def score_table(bi, preds: dict, refs=("frozen", "respec", "knnlm72",
                                       "adanpc_fused", "knnlm")):
    recs = []
    for name, p in preds.items():
        if name.startswith("_"):
            continue
        rec = dict(arm=name, **bi.scores(p))
        for r in refs:
            if r == name or r not in preds:
                continue
            rec.update(eas.contrast(bi, p, preds[r], n_boot=N_BOOT,
                                    prefix=f"vs_{r}_"))
        recs.append(rec)
    return pd.DataFrame(recs).sort_values("auc_bar", ascending=False)


def sign_test(wins: int, n: int) -> float:
    from scipy import stats
    return float(stats.binomtest(wins, n, 0.5).pvalue) if n else float("nan")


# ===========================================================================
# round 4 (coordinator steer, 2026-09-23): custom retrieval + self-calibration
# ===========================================================================
class RetrievedXY:
    """Retrieved-compatible neighbours from explicit coordinates, with the SAME
    distance, 256-pool argpartition and stable sort as cpc._route.  `purge`
    (n_query, n_memory) bool marks forbidden memory items (set to +inf)."""

    def __init__(self, X, Q, labels, purge=None, max_k=cpc.MAX_K):
        d = (np.einsum("ij,ij->i", Q, Q)[:, None] - 2.0 * (Q @ X.T)
             + np.einsum("ij,ij->i", X, X)[None, :]) / float(X.shape[1])
        np.maximum(d, 0.0, out=d)
        if purge is not None:
            d = np.where(purge, np.inf, d)
        keep = min(max_k, X.shape[0])
        local = np.argpartition(d, keep - 1, axis=1)[:, :keep]
        ld = np.take_along_axis(d, local, axis=1)
        order = np.argsort(ld, axis=1, kind="stable")
        self.idx = np.take_along_axis(local, order, axis=1)
        self.dist = np.take_along_axis(ld, order, axis=1)
        self.y = labels[self.idx].astype(np.float64)
        self.finite = np.isfinite(self.dist)
        self.n_query, self.keep = self.y.shape
        self.position = np.arange(self.keep)[None, :]
        d0 = self.dist[:, 0].astype(np.float64)
        self.d_min = np.where(np.isfinite(d0), d0, 0.0)[:, None]

    kernel = base.Retrieved.kernel
    mask = base.Retrieved.mask


def feature_weights(fold: Fold, space):
    """Diagonal metric from memory labels only: w_j = |2 AUC_j - 1| of the
    standardized coordinate against the matured label, rescaled so that
    sum w_j^2 = d (the isotropic scale the temperature grid assumes)."""
    X, _ = fold.coords(space)
    y = fold.memory.labels
    w = np.array([abs(2.0 * auc(y, X[:, j]) - 1.0) for j in range(X.shape[1])])
    w = np.where(np.isfinite(w), w, 0.0) + 1e-6
    return w * np.sqrt(X.shape[1] / (w ** 2).sum())


def samworth_weights(k, d):
    """Samworth (2012) optimal rank weights for i <= k (nearest = 1)."""
    i = np.arange(1, k + 1, dtype=float)
    a = 1.0 + 2.0 / d
    w = 1.0 + d / 2.0 - d / (2.0 * k ** (2.0 / d)) * (i ** a - (i - 1) ** a)
    w = np.maximum(w, 0.0)
    return w / w[0]


def evid_generic(fz, rb, mom_fn, spaces, temps, scales, c=1.0,
                 space_w=(("joint", 2.0),)):
    sw = dict(space_w)
    tot, wsum = np.zeros_like(fz), 0.0
    for s in spaces:
        ws = sw.get(s, 1.0)
        for T in temps:
            for k in scales:
                W, S = mom_fn(s, T, k)
                tot += ws * (S[rb] + c * k * fz) / (W[rb] + c * k)
                wsum += ws
    return tot / wsum


SELFCAL_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
PURGE_DAYS = 1


def memory_loo_c(fold: Fold, mem_anchor: np.ndarray) -> tuple[float, dict]:
    """Test-free self-calibration of the anchor strength c: every memory bar is
    a query against the rest of the memory, neighbours within +-PURGE_DAYS
    calendar days purged (a bar and its adjacent bars share regime and label),
    anchored on the OUT-OF-FOLD frozen probability of that memory bar; pick c
    in SELFCAL_GRID maximising memory bar AUC (ties -> c closest to 1)."""
    mem = fold.memory.frame
    y = fold.memory.labels
    day = pd.to_datetime(mem["datetime"]).dt.normalize().to_numpy()
    dd = np.abs((day[:, None] - day[None, :]) / np.timedelta64(1, "D"))
    purge = dd <= PURGE_DAYS
    rets = {}
    for s in SPACES:
        X, _ = fold.coords(s)
        rets[s] = RetrievedXY(X, X, y, purge=purge)

    def mom_fn(s, T, k):
        r = rets[s]
        sel = r.kernel(T) * r.mask(k)
        return sel.sum(1), (sel * r.y).sum(1)

    rb = np.arange(len(y))
    scores = {}
    for c in SELFCAL_GRID:
        p = evid_generic(mem_anchor, rb, mom_fn, list(SPACES), TEMPS, SCALES, c=c)
        scores[c] = auc(y, p)
    best = max(scores.values())
    cands = [c for c, v in scores.items() if v >= best - 1e-12]
    c_star = min(cands, key=lambda c: abs(np.log(c)))
    return float(c_star), {str(k): float(v) for k, v in scores.items()}


def round4_arms(fold: Fold, *, which, mem_anchor=None, base_preds=None):
    """c13..c16.  `base_preds` must hold knnlm72, local_linear, c02_wide72 and
    c10_evid_prob_joint2 when c13 is wanted; `mem_anchor` (OOF frozen prob per
    memory bar) when c14 is wanted."""
    fz, rb = fold.frozen, fold.row_bar
    out, info = {}, {}

    def mom_default(s, T, k):
        W, S, _ = fold.moments(s, T)[k]
        return W, S

    if "c13_superset" in which:
        fam = ("knnlm72", "local_linear", "c02_wide72", "c10_evid_prob_joint2")
        out["c13_superset"] = np.mean([base_preds[f] for f in fam], axis=0)
    if "c14_selfcal" in which:
        c_star, sc = memory_loo_c(fold, mem_anchor)
        info["c14_c_star"], info["c14_loo_auc"] = c_star, sc
        out["c14_selfcal"] = evid_generic(fz, rb, mom_default, list(SPACES), TEMPS,
                                          SCALES, c=c_star)
    if "c15_diagmetric" in which:
        rets = {}
        for s in SPACES:
            X, Q = fold.coords(s)
            w = feature_weights(fold, s)
            info[f"c15_w_{s}"] = w.tolist()
            rets[s] = RetrievedXY(X * w, Q * w, fold.memory.labels)

        def mom_diag(s, T, k):
            r = rets[s]
            sel = r.kernel(T) * r.mask(k)
            return sel.sum(1), (sel * r.y).sum(1)
        out["c15_diagmetric"] = evid_generic(fz, rb, mom_diag, list(SPACES), TEMPS,
                                             SCALES)
    if "c16_samworth" in which:
        def mom_sam(s, T, k):
            r = fold.ret[s]
            w = np.zeros(r.keep)
            w[:k] = samworth_weights(k, len(SPACES[s]))
            sel = w[None, :] * r.finite
            return sel.sum(1), (sel * r.y).sum(1)
        out["c16_samworth"] = evid_generic(fz, rb, mom_sam, list(SPACES), (None,),
                                           SCALES)
    return out, info
