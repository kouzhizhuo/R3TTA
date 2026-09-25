#!/usr/bin/env python3
"""Ablations and sensitivity of the 0923 final method.  2026-09-23 (T1).

Reporting only: nothing here selects anything.  The final arm and its
definition come from results/r3ttt_0923/protocol_0923.json.  `--panel test`
refuses to run unless that protocol exists and verifies (same check as
exp_final_test_0923.py).

The final method is either in the respec family (`respec_fam`, used when
final_spec.family == "respec") or in the anchor-centred Beta-binomial family

    p_e = (1/|M|) sum_{m=(space,T,k,c)} w_space * (S_{b,k} + c k p_e0) / (W_{b,k} + c k)

with S, W the kernel-weighted positive and total neighbour mass (eq. m_weights),
p_e0 the frozen probability, c k the anchor's pseudo-count.  `evid` below
computes any member of that family; each ablation removes one component.

    python code/exp_ablation_0923.py --panel wf|val|test
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
FOLD_START = pd.Timestamp("2025-04-01")
EXT_SCALES = (8, 16, 32, 64, 128, 256)


def moments(fold, space, T, k):
    r = fold.ret[space]
    if T == "flat":
        w = np.where(r.finite, 1.0, 0.0)
    else:
        w = r.kernel(T)
    sel = w * r.mask(k)
    return sel.sum(1), (sel * r.y).sum(1), (sel ** 2).sum(1)


def evid(fold, *, spaces=tuple(lib.SPACES), temps=lib.TEMPS, scales=lib.SCALES,
         cmult=(1.0,), space_w=None, weight="mass", center="anchor",
         fusion="prob"):
    """One member family of the anchor-centred posterior; see module doc.

    weight : "mass" alpha=W/(W+ck) | "fixed" alpha=1/2 | "neff" Kish count
    center : "anchor" prior mean = frozen p_e | "base" prior mean = memory pi0
             (then fused with the anchor at fixed 1/2 in probability)
    fusion : "prob" | "logit" (alpha applied to logits of the raw vote)
    """
    fz, rb, pi0 = fold.frozen, fold.row_bar, fold.prior
    lf = cpc.logit(fz)
    tot, wsum = np.zeros_like(fz), 0.0
    for s in spaces:
        ws = 1.0 if space_w is None else space_w.get(s, 1.0)
        for T in temps:
            for k in scales:
                W, S, W2 = moments(fold, s, T, k)
                for c in cmult:
                    lam = c * k
                    if weight == "mass":
                        a = W / (W + lam)
                    elif weight == "neff":
                        ne = W ** 2 / np.maximum(W2, 1e-300)
                        a = ne / (ne + lam)
                    else:
                        a = np.full_like(W, 0.5)
                    h = lib.raw_vote(W, S)
                    if center == "base":
                        q = (S + lam * pi0) / (W + lam)
                        p = 0.5 * fz + 0.5 * q[rb]
                    elif fusion == "logit":
                        hc = np.clip(h, lib.CLIP, 1 - lib.CLIP)
                        p = lib.sig((1 - a[rb]) * lf + a[rb] * cpc.logit(hc)[rb])
                    else:
                        p = (1 - a[rb]) * fz + a[rb] * h[rb]
                    tot += ws * p
                    wsum += ws
    return tot / wsum


def respec_fam(fold, *, spaces=tuple(lib.SPACES), temps=lib.TEMPS, lams=lib.LAMS,
               scales=lib.SCALES, beta=0.5, space_w=None, gate=False,
               shrink=True, fusion="logit", **_):
    """The respec family (eq. m_faststate / m_scalemean / m_fusion / m_sf):
    member m=(space,T,lam) -> sigmoid((1-beta*g) l_e + beta*g * mean_k u_k),
    members averaged in probability with space weights."""
    fz, rb, pi0 = fold.frozen, fold.row_bar, fold.prior
    lf = cpc.logit(fz)
    sw = dict(space_w or {})
    tot, wsum = np.zeros_like(fz), 0.0
    for s in spaces:
        ws = sw.get(s, 1.0)
        for T in temps:
            mom = {k: moments(fold, s, T, k) for k in scales}
            for lam in lams:
                us, qs, agr = [], [], []
                for k in scales:
                    W, S, _ = mom[k]
                    if shrink:
                        q = (S + lam * pi0) / (W + lam)
                    else:
                        q = np.clip(lib.raw_vote(W, S), lib.CLIP, 1 - lib.CLIP)
                    qs.append(q)
                    us.append(cpc.logit(q))
                    share = np.where(W > 0, S / np.maximum(W, 1e-12), pi0)
                    agr.append(np.abs(share - 0.5) * 2.0)
                u = np.stack(us, 0).mean(0)
                g = (np.mean(np.stack(agr, 0), 0) / (1 + np.stack(us, 0).std(0))
                     if gate else np.ones_like(u))
                e = beta * g[rb]
                if fusion == "prob":
                    p = (1 - e) * fz + e * np.stack(qs, 0).mean(0)[rb]
                else:
                    p = lib.sig((1 - e) * lf + e * u[rb])
                tot += ws * p
                wsum += ws
    return tot / wsum


def configs_respec(final_spec: dict):
    base = dict(final_spec)
    out = [("final", "final", base)]

    def v(**kw):
        d = dict(base)
        d.update(kw)
        return d
    out += [
        ("- joint emphasis (uniform spaces = respec)", "ablation", v(space_w=None)),
        ("+ reliability gate (published rho)", "ablation", v(gate=True)),
        ("- prior shrinkage (raw clipped vote)", "ablation", v(shrink=False)),
        ("- logit fusion (probability mixture)", "ablation", v(fusion="prob")),
        ("- multi-scale (k=32 only)", "ablation", v(scales=(32,))),
        ("- space average (timing only)", "ablation", v(spaces=("timing",))),
        ("- space average (market_state only)", "ablation", v(spaces=("market_state",))),
        ("- space average (joint only)", "ablation", v(spaces=("joint",))),
        ("- temperature average (T=0.25 only)", "ablation", v(temps=(0.25,))),
        ("- temperature average (T=1 only)", "ablation", v(temps=(1.0,))),
        ("- prior-strength average (lam=2 only)", "ablation", v(lams=(2.0,))),
        ("- prior-strength average (lam=8 only)", "ablation", v(lams=(8.0,))),
    ]
    for b in (0.1, 0.3, 0.5, 0.7, 0.9):
        out.append((f"beta={b}", "sensitivity", v(beta=b)))
    for wj in (1.0, 1.5, 2.0, 3.0, 4.0):
        out.append((f"joint_weight={wj}", "sensitivity", v(space_w={"joint": wj})))
    for lam in ((0.5,), (1.0,), (2.0,), (4.0,), (8.0,), (16.0,), (2.0, 8.0)):
        out.append((f"lam={lam}", "sensitivity", v(lams=lam)))
    for T in ((0.1,), (0.25,), (0.5,), (1.0,), (2.0,), (0.25, 1.0)):
        out.append((f"T={T}", "sensitivity", v(temps=T)))
    for ks in ((8,), (16,), (32,), (64,), (128,), (256,), (8, 16, 32, 64),
               EXT_SCALES):
        out.append((f"k={ks}", "sensitivity", v(scales=ks)))
    return out


FAMILY = {"evid": None, "respec": None}


def family_of(spec):
    return "respec" if spec.get("family") == "respec" else "evid"


def build(fold, spec):
    kw = {k: v for k, v in spec.items() if k != "family"}
    return respec_fam(fold, **kw) if family_of(spec) == "respec" else evid(fold, **kw)


def configs(final_spec: dict):
    if family_of(final_spec) == "respec":
        return configs_respec(final_spec)
    return configs_evid(final_spec)


def configs_evid(final_spec: dict):
    """(name, kind, kwargs).  kind in {final, ablation, sensitivity}."""
    base = dict(final_spec)
    out = [("final", "final", base)]

    def v(**kw):
        d = dict(base)
        d.update(kw)
        return d
    out += [
        ("- distance-aware weight (alpha=1/2)", "ablation", v(weight="fixed")),
        ("- anchor-centred prior (base-rate prior, fixed 1/2 fusion)", "ablation",
         v(center="base")),
        ("- probability-space pooling (logit pooling)", "ablation", v(fusion="logit")),
        ("- kernel (flat weights)", "ablation", v(temps=("flat",))),
        ("- multi-scale (k=32 only)", "ablation", v(scales=(32,))),
        ("- space average (timing only)", "ablation", v(spaces=("timing",), space_w=None)),
        ("- space average (market_state only)", "ablation",
         v(spaces=("market_state",), space_w=None)),
        ("- space average (joint only)", "ablation", v(spaces=("joint",), space_w=None)),
        ("- temperature average (T=0.25 only)", "ablation", v(temps=(0.25,))),
        ("- temperature average (T=1 only)", "ablation", v(temps=(1.0,))),
        ("Kish n_eff instead of W", "ablation", v(weight="neff")),
    ]
    if base.get("space_w"):
        out.append(("- joint emphasis (uniform spaces)", "ablation", v(space_w=None)))
    for c in (0.25, 0.5, 1.0, 2.0, 4.0):
        out.append((f"c={c}", "sensitivity", v(cmult=(c,))))
    for T in ((0.1,), (0.25,), (0.5,), (1.0,), (2.0,), (0.25, 1.0)):
        out.append((f"T={T}", "sensitivity", v(temps=T)))
    for ks in ((8,), (16,), (32,), (64,), (128,), (256,), (8, 16, 32, 64),
               EXT_SCALES):
        out.append((f"k={ks}", "sensitivity", v(scales=ks)))
    return out


def spec_from_protocol(test: bool):
    p = lib.OUT / "protocol_0923.json"
    if test:
        import exp_final_test_0923 as ft
        d = ft.load_protocol()
    else:
        d = json.loads(p.read_text())
    s = dict(d["final_spec"])
    for k in ("spaces", "temps", "scales", "cmult", "lams"):
        if k in s:
            s[k] = tuple(s[k])
    return d["final_arm"], s


def score(bi, preds, refs):
    t = lib.score_table(bi, preds, refs=refs)
    return t


def run(panel_name, spec):
    cfgs = configs(spec)
    if panel_name == "wf":
        panel = lib.load_panel(allow_test=False)
        edges = pd.date_range(FOLD_START, cpc.TEST_START, freq="MS")
        pooled, ys, ts = {}, [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            fold = lib.Fold(panel[panel.datetime < lo],
                            panel[(panel.datetime >= lo) & (panel.datetime < hi)], lo)
            P = lib.all_arms(fold, which={"frozen", "respec", "knnlm72"})
            P.pop("_n_members")
            for n, _, kw in cfgs:
                P[n] = build(fold, kw)
            for n, p in P.items():
                pooled.setdefault(n, []).append(p)
            ys.append(fold.bi.y_row)
            ts.append(fold.stream.datetime.to_numpy())
            log(f"fold {lo.date()} done")
        bi = eas.BarIndex(pd.DataFrame({"datetime": np.concatenate(ts),
                                        "target_hi_vol": np.concatenate(ys)}))
        P = {n: np.concatenate(v) for n, v in pooled.items()}
    else:
        test = panel_name == "test"
        panel = lib.load_panel(allow_test=test)
        lo = cpc.TEST_START if test else cpc.TRAIN_END
        hi = lib.TEST_END if test else cpc.TEST_START
        fold = lib.Fold(panel[panel.datetime < lo],
                        panel[(panel.datetime >= lo) & (panel.datetime < hi)], lo)
        P = lib.all_arms(fold, which={"frozen", "respec", "knnlm72"})
        P.pop("_n_members")
        for n, _, kw in cfgs:
            P[n] = build(fold, kw)
        bi = fold.bi
    t = score(bi, P, refs=("final", "frozen", "respec", "knnlm72"))
    kind = {n: k for n, k, _ in cfgs}
    t.insert(1, "kind", t["arm"].map(kind).fillna("reference"))
    t = t.rename(columns={"arm": "configuration"})
    t.to_csv(lib.OUT / f"ablation_sensitivity_{panel_name}.csv", index=False)
    # T5 schemas
    ab = t[t.kind.isin(["final", "ablation", "reference"])]
    ab.to_csv(lib.OUT / f"ablation_{panel_name}.csv", index=False)
    se = t[t.kind.isin(["final", "sensitivity"])].copy()
    se.to_csv(lib.OUT / f"sensitivity_{panel_name}.csv", index=False)
    log("\n" + t[["configuration", "kind", "auc_bar", "vs_final_bar_delta",
                  "vs_final_bar_p_le_zero", "vs_frozen_bar_delta",
                  "vs_knnlm72_bar_delta"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, choices=["wf", "val", "test"])
    a = ap.parse_args()
    final, spec = spec_from_protocol(a.panel == "test")
    log(f"final={final} spec={spec}")
    run(a.panel, spec)


if __name__ == "__main__":
    main()
