#!/usr/bin/env python3
"""CLEAN shared protocol module for the R3-TTT ICLR submission.

This is ``common_protocol.py`` with the look-ahead-contaminated engagement
snapshot removed, and NOTHING else changed.  ``common_protocol.py`` is left in
place, untouched, as historical evidence for the 14 auxiliary tracks that were
computed on the contaminated panel.

Why this module exists
----------------------
A provenance audit (``results/engagement_audit/``) established that

    log_reblogs_count, log_favourites_count, log_replies_count

are a SINGLE CRAWL-INSTANT SNAPSHOT taken 2026-04-10 -- months after the test
period opens on 2026-01-01.  A post published in January 2026 carries the
engagement it had accumulated by April 2026.  These are look-ahead features.
They were struck from the paper's prose but continued to ship in
``common_protocol.TIMING_ENGAGEMENT``, hence in ``MODEL_FEATURES`` and in two of
the three ``RETRIEVAL_SPACES``.

The leaked column is empirically INERT (engagement-only test AUC 0.5011) and was
DILUTING the model.  Removing it lowers the absolute level and RAISES the
adaptation effect.  On the clean panel the headline is

    R3-TTT-SF 0.8276 vs frozen seed-average 0.8138,
    delta +0.0138, 95% CI [0.0053, 0.0202], P(delta<=0) = 0.0002

against +0.0098 [0.0003, 0.0162], P = 0.021 on the contaminated panel.

    The clean panel is a NEW ENVIRONMENT.  Its point estimates must NEVER be
    differenced against contaminated-panel point estimates.

What changed, exactly
---------------------
1. ``TIMING_ENGAGEMENT`` -> ``TIMING``: the three engagement columns are gone.
   The timing routing space drops from 8 dimensions to 5 and now contains
   exactly ``hour_sin, hour_cos, dow_sin, dow_cos, is_market_hours``.
   ``MODEL_FEATURES`` drops 12 -> 9.  ``RETRIEVAL_SPACES`` becomes
   ``timing`` (5) / ``market_state`` (4) / ``joint`` (9).
2. The derived per-bar maxima ``log_reblogs_max``, ``log_favourites_max``,
   ``log_replies_max`` are removed from ``_event_features`` and from the asset
   panel's ``base_no_content`` feature set (65 -> 62 columns), which propagates
   to all four asset feature sets.
3. The raw ``reblogs_count`` / ``favourites_count`` / ``replies_count`` columns
   and their ``log_`` transforms are no longer read out of ``posts.csv`` at all,
   so a contaminated column cannot re-enter a downstream track by accident.
   ``assert_clean(frame)`` will raise if one ever does.
4. The panel cache moves to ``data/cache_clean/`` so that the contaminated
   caches in ``data/cache/`` are never overwritten.
5. Added, purely as convenience wrappers over unchanged internals:
   ``CONTAMINATED_FEATURES``, ``assert_clean``, ``headline_event_run``.

A CONTROL THAT FIRES -- read before using the timing space
----------------------------------------------------------
Removing engagement makes the timing routing space DEGENERATE.  The three
engagement columns were the only near-continuous coordinates in it; what remains
is four cyclic hour/day-of-week terms plus a binary, all constant within an
aligned bar.  Measured on the clean panel:

  * 1,973 of the 2,249 memory bars are EXACT DUPLICATES in the 5-d timing space
    (only 276 distinct rows);
  * 83.8% of test query bars have distance ties straddling the k=64 boundary,
    with ~13.5 extra tied rows at that boundary;
  * the 256-candidate pool holds, on average, 28.3 DISTINCT distance values.

So in the timing space "the k nearest memory items" is not well defined: which
tied bar enters the top-k is decided by sort order, not by the metric.  The
market_state (4-d) and joint (9-d) spaces are unaffected -- zero duplicate rows,
256 distinct distances.

This module therefore FIXES the tie-break rule (``TIEBREAK_RULE`` below) so the
headline is deterministic, and quantifies what the rule is worth: over 24 random
permutations of the memory row order the selection-free test AUC spans
0.82744-0.82882 (sd 0.00034).  The engagement audit's 0.82758 and this module's
canonical 0.82827 both sit inside that band, which is the whole of the 0.0007
difference between them.  The DELTA is not at risk -- the frozen baseline is
tie-free at 0.81381 and the gap stays in [+0.0136, +0.0150] across every
permutation -- but the third decimal of the clean SF LEVEL is indeterminate, and
two clean-panel configurations that differ by less than ~0.0014 in the timing
space have not been shown to differ at all.

Everything else is a byte-for-byte copy: the 2025-10-01 / 2026-01-01 split, the
17.1490 bps training-median threshold, forward alignment to the next tradable
15-minute SPY bar, one-hour label maturity, bar-deduplicated immutable
pre-deployment memory, the 72-member selection-free grid, the k in {8,16,32,64}
inner scale ensemble, plain top-k retrieval (``BALANCED_DEFAULT = False``), the
5 slow-model seeds, and the paired 5-day moving-block bootstrap clustered by
calendar day.  The generator that produced this file and the enumerated edit
list live in ``results/_clean_stage/``.

Provenance of the untouched parts
---------------------------------
The panel construction is lifted verbatim (modulo data paths) from

  code/paper_code/run_leakage_safe_primary.py      -> build_panel()
  code/handover_code/run_exposure_tunnel_experiment.py -> prepare_panel()

The R3-TTT fast state mirrors the published implementation, which survives in

  code/paper_code/run_ttt_strong_baselines.py      -> representative_memory_ttt()
  code/handover_code/run_asset_level_r3ttt.py      -> fast_states()  ("mirrors v4")

The archived v4 ensemble uses PLAIN top-k retrieval (``balanced=False``), NOT the
class-balanced k/2-positive / k/2-negative retrieval of the earlier
``representative_memory_ttt``.  ``balanced=True`` remains available as an
ablation.  See README_PROTOCOL_CLEAN.md.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache_clean"          # NOT data/cache -- never overwrite the
                                      # contaminated historical caches.
POSTS_CSV = DATA / "posts.csv"
MARKET_CSV = DATA / "market_15min.csv"
CACHE.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------- protocol
TRAIN_END = pd.Timestamp("2025-10-01")
TEST_START = pd.Timestamp("2026-01-01")
LABEL_DELAY = pd.Timedelta(hours=1)
POST_START = pd.Timestamp("2024-11-06", tz="UTC")
RANDOM_SEED = 42
EPS = 1e-6

# ------------------------------------------------------------- feature blocks
# The crawl-instant engagement snapshot.  Present in NO panel, NO model, NO
# routing space built by this module.  Kept only so that assert_clean() and the
# audit trail can name them.
CONTAMINATED_FEATURES = [
    "log_reblogs_count",
    "log_favourites_count",
    "log_replies_count",
]
CONTAMINATED_AGGREGATES = [
    "log_reblogs_max",
    "log_favourites_max",
    "log_replies_max",
]
CONTAMINATED_RAW = ["reblogs_count", "favourites_count", "replies_count"]
CONTAMINATED_ALL = CONTAMINATED_FEATURES + CONTAMINATED_AGGREGATES + CONTAMINATED_RAW

# 5 dimensions, was 8.  These are the ONLY timing features.
TIMING = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_market_hours",
]
# Backward-compatible alias for call sites that spell the old name.  It no
# longer contains any engagement column; the name is retained only so that
# ``routing_features=cp.TIMING_ENGAGEMENT`` keeps working.
TIMING_ENGAGEMENT = TIMING
MARKET_STATE = [
    "lagged_vol_4bar",
    "lagged_vol_16bar",
    "lagged_ret_4bar",
    "lagged_range_16bar",
]
MODEL_FEATURES = TIMING + MARKET_STATE          # 9, was 12

# The three pre-specified routing spaces of the selection-free variant.
# Key renamed "timing_engagement" -> "timing": the space no longer contains
# engagement, and a KeyError on the old name is the intended, loud signal that a
# downstream track needs review.
RETRIEVAL_SPACES: dict[str, list[str]] = {
    "timing": TIMING,
    "market_state": MARKET_STATE,
    "joint": MODEL_FEATURES,
}


def assert_clean(frame, *, where: str = "frame") -> None:
    """Raise if any contaminated column is present in a frame or column list."""
    columns = set(getattr(frame, "columns", frame))
    bad = sorted(columns & set(CONTAMINATED_ALL))
    if bad:
        raise AssertionError(
            f"{where}: look-ahead engagement columns present: {bad}. "
            "These are a 2026-04-10 crawl snapshot and must not reach any model, "
            "router or panel built by common_protocol_clean."
        )

# Pre-specified adapter grid.  Nothing below is ever selected on validation for
# the headline (selection-free) variant.
SCALES = (8, 16, 32, 64)            # k, averaged INSIDE every member
BLENDS = (0.3, 0.5, 0.7)            # beta
TEMPERATURES = (0.25, 1.0)          # T
PRIOR_STRENGTHS = (2.0, 8.0)        # lambda
GATES = (False, True)               # trust gate on / off
MAX_K = 256                         # retrieved candidate pool
# Retrieval balance of the HEADLINE variant. False = plain top-k (what the
# archived v4 artifact actually does, verified to 7 decimals). True = the
# class-balanced k/2-positive / k/2-negative variant, kept as an ablation.
BALANCED_DEFAULT = False
# Tie-breaking.  With engagement removed the 5-d timing space is heavily
# degenerate (1,973 of 2,249 memory bars are exact duplicates; 83.8% of query
# bars have ties straddling k=64), so retrieval MUST pin a tie-break rule or the
# headline is not reproducible.  The rule: memory rows stay in the order
# ``build_bar_memory`` produces them -- ascending aligned-bar time -- and the
# distance sort is STABLE, so among equidistant memory bars the EARLIEST bar is
# retrieved first.  This is arbitrary but pre-specified and deterministic.
# Worth ~0.0014 of SF test AUC in the timing space; see the module docstring.
TIEBREAK_RULE = "stable sort on memory rows ordered by ascending bar time"
# 3 spaces x 3 beta x 2 T x 2 lambda x 2 gate = 72 members.
N_ENSEMBLE_MEMBERS = (
    len(RETRIEVAL_SPACES) * len(BLENDS) * len(TEMPERATURES)
    * len(PRIOR_STRENGTHS) * len(GATES)
)

# Frozen slow model.
SLOW_TREES = 200
SLOW_DEPTH = 4
SLOW_SEEDS = (0, 1, 2, 3, 4)

# Asset panel slow model (HistGBT), inherited from the exposure-tunnel study.
ASSET_SLOW_PARAMS = dict(
    learning_rate=0.06,
    max_iter=160,
    max_leaf_nodes=15,
    min_samples_leaf=40,
    l2_regularization=1.0,
)
ASSET_SLOW_SEEDS = (0, 1, 2)

HORIZON_BARS_DEFAULT = 4
BAR_MINUTES = 15


# ------------------------------------------------------------------- numerics
def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def logit(probability):
    value = np.clip(probability, EPS, 1.0 - EPS)
    return np.log(value / (1.0 - value))


def fast_binary_auc(y, score) -> float:
    """Exact tie-aware binary AUC; much cheaper than sklearn inside bootstraps."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")
    y_sorted = y[order]
    score_sorted = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(score_sorted) != 0.0) + 1]
    ends = np.r_[starts[1:], len(y_sorted)]
    positives = np.add.reduceat(y_sorted, starts).astype(float)
    negatives = (ends - starts).astype(float) - positives
    n_positive = float(positives.sum())
    n_negative = float(negatives.sum())
    if n_positive == 0.0 or n_negative == 0.0:
        return float("nan")
    negatives_before = np.cumsum(negatives) - negatives
    concordant = np.sum(positives * negatives_before + 0.5 * positives * negatives)
    return float(concordant / (n_positive * n_negative))


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


# ------------------------------------------------------------ post-side panel
TOPIC_RULES = {
    "trade_tariff": [r"tariff", r"\btrade\b", r"duty", r"import", r"export"],
    "china": [r"\bchina\b", r"chinese", r"beijing", r"xi"],
    "macro_fed": [r"\bfed\b", r"powell", r"inflation", r"interest rate", r"rates"],
    "energy": [r"\boil\b", r"\bgas\b", r"energy", r"drill"],
    "crypto": [r"crypto", r"bitcoin", r"coinbase", r"btc"],
    "defense": [r"defense", r"military", r"missile", r"war", r"pentagon"],
    "immigration": [r"border", r"immigration", r"migrant"],
    "tax": [r"tax", r"corporate tax", r"tax cut"],
}
TOPICS = tuple(TOPIC_RULES)


def _match_topics(text: str) -> list[str]:
    text = text or ""
    return [
        topic
        for topic, patterns in TOPIC_RULES.items()
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
    ]


def build_post_panel() -> pd.DataFrame:
    """Post-level panel aligned FORWARD to the next tradable 15-minute SPY bar.

    Verbatim port of ``run_leakage_safe_primary.build_panel``.
    """
    posts = pd.read_csv(
        POSTS_CSV,
        # The engagement counts are deliberately NOT read.  They are a single
        # 2026-04-10 crawl snapshot and are look-ahead for every test row.
        usecols=[
            "post_id",
            "created_at_utc",
            "text",
        ],
    )
    posts["created_at_utc"] = pd.to_datetime(posts["created_at_utc"], utc=True)
    posts = posts[posts["created_at_utc"] >= POST_START].copy()
    posts["event_time_et"] = (
        posts["created_at_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    )
    posts["text"] = posts["text"].fillna("")
    posts["topics"] = posts["text"].apply(_match_topics)
    posts["topic_count"] = posts["topics"].str.len()
    for topic in TOPIC_RULES:
        posts[f"kg_topic_{topic}"] = posts["topics"].apply(
            lambda values, topic=topic: int(topic in values)
        )
    posts["lex_sent"] = (
        posts["text"].str.contains(
            r"great|good|success|strong|win|beautiful", case=False, regex=True
        ).astype(int)
        - posts["text"].str.contains(
            r"bad|weak|fail|problem|crime|radical", case=False, regex=True
        ).astype(int)
    )
    posts["word_count"] = posts["text"].str.split().str.len()
    posts["hour"] = posts["event_time_et"].dt.hour
    posts["dow"] = posts["event_time_et"].dt.dayofweek
    posts["hour_sin"] = np.sin(2 * np.pi * posts["hour"] / 24.0)
    posts["hour_cos"] = np.cos(2 * np.pi * posts["hour"] / 24.0)
    posts["dow_sin"] = np.sin(2 * np.pi * posts["dow"] / 7.0)
    posts["dow_cos"] = np.cos(2 * np.pi * posts["dow"] / 7.0)
    posts["is_market_hours"] = posts["hour"].between(9, 15).astype(int)
    # REMOVED: log1p transforms of the crawl-instant engagement counts.

    prices = pd.read_csv(
        MARKET_CSV, usecols=["symbol", "datetime", "open", "high", "low", "close"]
    )
    prices["datetime"] = pd.to_datetime(prices["datetime"])
    spy = prices[prices["symbol"] == "SPY"].sort_values("datetime").copy()
    hist_ret = spy["close"].pct_change()
    # last completed close (t-1) -> close three bars after the aligned bar t
    spy["ret_fwd_1h"] = spy["close"].shift(-3) / spy["close"].shift(1) - 1.0
    spy["abs_ret_fwd_1h"] = spy["ret_fwd_1h"].abs()
    # at aligned bar t the last observable completed bar is t-1
    spy["lagged_vol_4bar"] = hist_ret.abs().rolling(4).mean().shift(1)
    spy["lagged_vol_16bar"] = hist_ret.abs().rolling(16).mean().shift(1)
    spy["lagged_ret_4bar"] = spy["close"].pct_change(4).shift(1)
    spy["lagged_range_16bar"] = (
        (spy["high"].rolling(16).max() - spy["low"].rolling(16).min())
        / spy["close"].rolling(16).mean()
    ).shift(1)

    market_cols = [
        "datetime",
        "ret_fwd_1h",
        "abs_ret_fwd_1h",
        "lagged_vol_4bar",
        "lagged_vol_16bar",
        "lagged_ret_4bar",
        "lagged_range_16bar",
    ]
    panel = pd.merge_asof(
        posts.sort_values("event_time_et"),
        spy[market_cols].sort_values("datetime"),
        left_on="event_time_et",
        right_on="datetime",
        direction="forward",
    ).dropna(subset=["ret_fwd_1h"])
    return panel.sort_values(["datetime", "event_time_et", "post_id"]).reset_index(
        drop=True
    )


# ------------------------------------------------------------- public loaders
def load_event_panel(*, use_cache: bool = True, rebuild: bool = False) -> pd.DataFrame:
    """Event-level panel: one row per post, aligned to the next 15-min SPY bar.

    Target ``target_hi_vol`` = 1[|SPY 1h return| > training median], where the
    median is taken on ``datetime < TRAIN_END`` only.  ``available_time`` is the
    aligned bar plus one hour (label maturity).  All market features are lagged
    to t-1.
    """
    cache = CACHE / "event_panel.parquet"
    if use_cache and not rebuild and cache.exists():
        return pd.read_parquet(cache)
    panel = build_post_panel()
    panel = panel.drop(columns=["topics"])
    threshold = float(panel.loc[panel["datetime"] < TRAIN_END, "abs_ret_fwd_1h"].median())
    panel["target_hi_vol"] = (panel["abs_ret_fwd_1h"] > threshold).astype(int)
    panel["available_time"] = panel["datetime"] + LABEL_DELAY
    panel.attrs["threshold_bps"] = threshold * 1e4
    panel = panel.sort_values(["datetime", "event_time_et", "post_id"]).reset_index(
        drop=True
    )
    if use_cache:
        panel.to_parquet(cache, index=False)
    return panel


def event_threshold_bps(panel: pd.DataFrame | None = None) -> float:
    panel = load_event_panel() if panel is None else panel
    return float(
        panel.loc[panel["datetime"] < TRAIN_END, "abs_ret_fwd_1h"].median() * 1e4
    )


def _event_features(post_panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse posts sharing an aligned bar; no outcome is observed."""
    aggregations: dict[str, tuple[str, str]] = {
        "n_posts": ("post_id", "size"),
        "hour_sin": ("hour_sin", "mean"),
        "hour_cos": ("hour_cos", "mean"),
        "dow_sin": ("dow_sin", "mean"),
        "dow_cos": ("dow_cos", "mean"),
        "market_hours_share": ("is_market_hours", "mean"),
        # REMOVED: log_reblogs_max / log_favourites_max / log_replies_max.
        "word_count_max": ("word_count", "max"),
        "lex_sent_mean": ("lex_sent", "mean"),
        "topic_count_sum": ("topic_count", "sum"),
    }
    for topic in TOPICS:
        aggregations[f"topic_{topic}"] = (f"kg_topic_{topic}", "sum")
    return post_panel.groupby("datetime", as_index=False).agg(**aggregations)


CATEGORY_EXPOSURE = {
    "trade_tariff": {
        "broad_market": -0.25, "china_exposed": -1.00, "tariff_winners": 1.00,
        "tariff_losers": -0.85, "sector_etfs": 0.30, "volatility": 0.70,
    },
    "china": {
        "broad_market": -0.25, "china_exposed": -1.00, "tariff_winners": 0.65,
        "tariff_losers": -0.55, "volatility": 0.60,
    },
    "macro_fed": {
        "broad_market": 0.60, "macro": 0.85, "sector_etfs": 0.45, "volatility": 0.85,
    },
    "energy": {"macro": 0.65, "sector_etfs": 0.55, "tariff_losers": -0.45},
    "crypto": {
        "crypto_adjacent": 1.00, "broad_market": 0.10, "trump_direct": 0.25,
        "tariff_losers": 0.10,
    },
    "defense": {"defense": 1.00, "sector_etfs": 0.35, "broad_market": 0.10},
    "immigration": {"broad_market": -0.10, "sector_etfs": 0.25, "volatility": 0.25},
    "tax": {"broad_market": 0.45, "sector_etfs": 0.55, "macro": 0.25, "trump_direct": 0.35},
}
ASSET_OVERRIDES = {
    "trade_tariff": {"X": 1.0, "CLF": 1.0, "NUE": 1.0, "STLD": 1.0,
                     "BABA": -1.0, "JD": -1.0, "FXI": -1.0},
    "china": {"BABA": -1.0, "JD": -1.0, "FXI": -1.0, "UUP": 0.45},
    "macro_fed": {"TLT": -1.0, "UUP": 0.75, "GLD": 0.70, "UVXY": 1.0, "VIXY": 1.0},
    "energy": {"USO": 1.0, "XLE": 1.0, "ENPH": -0.7, "FSLR": -0.7, "SEDG": -0.7},
    "crypto": {"COIN": 1.0, "GBTC": 1.0, "MSTR": 1.0, "TSLA": 0.2},
    "defense": {"GD": 1.0, "LMT": 1.0, "NOC": 1.0, "RTX": 1.0, "XAR": 1.0},
    "tax": {"XLF": 0.85, "XLI": 0.55, "XLY": 0.55},
}


def _exposure_value(topic: str, symbol: str, category: str) -> float:
    if symbol in ASSET_OVERRIDES.get(topic, {}):
        return float(ASSET_OVERRIDES[topic][symbol])
    return float(CATEGORY_EXPOSURE.get(topic, {}).get(category, 0.0))


def _market_panel(horizon_bars: int) -> pd.DataFrame:
    prices = pd.read_csv(
        MARKET_CSV,
        usecols=["symbol", "category", "datetime", "open", "high", "low", "close", "volume"],
    )
    prices["datetime"] = pd.to_datetime(prices["datetime"], errors="coerce")
    prices = prices.dropna(subset=["datetime", "open", "close"]).sort_values(
        ["symbol", "datetime"]
    )
    max_elapsed = BAR_MINUTES * horizon_bars - 5.0
    output = []
    for (symbol, category), group in prices.groupby(["symbol", "category"], sort=False):
        frame = group.copy().reset_index(drop=True)
        close_ret = frame["close"].pct_change()
        exit_close = frame["close"].shift(-(horizon_bars - 1))
        exit_time = frame["datetime"].shift(-(horizon_bars - 1))
        elapsed = (exit_time - frame["datetime"]).dt.total_seconds() / 60.0
        frame["fwd_ret"] = exit_close / frame["open"] - 1.0
        frame.loc[elapsed > max_elapsed, "fwd_ret"] = np.nan
        frame["lagged_vol_4bar"] = close_ret.abs().rolling(4).mean().shift(1)
        frame["lagged_vol_16bar"] = close_ret.abs().rolling(16).mean().shift(1)
        frame["lagged_ret_4bar"] = frame["close"].pct_change(4).shift(1)
        frame["lagged_range_16bar"] = (
            (frame["high"].rolling(16).max() - frame["low"].rolling(16).min())
            / frame["close"].rolling(16).mean()
        ).shift(1)
        frame["lagged_log_volume_16bar"] = np.log1p(
            frame["volume"].rolling(16).median().shift(1)
        )
        output.append(frame[[
            "symbol", "category", "datetime", "fwd_ret", "lagged_vol_4bar",
            "lagged_vol_16bar", "lagged_ret_4bar", "lagged_range_16bar",
            "lagged_log_volume_16bar",
        ]])
    return pd.concat(output, ignore_index=True)


def load_asset_panel(
    horizon_bars: int = HORIZON_BARS_DEFAULT,
    *,
    use_cache: bool = True,
    rebuild: bool = False,
    with_feature_sets: bool = False,
):
    """Post x asset panel: one row per (aligned bar, asset).

    Entry = open of the aligned 15-minute bar (the next bar after the post
    batch); exit = close of the ``horizon_bars``-th bar.  Windows whose true
    elapsed span exceeds ``15 * horizon_bars - 5`` minutes (a session gap) are
    dropped.  Posts sharing an aligned bar are collapsed BEFORE the asset panel
    is formed, so every (bar, asset) pair is unique.

    Returns the panel, or ``(panel, feature_sets)`` when ``with_feature_sets``.
    """
    cache = CACHE / f"asset_panel_h{horizon_bars}.parquet"
    meta = CACHE / f"asset_panel_h{horizon_bars}_features.json"
    if use_cache and not rebuild and cache.exists() and meta.exists():
        panel = pd.read_parquet(cache)
        feature_sets = json.loads(meta.read_text())
        return (panel, feature_sets) if with_feature_sets else panel

    posts = build_post_panel()
    events = _event_features(posts)
    panel = events.merge(
        _market_panel(horizon_bars), on="datetime", how="inner"
    ).dropna(subset=["fwd_ret"])

    signed_cols, abs_cols = [], []
    for topic in TOPICS:
        exposure = np.array([
            _exposure_value(topic, symbol, category)
            for symbol, category in zip(panel["symbol"], panel["category"])
        ])
        panel[f"prior_{topic}"] = exposure
        panel[f"interaction_{topic}"] = panel[f"topic_{topic}"] * exposure
        panel[f"abs_interaction_{topic}"] = panel[f"topic_{topic}"] * np.abs(exposure)
        panel[f"stance_interaction_{topic}"] = (
            panel[f"interaction_{topic}"] * panel["lex_sent_mean"]
        )
        signed_cols.append(f"interaction_{topic}")
        abs_cols.append(f"abs_interaction_{topic}")
    panel["signed_exposure_match"] = panel[signed_cols].sum(axis=1)
    panel["absolute_exposure_match"] = panel[abs_cols].sum(axis=1)

    train_mask = panel["datetime"] < TRAIN_END
    vol_scale = (
        panel.loc[train_mask].groupby("symbol")["lagged_vol_16bar"].median().clip(lower=1e-6)
    )
    abs_threshold = (
        panel.loc[train_mask].groupby("symbol")["fwd_ret"].apply(lambda x: x.abs().median())
    )
    panel["lagged_vol_scaled"] = panel["lagged_vol_16bar"] / panel["symbol"].map(vol_scale)
    panel["target_high_abs_return"] = (
        panel["fwd_ret"].abs() > panel["symbol"].map(abs_threshold)
    ).astype(int)
    panel["target_up"] = (panel["fwd_ret"] > 0.0).astype(int)
    panel["exposure_x_lagged_vol"] = (
        panel["absolute_exposure_match"] * panel["lagged_vol_scaled"]
    )
    for topic in TOPICS:
        panel[f"vol_interaction_{topic}"] = (
            panel[f"abs_interaction_{topic}"] * panel["lagged_vol_scaled"]
        )

    indicators = pd.get_dummies(
        panel[["symbol", "category"]], prefix=["asset", "category"], dtype=float
    )
    panel = pd.concat(
        [panel.reset_index(drop=True), indicators.reset_index(drop=True)], axis=1
    )

    # REMOVED from base: log_reblogs_max, log_favourites_max, log_replies_max.
    base = [
        "n_posts", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "market_hours_share", "word_count_max", "lagged_vol_4bar",
        "lagged_vol_16bar", "lagged_ret_4bar", "lagged_range_16bar",
        "lagged_log_volume_16bar", "lagged_vol_scaled",
    ] + [c for c in panel.columns if c.startswith("asset_") or c.startswith("category_")]
    content = base + ["lex_sent_mean", "topic_count_sum"] + [f"topic_{t}" for t in TOPICS]
    exposure = base + [
        "lex_sent_mean", "topic_count_sum", "signed_exposure_match",
        "absolute_exposure_match",
    ] + [f"interaction_{t}" for t in TOPICS] + [f"abs_interaction_{t}" for t in TOPICS] + [
        f"stance_interaction_{t}" for t in TOPICS
    ]
    exposure_vol = exposure + ["exposure_x_lagged_vol"] + [
        f"vol_interaction_{t}" for t in TOPICS
    ]
    feature_sets = {
        "base_no_content": base,
        "raw_content": content,
        "content_x_exposure": exposure,
        "content_x_exposure_x_volatility": exposure_vol,
    }

    panel["available_time"] = panel["datetime"] + LABEL_DELAY
    panel["asset_code"] = panel["symbol"].astype("category").cat.codes.astype(np.int32)
    panel = panel.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    if use_cache:
        panel.to_parquet(cache, index=False)
        meta.write_text(json.dumps(feature_sets, indent=2))
    return (panel, feature_sets) if with_feature_sets else panel


def asset_feature_sets(horizon_bars: int = HORIZON_BARS_DEFAULT) -> dict[str, list[str]]:
    _, feature_sets = load_asset_panel(horizon_bars, with_feature_sets=True)
    return feature_sets


# --------------------------------------------------------------------- splits
def split(df: pd.DataFrame, column: str = "datetime"):
    """Canonical (train, validation, test) split.  Never changes."""
    t = df[column]
    train = df[t < TRAIN_END].reset_index(drop=True)
    validation = df[(t >= TRAIN_END) & (t < TEST_START)].reset_index(drop=True)
    test = df[t >= TEST_START].reset_index(drop=True)
    return train, validation, test


def pre_deployment(df: pd.DataFrame, deploy_start: pd.Timestamp) -> pd.DataFrame:
    """Rows usable as memory at ``deploy_start``: labels must have matured."""
    return df[
        (df["datetime"] < deploy_start) & (df["available_time"] <= deploy_start)
    ].reset_index(drop=True)


# ---------------------------------------------------------------- slow model
def fit_slow_model(
    train: pd.DataFrame,
    features: Sequence[str] = MODEL_FEATURES,
    target: str = "target_hi_vol",
    *,
    seed: int = 0,
    trees: int = SLOW_TREES,
    depth: int = SLOW_DEPTH,
) -> GradientBoostingClassifier:
    """The FROZEN slow gradient-boosted tree (200 trees, depth 4)."""
    x = train[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return GradientBoostingClassifier(
        n_estimators=trees, max_depth=depth, random_state=seed
    ).fit(x, train[target].to_numpy(dtype=int))


def slow_probability(
    train: pd.DataFrame,
    stream: pd.DataFrame,
    features: Sequence[str] = MODEL_FEATURES,
    target: str = "target_hi_vol",
    *,
    seeds: Iterable[int] = SLOW_SEEDS,
    trees: int = SLOW_TREES,
    depth: int = SLOW_DEPTH,
) -> np.ndarray:
    """Seed-averaged frozen slow probability.  This is the ``frozen`` baseline."""
    x_stream = stream[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    seeds = list(seeds)
    total = np.zeros(len(stream), dtype=float)
    for seed in seeds:
        model = fit_slow_model(
            train, features, target, seed=seed, trees=trees, depth=depth
        )
        total += model.predict_proba(x_stream)[:, 1]
    return total / len(seeds)


def fit_asset_slow_model(train, features, target, *, seed=0):
    """HistGBT slow model used by the asset-level track."""
    x = train[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return HistGradientBoostingClassifier(random_state=seed, **ASSET_SLOW_PARAMS).fit(
        x, train[target].to_numpy(dtype=int)
    )


def asset_slow_probability(train, stream, features, target, *, seeds=ASSET_SLOW_SEEDS):
    x = stream[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    seeds = list(seeds)
    total = np.zeros(len(stream), dtype=float)
    for seed in seeds:
        total += fit_asset_slow_model(train, features, target, seed=seed).predict_proba(x)[:, 1]
    return total / len(seeds)


# ---------------------------------------------------------------- R3-TTT core
@dataclass(frozen=True)
class BarMemory:
    """Immutable, bar-deduplicated, label-matured pre-deployment memory."""

    frame: pd.DataFrame
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def build_bar_memory(
    initial: pd.DataFrame,
    *,
    target: str = "target_hi_vol",
    deploy_start: pd.Timestamp | None = None,
    features: Sequence[str] = MODEL_FEATURES,
) -> BarMemory:
    """Exactly one memory item per aligned bar, from pre-deployment data only."""
    frame = initial
    if deploy_start is not None:
        frame = frame[frame["available_time"] <= deploy_start]
    aggregation = {feature: "mean" for feature in features}
    aggregation[target] = "first"
    memory = frame.groupby("datetime", as_index=False).agg(aggregation)
    return BarMemory(memory, memory[target].to_numpy(dtype=float))


def bar_groups(stream: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each aligned bar, in time order.  Same-bar = one batch."""
    return [
        np.asarray(locs, dtype=int)
        for _, locs in sorted(stream.groupby("datetime", sort=True).indices.items())
    ]


def _route(
    memory: pd.DataFrame,
    stream: pd.DataFrame,
    groups: list[np.ndarray],
    features: Sequence[str],
    *,
    max_k: int = MAX_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar retrieval in one standardized routing space.

    Returns ``(idx, dist)`` of shape (n_bars, keep): the ``keep`` nearest memory
    items per query bar, sorted by squared distance (per-dimension mean).
    The scaler is fitted on the MEMORY only.

    NOTE (clean panel): in the 5-d timing space the distances are massively
    tied -- 1,973 of 2,249 memory bars are exact duplicates there.  The
    ``kind="stable"`` argsort below is load-bearing, not cosmetic: it is what
    makes the headline reproducible.  See ``TIEBREAK_RULE``.
    """
    features = list(features)
    mem_raw = memory[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler().fit(mem_raw)
    memory_x = scaler.transform(mem_raw)
    stream_x = scaler.transform(
        stream[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    queries = np.stack([stream_x[locs].mean(axis=0) for locs in groups], axis=0)
    keep = min(max_k, len(memory_x))
    # (n_bars, n_memory) squared distance, averaged over dimensions
    d = (
        np.einsum("ij,ij->i", queries, queries)[:, None]
        - 2.0 * (queries @ memory_x.T)
        + np.einsum("ij,ij->i", memory_x, memory_x)[None, :]
    ) / float(len(features))
    np.maximum(d, 0.0, out=d)
    local = np.argpartition(d, keep - 1, axis=1)[:, :keep]
    local_d = np.take_along_axis(d, local, axis=1)
    order = np.argsort(local_d, axis=1, kind="stable")
    return (
        np.take_along_axis(local, order, axis=1),
        np.take_along_axis(local_d, order, axis=1),
    )


def _fast_states(
    idx: np.ndarray,
    dist: np.ndarray,
    memory_labels: np.ndarray,
    *,
    balanced: bool = BALANCED_DEFAULT,
    scales: Sequence[int] = SCALES,
    temperatures: Sequence[float] = TEMPERATURES,
    prior_strengths: Sequence[float] = PRIOR_STRENGTHS,
    gates: Sequence[bool] = GATES,
) -> dict[tuple[float, float, bool], tuple[np.ndarray, np.ndarray]]:
    """Closed-form weighted Bernoulli fast state, scale-ensembled over k.

    For each k the retrieval keeps the k nearest memory items (``balanced=False``,
    the archived v4 behaviour) or the k/2 nearest positives plus the k/2 nearest
    negatives (``balanced=True``), then

        w_j = exp(-(d_j - d_min)/T)
        q   = (sum_j w_j y_j + lambda*pi0) / (sum_j w_j + lambda)
        u   = logit(q)

    ``u`` is averaged over k; ``spread`` is its std over k; ``consensus`` is the
    mean over k of |share - 0.5| * 2.  ``trust = consensus/(1+spread)`` when the
    gate is on and 1 otherwise.
    """
    global_prior = float(np.mean(memory_labels))
    labels = memory_labels[idx].astype(np.float64)
    n_query, keep = labels.shape
    finite = np.isfinite(dist)
    labels = np.where(finite, labels, 0.0)
    rank_one = np.cumsum(labels, axis=1)
    rank_zero = np.cumsum(np.where(finite, 1.0 - labels, 0.0), axis=1)
    position = np.arange(keep)[None, :]
    nearest = dist[:, 0].astype(np.float64)
    d_min = np.where(np.isfinite(nearest), nearest, 0.0)[:, None]

    selections = {}
    for scale in scales:
        if balanced:
            per_class = max(1, scale // 2)
            mask = (
                ((labels == 1.0) & (rank_one <= per_class))
                | ((labels == 0.0) & (rank_zero <= per_class))
            )
        else:
            mask = position < scale
        selections[scale] = (mask & finite).astype(np.float64)

    output: dict[tuple[float, float, bool], tuple[np.ndarray, np.ndarray]] = {}
    for temperature in temperatures:
        gap = np.where(finite, dist.astype(np.float64) - d_min, np.inf)
        weight = np.exp(-np.clip(gap / temperature, 0.0, 700.0))
        biases = {prior: [] for prior in prior_strengths}
        agreements = []
        for scale in scales:
            selected = weight * selections[scale]
            mass = selected.sum(axis=1)
            positive = (selected * labels).sum(axis=1)
            share = np.where(mass > 0.0, positive / np.maximum(mass, 1e-12), global_prior)
            agreements.append(np.abs(share - 0.5) * 2.0)
            for prior in prior_strengths:
                q = (positive + prior * global_prior) / (mass + prior)
                biases[prior].append(logit(q))
        consensus = np.mean(np.stack(agreements, axis=0), axis=0)
        for prior in prior_strengths:
            stacked = np.stack(biases[prior], axis=0)
            bias = stacked.mean(axis=0)
            spread = stacked.std(axis=0)
            for gate in gates:
                trust = consensus / (1.0 + spread) if gate else np.ones(n_query)
                output[(temperature, prior, bool(gate))] = (bias, trust)
    return output


def fuse(slow_probability_: np.ndarray, bias: np.ndarray, trust: np.ndarray, beta: float):
    """p = sigmoid((1 - beta*trust) * slow_logit + beta*trust * u)."""
    effective = beta * trust
    return sigmoid((1.0 - effective) * logit(slow_probability_) + effective * bias)


def r3ttt_predict(
    memory: BarMemory,
    stream: pd.DataFrame,
    slow_probability_: np.ndarray,
    *,
    routing_features: Sequence[str] = TIMING,
    k: int = 32,
    beta: float = 0.5,
    temperature: float = 1.0,
    prior_strength: float = 2.0,
    gate: bool = False,
    balanced: bool = BALANCED_DEFAULT,
    groups: list[np.ndarray] | None = None,
    max_k: int = MAX_K,
    return_state: bool = False,
):
    """ONE R3-TTT configuration: route -> fast state -> fuse.

    ``k`` may be an int (single scale) or a sequence (scale ensemble).  Every
    row of a bar receives that bar's fast state; same-bar posts are one batch.
    """
    groups = bar_groups(stream) if groups is None else groups
    scales = (k,) if np.isscalar(k) else tuple(k)
    idx, dist = _route(memory.frame, stream, groups, routing_features, max_k=max_k)
    states = _fast_states(
        idx, dist, memory.labels,
        balanced=balanced, scales=scales,
        temperatures=(temperature,), prior_strengths=(prior_strength,), gates=(gate,),
    )
    bias_bar, trust_bar = states[(temperature, prior_strength, bool(gate))]
    bias = np.empty(len(stream), dtype=float)
    trust = np.empty(len(stream), dtype=float)
    for position, locs in enumerate(groups):
        bias[locs] = bias_bar[position]
        trust[locs] = trust_bar[position]
    probability = fuse(slow_probability_, bias, trust, beta)
    if return_state:
        return probability, bias, trust
    return probability


def r3ttt_selection_free(
    memory: BarMemory,
    stream: pd.DataFrame,
    slow_probability_: np.ndarray,
    *,
    retrieval_spaces: dict[str, list[str]] | None = None,
    scales: Sequence[int] = SCALES,
    blends: Sequence[float] = BLENDS,
    temperatures: Sequence[float] = TEMPERATURES,
    prior_strengths: Sequence[float] = PRIOR_STRENGTHS,
    gates: Sequence[bool] = GATES,
    balanced: bool = BALANCED_DEFAULT,
    groups: list[np.ndarray] | None = None,
    max_k: int = MAX_K,
    return_members: bool = False,
):
    """The HEADLINE variant: uniform average over the 72-member grid.

    3 routing spaces x 3 beta x 2 T x 2 lambda x 2 gate = 72 members.  Every
    member internally averages the fast state over k in {8,16,32,64}.  Nothing
    is selected; validation is consulted for nothing.

    Returns ``(probability, bias, trust, n_members)``.
    """
    retrieval_spaces = RETRIEVAL_SPACES if retrieval_spaces is None else retrieval_spaces
    groups = bar_groups(stream) if groups is None else groups
    n = len(stream)
    total = np.zeros(n, dtype=float)
    total_bias = np.zeros(n, dtype=float)
    total_trust = np.zeros(n, dtype=float)
    members: list[dict] = []
    count = 0
    row_bar = np.empty(n, dtype=int)
    for position, locs in enumerate(groups):
        row_bar[locs] = position
    for space_name, features in retrieval_spaces.items():
        idx, dist = _route(memory.frame, stream, groups, features, max_k=max_k)
        states = _fast_states(
            idx, dist, memory.labels, balanced=balanced, scales=scales,
            temperatures=temperatures, prior_strengths=prior_strengths, gates=gates,
        )
        for (temperature, prior, gate), (bias_bar, trust_bar) in states.items():
            bias = bias_bar[row_bar]
            trust = trust_bar[row_bar]
            for beta in blends:
                probability = fuse(slow_probability_, bias, trust, beta)
                total += probability
                total_bias += bias
                total_trust += trust
                count += 1
                if return_members:
                    members.append({
                        "space": space_name, "temperature": temperature,
                        "prior_strength": prior, "gate": bool(gate), "blend": beta,
                        "probability": probability,
                    })
    result = (total / count, total_bias / count, total_trust / count, count)
    return (*result, members) if return_members else result


# ----------------------------------------------------------- headline runner
def headline_event_run(
    *,
    n_boot: int = 5000,
    block_days: int = 5,
    seed: int = RANDOM_SEED,
    panel: pd.DataFrame | None = None,
):
    """The canonical clean headline, end to end, in one call.

    The slow model is fitted on **pretest = train + validation** (everything with
    ``datetime < TEST_START``), NOT on train alone.  The memory is built from the
    same pretest rows, bar-deduplicated and label-matured at ``TEST_START``.  The
    test split is scored once.

    Returns a dict with ``sf_auc``, ``frozen_auc``, ``bootstrap`` and the raw
    per-row arrays, plus the test frame.
    """
    panel = load_event_panel() if panel is None else panel
    assert_clean(panel, where="event panel")
    pretest = panel[panel["datetime"] < TEST_START].reset_index(drop=True)
    test = panel[panel["datetime"] >= TEST_START].reset_index(drop=True)
    y = test["target_hi_vol"].to_numpy(dtype=int)

    frozen = slow_probability(pretest, test)
    memory = build_bar_memory(pretest, deploy_start=TEST_START)
    sf, bias, trust, n_members = r3ttt_selection_free(memory, test, frozen)

    boot = block_bootstrap_paired(
        y, sf, frozen, test["datetime"],
        n_boot=n_boot, block_days=block_days, metric="auc", seed=seed,
    )
    return {
        "sf_auc": fast_binary_auc(y, sf),
        "frozen_auc": fast_binary_auc(y, frozen),
        "sf_brier": brier(y, sf),
        "frozen_brier": brier(y, frozen),
        "delta_auc": fast_binary_auc(y, sf) - fast_binary_auc(y, frozen),
        "n_members": int(n_members),
        "n_memory": len(memory),
        "n_test": int(len(test)),
        "bootstrap": boot,
        "y": y, "sf": sf, "frozen": frozen, "bias": bias, "trust": trust,
        "test": test, "pretest": pretest, "memory": memory,
    }


# ------------------------------------------------------------------ bootstrap
def block_bootstrap_paired(
    y,
    score_a,
    score_b,
    days,
    n_boot: int = 5000,
    block_days: int = 5,
    metric: str = "auc",
    seed: int = RANDOM_SEED,
) -> dict:
    """Paired 5-day moving-block bootstrap, clustered BY CALENDAR DAY.

    Contiguous blocks of ``block_days`` distinct calendar days are resampled with
    replacement; every row of a sampled day travels with it (39 correlated assets
    are not 39 independent samples).  ``delta = metric(a) - metric(b)``, and for
    ``metric='brier'`` the sign is flipped so positive always means "a is better".
    """
    y = np.asarray(y)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    days = pd.to_datetime(pd.Series(np.asarray(days))).dt.normalize().to_numpy()
    unique_days = np.array(sorted(set(days)))
    day_rows = {day: np.flatnonzero(days == day) for day in unique_days}
    n_days = len(unique_days)
    n_blocks = max(1, int(np.ceil(n_days / block_days)))
    high = max(1, n_days - block_days + 1)

    if metric == "auc":
        def evaluate(yy, pp):
            return fast_binary_auc(yy, pp) if np.unique(yy).size == 2 else np.nan
        sign = 1.0
    elif metric == "brier":
        def evaluate(yy, pp):
            return brier(yy, pp)
        sign = -1.0
    elif metric == "mean":
        def evaluate(yy, pp):
            return float(np.nanmean(pp))
        sign = 1.0
    else:
        raise ValueError(f"unknown metric {metric!r}")

    point = sign * (evaluate(y, score_a) - evaluate(y, score_b))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    deltas[:] = np.nan
    for b in range(n_boot):
        starts = rng.integers(0, high, size=n_blocks)
        chosen = [d for start in starts for d in unique_days[start:start + block_days]]
        rows = np.concatenate([day_rows[d] for d in chosen])
        va = evaluate(y[rows], score_a[rows])
        vb = evaluate(y[rows], score_b[rows])
        if np.isfinite(va) and np.isfinite(vb):
            deltas[b] = sign * (va - vb)
    finite = deltas[np.isfinite(deltas)]
    return {
        "metric": metric,
        "point_a": float(evaluate(y, score_a)),
        "point_b": float(evaluate(y, score_b)),
        "delta": float(point),
        "ci_95": [float(v) for v in np.percentile(finite, [2.5, 97.5])],
        "p_delta_le_zero": float(np.mean(finite <= 0.0)),
        "n_boot": int(len(finite)),
        "block_days": int(block_days),
        "n_days": int(n_days),
        "clustered_by": "calendar day",
        "seed": int(seed),
    }


# ------------------------------------------------------------------ protocol
def protocol_json(**extra) -> dict:
    """Canonical protocol record; every track should dump this (plus extras)."""
    event = load_event_panel()
    e_train, e_val, e_test = split(event)
    asset = load_asset_panel()
    a_train, a_val, a_test = split(asset)
    record = {
        "panel": "CLEAN -- crawl-instant engagement snapshot removed",
        "removed_features": {
            "raw": list(CONTAMINATED_RAW),
            "log": list(CONTAMINATED_FEATURES),
            "bar_aggregates": list(CONTAMINATED_AGGREGATES),
            "reason": (
                "single crawl-instant snapshot taken 2026-04-10, months after "
                "the test period opens 2026-01-01; look-ahead for every test row"
            ),
        },
        "environment_warning": (
            "clean panel is a NEW environment; never difference its point "
            "estimates against contaminated-panel point estimates"
        ),
        "split": {
            "train": "datetime < 2025-10-01",
            "validation": "2025-10-01 <= datetime < 2026-01-01",
            "test": "datetime >= 2026-01-01",
        },
        "event_panel": {
            "n_train": len(e_train), "n_validation": len(e_val), "n_test": len(e_test),
            "n_features": len(MODEL_FEATURES),
            "features": list(MODEL_FEATURES),
            "target": "1[|SPY 1h return| > training median]",
            "threshold_bps": event_threshold_bps(event),
        },
        "asset_panel": {
            "n_train": len(a_train), "n_validation": len(a_val), "n_test": len(a_test),
            "n_train_bars": int(a_train["datetime"].nunique()),
            "n_validation_bars": int(a_val["datetime"].nunique()),
            "n_test_bars": int(a_test["datetime"].nunique()),
            "n_assets": int(asset["symbol"].nunique()),
        },
        "slow_model": {
            "estimator": "GradientBoostingClassifier",
            "n_estimators": SLOW_TREES, "max_depth": SLOW_DEPTH,
            "seeds_averaged": list(SLOW_SEEDS), "frozen": True,
        },
        "r3ttt": {
            "retrieval_spaces": {k: list(v) for k, v in RETRIEVAL_SPACES.items()},
            "scales_k": list(SCALES), "blends_beta": list(BLENDS),
            "temperatures_T": list(TEMPERATURES),
            "prior_strengths_lambda": list(PRIOR_STRENGTHS),
            "gates": list(GATES), "candidate_pool": MAX_K,
            "balanced_retrieval": BALANCED_DEFAULT,
            "tiebreak_rule": TIEBREAK_RULE,
            "timing_space_degeneracy": (
                "1973/2249 memory bars are exact duplicates in the 5-d timing "
                "space; 83.8% of query bars have ties straddling k=64; "
                "tie-break is worth ~0.0014 of SF test AUC"
            ),
            "n_ensemble_members": N_ENSEMBLE_MEMBERS,
            "memory": "immutable, bar-deduplicated, label-matured, pre-deployment only",
            "selection_rule": "NONE for the headline variant (selection-free ensemble)",
        },
        "uncertainty": {
            "estimator": "paired 5-day moving-block bootstrap",
            "clustered_by": "calendar day", "default_n_boot": 5000,
        },
        "label_delay_hours": 1,
        "seed": RANDOM_SEED,
    }
    record.update(extra)
    return record


__all__ = [
    "TRAIN_END", "TEST_START", "LABEL_DELAY", "RANDOM_SEED",
    "CONTAMINATED_FEATURES", "CONTAMINATED_AGGREGATES", "CONTAMINATED_RAW",
    "CONTAMINATED_ALL", "assert_clean", "headline_event_run",
    "TIMING", "TIMING_ENGAGEMENT", "MARKET_STATE", "MODEL_FEATURES",
    "RETRIEVAL_SPACES",
    "SCALES", "BLENDS", "TEMPERATURES", "PRIOR_STRENGTHS", "GATES", "MAX_K",
    "BALANCED_DEFAULT",
    "N_ENSEMBLE_MEMBERS", "SLOW_SEEDS", "TOPICS",
    "sigmoid", "logit", "fast_binary_auc", "brier",
    "build_post_panel", "load_event_panel", "load_asset_panel",
    "asset_feature_sets", "event_threshold_bps",
    "split", "pre_deployment",
    "fit_slow_model", "slow_probability",
    "fit_asset_slow_model", "asset_slow_probability",
    "BarMemory", "build_bar_memory", "bar_groups",
    "r3ttt_predict", "r3ttt_selection_free", "fuse",
    "block_bootstrap_paired", "protocol_json",
]
