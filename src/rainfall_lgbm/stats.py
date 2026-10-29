"""Statistical comparisons: paired weekly block bootstrap and Wilcoxon + Holm."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .metrics import macro_f1_from_counts


def _block_counts(pred: pd.DataFrame, week: pd.Series, blocks: np.ndarray) -> np.ndarray:
    y, p = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    df = pd.DataFrame({"w": week.to_numpy(), "tp": (y == 1) & (p == 1), "fp": (y == 0) & (p == 1),
                       "fn": (y == 1) & (p == 0), "tn": (y == 0) & (p == 0)})
    agg = df.groupby("w")[["tp", "fp", "fn", "tn"]].sum().reindex(blocks).fillna(0)
    return agg.to_numpy(dtype=float)  # shape (n_blocks, 4)


def weekly_block_bootstrap(pred_a: pd.DataFrame, pred_b: pd.DataFrame, n_resamples: int = 1500,
                           seed: int = 2026, confidence: float = 0.95) -> dict:
    """Paired bootstrap of macro-F1(a) - macro-F1(b) over calendar-week blocks (Mon-Sun).

    All observations from all stations in the same week stay together; the same sampled
    blocks are used for both models; confusion-matrix counts are aggregated over the
    sampled blocks and macro F1 is recomputed; percentile confidence interval.
    """
    key = ["Location", "Date"]
    a = pred_a.sort_values(key).reset_index(drop=True)
    b = pred_b.sort_values(key).reset_index(drop=True)
    if not (a[key].equals(b[key]) and np.array_equal(a["y_true"], b["y_true"])):
        raise ValueError("Predictions must refer to the same test observations")
    dates = pd.to_datetime(a["Date"])
    week = dates - pd.to_timedelta(dates.dt.weekday, unit="D")  # Monday of each week
    blocks = np.sort(week.unique())
    ca, cb = _block_counts(a, week, blocks), _block_counts(b, week, blocks)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_resamples, len(blocks)))
    sa, sb = ca[idx].sum(axis=1), cb[idx].sum(axis=1)
    diffs = macro_f1_from_counts(*sa.T) - macro_f1_from_counts(*sb.T)
    point = float(macro_f1_from_counts(*ca.sum(0)) - macro_f1_from_counts(*cb.sum(0)))
    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return {"n_blocks": int(len(blocks)), "difference": point, "ci_low": float(lo), "ci_high": float(hi),
            "ci_includes_zero": bool(lo <= 0 <= hi), "bootstrap_mean": float(diffs.mean())}


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def wilcoxon_holm(scores: pd.DataFrame, reference: str, alpha: float = 0.05) -> pd.DataFrame:
    """Exploratory paired Wilcoxon signed-rank tests on per-seed macro F1 vs. ``reference``.

    ``scores``: rows = seeds, columns = scenarios. With n = 5 pairs the smallest attainable
    exact two-sided p-value is 2/2^5 = 0.0625, so significance at 0.05 is unattainable by
    design; results are reported as exploratory evidence only.
    """
    rows = []
    for sc in [c for c in scores.columns if c != reference]:
        x, y = scores[sc].to_numpy(), scores[reference].to_numpy()
        d = x - y
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(x, y, alternative="two-sided", method="exact").pvalue)
        rows.append({"comparison": f"{sc} vs {reference}", "n_pairs": len(d), "mean_difference": float(d.mean()),
                     "positive_differences": int((d > 0).sum()), "p_value": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"].tolist())
    out["significant_at_alpha"] = out["p_holm"] < alpha
    out["min_attainable_p"] = 2 / 2 ** len(scores)
    return out
