"""Post-hoc error analysis. Rainfall amount is used ONLY here, never as a predictor."""
from __future__ import annotations

import numpy as np
import pandas as pd


def false_negatives_by_intensity(pred: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    yes = pred[pred["y_true"] == 1].copy()
    bins = list(edges) + [np.inf]
    labels = [f"{a:g}–{b:g} mm" for a, b in zip(edges[:-1], edges[1:])] + [f"Above {edges[-1]:g} mm"]
    yes["intensity"] = pd.cut(yes["Rainfall"], bins=bins, labels=labels, include_lowest=True, right=True)
    g = yes.groupby("intensity", observed=False)
    out = pd.DataFrame({"rainfall_cases": g.size(), "false_negatives": g["y_pred"].apply(lambda s: int((s == 0).sum()))})
    out["false_negative_rate"] = out["false_negatives"] / out["rainfall_cases"].replace(0, np.nan)
    missing = int(yes["intensity"].isna().sum())
    out.attrs["rainfall_yes_without_amount_or_out_of_range"] = missing
    return out.reset_index()


def recall_by_location(pred: pd.DataFrame) -> pd.DataFrame:
    yes = pred[pred["y_true"] == 1]
    g = yes.groupby("Location")["y_pred"]
    out = pd.DataFrame({"rainfall_cases": g.size(), "recall_yes": g.mean()}).sort_values("recall_yes")
    return out.reset_index()
