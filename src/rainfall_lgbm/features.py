"""Temporal lag features based on exact previous calendar days."""
from __future__ import annotations

import pandas as pd


def lag_name(var: str, k: int) -> str:
    return f"{var}_lag{k}"


def add_lag_features(df: pd.DataFrame, variables: list[str], lags: list[int],
                     source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add X_{l,t-k} for each variable and lag k.

    The features are generated ONCE on the complete dataset, sorted by Location and
    Date, BEFORE any partitioning. Each feature looks up the value recorded exactly
    k calendar days earlier at the same location. When that date is not present,
    the feature stays missing (it is never filled with the previous available row).
    Because only earlier dates are used, no same-day or future information enters
    a feature, and observations at partition boundaries can legitimately use values
    from the preceding period.

    ``source`` (optional) is the table the historical values are taken from. Passing the
    raw dataset (including rows whose RainToday label is missing) lets lags use predictor
    values recorded on unlabelled days; see ``lag_features.source`` in configs/base.yaml.
    """
    out = df.copy()
    src = out if source is None else source
    base = src[["Location", "Date"] + list(variables)]
    for k in lags:
        shifted = base.copy()
        shifted["Date"] = shifted["Date"] + pd.Timedelta(days=k)
        shifted = shifted.rename(columns={v: lag_name(v, k) for v in variables})
        out = out.merge(shifted, on=["Location", "Date"], how="left", validate="one_to_one")
    return out


def predictor_columns(cfg: dict, lags: list[int] | None) -> tuple[list[str], list[str], list[str]]:
    """Return (all predictors, numeric predictors, categorical predictors)."""
    from .data_schema import WEATHERAUS_COLUMNS

    excluded = set(cfg["data"]["exclude_from_predictors"]) | set(cfg["data"]["metadata"])
    base = [c for c in WEATHERAUS_COLUMNS if c not in excluded]
    cat = [c for c in base if c in cfg["data"]["categorical"]]
    num = [c for c in base if c not in cat]
    if lags:
        num = num + [lag_name(v, k) for k in lags for v in cfg["lag_features"]["variables"]]
    return cat + num, num, cat
