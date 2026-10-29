"""Loading and auditing the WeatherAUS dataset."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve


def sha256sum(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_weatheraus(cfg: dict) -> pd.DataFrame:
    """Load WeatherAUS, keep labelled rows, sort by Location and Date.

    Returns a DataFrame with a binary column ``y`` (1 = rainfall occurrence).
    """
    path = resolve(cfg["data"]["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download weatherAUS.csv from Kaggle "
            "(see data/README_data.md) and place it in data/raw/."
        )
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    target = cfg["data"]["target"]
    df = df[df[target].notna()].copy()
    df["y"] = (df[target] == cfg["data"]["positive_label"]).astype(int)
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)
    return df


def load_raw(cfg: dict) -> pd.DataFrame:
    raw = pd.read_csv(resolve(cfg["data"]["path"]))
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw


def add_lags(df: pd.DataFrame, cfg: dict, lags: list[int]) -> pd.DataFrame:
    """Add lag features using the source table selected in ``lag_features.source``."""
    from .features import add_lag_features
    source = load_raw(cfg) if cfg["lag_features"].get("source", "labelled") == "raw" else None
    return add_lag_features(df, cfg["lag_features"]["variables"], lags, source=source)


def audit(raw_path: str | Path, cfg: dict) -> dict:
    """Dataset audit reported in the manuscript (sizes, duplicates, gaps, 999.9 hPa)."""
    raw = pd.read_csv(resolve(raw_path))
    raw["Date"] = pd.to_datetime(raw["Date"])
    target = cfg["data"]["target"]
    lab = raw[raw[target].notna()].sort_values(["Location", "Date"])
    srt = raw.sort_values(["Location", "Date"])
    gaps = srt.groupby("Location")["Date"].diff().dt.days          # all rows (manuscript: 143)
    gaps_lab = lab.groupby("Location")["Date"].diff().dt.days      # labelled rows only
    pressure_cols = [c for c in ("Pressure9am", "Pressure3pm") if c in raw.columns]
    return {
        "sha256": sha256sum(resolve(raw_path)),
        "rows_total": int(len(raw)),
        "columns_total": int(raw.shape[1]),
        "rows_missing_target": int(raw[target].isna().sum()),
        "rows_labelled": int(len(lab)),
        "class_counts": {str(k): int(v) for k, v in lab[target].value_counts().items()},
        "locations": int(lab["Location"].nunique()),
        "date_min": str(lab["Date"].min().date()),
        "date_max": str(lab["Date"].max().date()),
        "duplicated_location_date_pairs": int(lab.duplicated(["Location", "Date"]).sum()),
        "intervals_longer_than_one_day": int((gaps > 1).sum()),
        "intervals_longer_than_one_day_labelled_rows": int((gaps_lab > 1).sum()),
        "pressure_999_9_hpa_count": {c: int(np.isclose(raw[c], 999.9).sum()) for c in pressure_cols},
    }
