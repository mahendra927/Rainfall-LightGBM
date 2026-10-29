"""Chronological and stratified data partitions, stored as Location + Date keys."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import resolve



def chronological_split(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """All observations of a calendar date go to the same subset (no date overlap)."""
    tr_end = pd.Timestamp(cfg["chronological"]["train_end"])
    va_end = pd.Timestamp(cfg["chronological"]["validation_end"])
    subset = pd.Series("test", index=df.index)
    subset[df["Date"] <= va_end] = "validation"
    subset[df["Date"] <= tr_end] = "train"
    return subset


def stratified_split(df: pd.DataFrame, seed: int, cfg: dict) -> pd.Series:
    """Stratified 70:15:15 split (supplementary retrospective evaluation only)."""
    s = cfg["stratified"]
    idx = df.index.to_numpy()
    y = df["y"].to_numpy()
    tr, tmp = train_test_split(idx, test_size=1 - s["train_size"], stratify=y, random_state=seed)
    rel = s["test_size"] / (s["validation_size"] + s["test_size"])
    va, te = train_test_split(tmp, test_size=rel, stratify=df.loc[tmp, "y"], random_state=seed)
    subset = pd.Series("", index=df.index)
    subset[tr], subset[va], subset[te] = "train", "validation", "test"
    return subset


def save_split(df: pd.DataFrame, subset: pd.Series, name: str, cfg: dict) -> Path:
    path = resolve(cfg["paths"]["splits"]) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"Location": df["Location"], "Date": df["Date"].dt.strftime("%Y-%m-%d"), "subset": subset})
    out.to_csv(path, index=False)
    return path


def load_split(df: pd.DataFrame, name: str, cfg: dict) -> pd.Series:
    """Map a stored split back onto ``df`` using the Location + Date key."""
    path = resolve(cfg["paths"]["splits"]) / f"{name}.csv"
    sp = pd.read_csv(path)
    sp["Date"] = pd.to_datetime(sp["Date"])
    merged = df[["Location", "Date"]].merge(sp, on=["Location", "Date"], how="left", validate="one_to_one")
    if merged["subset"].isna().any():
        raise ValueError(f"{name}: {merged['subset'].isna().sum()} rows are not covered by the stored split")
    return pd.Series(merged["subset"].to_numpy(), index=df.index)


def get_split(df: pd.DataFrame, cfg: dict, protocol: str, seed: int | None = None) -> pd.Series:
    name = "chronological" if protocol == "chronological" else f"stratified_seed{seed}"
    if (resolve(cfg["paths"]["splits"]) / f"{name}.csv").exists():
        return load_split(df, name, cfg)
    subset = chronological_split(df, cfg) if protocol == "chronological" else stratified_split(df, seed, cfg)
    save_split(df, subset, name, cfg)
    return subset
