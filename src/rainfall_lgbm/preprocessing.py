"""Training-based imputation and categorical encoding.

All statistics are estimated on the training subset only and then applied
unchanged to validation and test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self, numeric: list[str], categorical: list[str]):
        self.numeric = list(numeric)
        self.categorical = list(categorical)

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.medians_ = X[self.numeric].median()
        self.modes_ = {c: X[c].mode(dropna=True).iloc[0] for c in self.categorical}
        self.mappings_ = {}
        for c in self.categorical:
            cats = sorted(X[c].fillna(self.modes_[c]).astype(str).unique())
            self.mappings_[c] = {v: i for i, v in enumerate(cats)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in self.categorical:
            unknown = len(self.mappings_[c])  # designated code for unseen categories
            s = X[c].fillna(self.modes_[c]).astype(str)
            out[c] = s.map(self.mappings_[c]).fillna(unknown).astype(int)
        num = X[self.numeric].astype(float).fillna(self.medians_)
        out = pd.concat([out, num], axis=1)
        return out[self.categorical + self.numeric]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def numeric_array(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return X[cols].to_numpy(dtype=float)
