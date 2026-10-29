"""Data-quality treatments and class-imbalance handling.

Four operations are distinguished:
* anomaly detection  - flags multivariate unusual observations (Isolation Forest)
* outlier removal    - excludes flagged TRAINING observations (A5, A6)
* extreme-value capping - keeps all observations, clips values to bounds (IQR: A7, Z-score: A8)
* error correction   - repairing verified erroneous records (NOT evaluated in this study)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------- extreme-value capping ----------
def iqr_bounds(X: pd.DataFrame, cols: list[str], k: float = 1.5) -> tuple[pd.Series, pd.Series]:
    q1, q3 = X[cols].quantile(0.25), X[cols].quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def zscore_bounds(X: pd.DataFrame, cols: list[str], tau: float = 3.0) -> tuple[pd.Series, pd.Series]:
    mu, sd = X[cols].mean(), X[cols].std(ddof=0)
    return mu - tau * sd, mu + tau * sd


def clip(X: pd.DataFrame, cols: list[str], lower: pd.Series, upper: pd.Series) -> tuple[pd.DataFrame, int]:
    out = X.copy()
    before = out[cols].to_numpy()
    out[cols] = out[cols].clip(lower=lower, upper=upper, axis=1)
    changed = int((out[cols].to_numpy() != before).sum())
    return out, changed


# ---------- anomaly detection + outlier removal ----------
def isolation_forest_mask(X_train: pd.DataFrame, cols: list[str], n_estimators: int,
                          contamination: float, seed: int) -> np.ndarray:
    """Return a boolean mask of TRAINING rows to keep (True = not anomalous)."""
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                          random_state=seed, n_jobs=-1)
    pred = iso.fit_predict(X_train[cols].to_numpy(dtype=float))
    return pred == 1


# ---------- class imbalance ----------
def apply_smotenc(X: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str],
                  sampling_strategy: float, k_neighbors: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTENC on the training subset only.

    Numerical features are standardized during oversampling and returned to their
    original scale afterwards. Wind-direction variables are declared categorical, so
    synthetic samples receive an observed category instead of an interpolated value.
    """
    cols = categorical + numeric
    scaler = StandardScaler().fit(X[numeric])
    Z = X[cols].copy()
    Z[numeric] = scaler.transform(X[numeric])
    sm = SMOTENC(categorical_features=list(range(len(categorical))),
                 sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=seed)
    Zr, yr = sm.fit_resample(Z, y)
    Zr = pd.DataFrame(Zr, columns=cols)
    Zr[numeric] = scaler.inverse_transform(Zr[numeric])
    for c in categorical:
        Zr[c] = Zr[c].round().astype(int)
    return Zr, pd.Series(np.asarray(yr), name="y")
