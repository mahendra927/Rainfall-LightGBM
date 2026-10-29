"""Unit tests for the leakage-sensitive and statistical components (pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rainfall_lgbm.features import add_lag_features  # noqa: E402
from rainfall_lgbm.metrics import macro_f1_from_counts  # noqa: E402
from rainfall_lgbm.stats import holm, wilcoxon_holm  # noqa: E402


def test_lag_uses_exact_calendar_days_and_keeps_gaps_missing():
    df = pd.DataFrame({"Location": ["A"] * 4, "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"]),
                       "X": [1.0, 2.0, 4.0, 5.0]})
    out = add_lag_features(df, ["X"], [1, 2])
    assert out["X_lag1"].tolist()[1] == 1.0          # Jan 2 <- Jan 1
    assert np.isnan(out["X_lag1"].tolist()[2])        # Jan 4: Jan 3 missing -> NaN, not Jan 2
    assert out["X_lag2"].tolist()[2] == 2.0          # Jan 4 <- Jan 2
    assert out["X_lag1"].tolist()[3] == 4.0


def test_lag_never_crosses_locations():
    df = pd.DataFrame({"Location": ["A", "B"], "Date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "X": [1.0, 2.0]})
    out = add_lag_features(df, ["X"], [1])
    assert out["X_lag1"].isna().all()


def test_wilcoxon_minimum_p_with_five_pairs():
    scores = pd.DataFrame({"A1": [0.77, 0.77, 0.78, 0.77, 0.77], "A8": [0.82, 0.83, 0.83, 0.82, 0.83]})
    res = wilcoxon_holm(scores, "A1")
    assert abs(res.loc[0, "p_value"] - 0.0625) < 1e-12
    assert not res.loc[0, "significant_at_alpha"]


def test_holm_monotone():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_macro_f1_from_counts():
    assert abs(macro_f1_from_counts(5, 0, 0, 5) - 1.0) < 1e-12
