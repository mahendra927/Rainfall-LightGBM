"""Generate a small synthetic file with the WeatherAUS schema for smoke testing.

The values are random and carry NO meteorological meaning. They only verify that
the full pipeline runs end to end without the real dataset.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rainfall_lgbm.data_schema import WEATHERAUS_COLUMNS  # noqa: E402


def make(n_locations=3, start="2007-11-01", end="2017-06-25", every_k_days=1, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq=f"{every_k_days}D")
    dirs = ["N", "NNE", "NE", "E", "SE", "S", "SW", "W", "NW"]
    rows = []
    for li in range(n_locations):
        for d in dates:
            if rng.random() < 0.01:   # create a few calendar gaps
                continue
            hum = rng.uniform(20, 100)
            rain = max(0.0, rng.gamma(0.6, 6) * (hum > 60))
            rows.append({
                "Date": d.strftime("%Y-%m-%d"), "Location": f"Loc{li}",
                "MinTemp": rng.normal(12, 5), "MaxTemp": rng.normal(23, 6), "Rainfall": round(rain, 1),
                "Evaporation": rng.gamma(2, 2) if rng.random() > 0.4 else np.nan,
                "Sunshine": rng.uniform(0, 13) if rng.random() > 0.4 else np.nan,
                "WindGustDir": rng.choice(dirs) if rng.random() > 0.05 else np.nan,
                "WindGustSpeed": rng.normal(40, 12), "WindDir9am": rng.choice(dirs), "WindDir3pm": rng.choice(dirs),
                "WindSpeed9am": rng.normal(14, 8), "WindSpeed3pm": rng.normal(18, 8),
                "Humidity9am": min(100, hum + rng.normal(5, 5)), "Humidity3pm": hum,
                "Pressure9am": rng.normal(1017, 7), "Pressure3pm": rng.normal(1015, 7),
                "Cloud9am": rng.integers(0, 9), "Cloud3pm": rng.integers(0, 9),
                "Temp9am": rng.normal(17, 6), "Temp3pm": rng.normal(21, 6),
                "RainToday": ("Yes" if rain > 1 else "No") if rng.random() > 0.02 else np.nan,
                "RainTomorrow": rng.choice(["Yes", "No"]),
            })
    return pd.DataFrame(rows)[WEATHERAUS_COLUMNS]


if __name__ == "__main__":
    out = ROOT / "tests" / "_synthetic" / "weatherAUS_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print(f"written {out}")
