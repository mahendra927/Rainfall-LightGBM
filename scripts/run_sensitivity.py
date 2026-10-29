"""Exploratory, validation-only sensitivity analysis (Table 7). Test data are not inspected."""
import pandas as pd

from _common import ROOT, config_from, out_dir, parser
from rainfall_lgbm.config import deep_update, load_scenario, load_yaml
from rainfall_lgbm.data import add_lags, load_weatheraus
from rainfall_lgbm.pipeline import run_scenario
from rainfall_lgbm.splits import get_split

MAIN = {"lag_depth": 3, "isolation_forest_contamination": 0.08, "zscore_threshold": 3.0, "max_estimators": 900}

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    cfg = config_from(args)
    sens = load_yaml(ROOT / "configs" / "sensitivity.yaml")
    max_lag = max(sens["lag_depth"]["values"])
    df = add_lags(load_weatheraus(cfg), cfg, list(range(1, max_lag + 1)))
    subset = get_split(df, cfg, "chronological")
    seed = cfg["chronological"]["seed"]
    rows = []
    for param, spec in sens.items():
        sc = load_scenario(spec["scenario"])
        for v in spec["values"]:
            c, lags = cfg, None
            if param == "lag_depth":
                lags = list(range(1, int(v) + 1))
            elif param == "isolation_forest_contamination":
                c = deep_update(cfg, {"isolation_forest": {"contamination": v}})
            elif param == "zscore_threshold":
                c = deep_update(cfg, {"zscore": {"threshold": v}})
            elif param == "max_estimators":
                c = deep_update(cfg, {"lightgbm": {"n_estimators": int(v)}})
            r = run_scenario(df, subset, sc, c, seed, lags=lags, keep_predictions=False)
            rows.append({"parameter": param, "scenario": spec["scenario"], "value": v,
                         "main_setting": MAIN[param], "validation_macro_f1": r["validation"]["macro_f1"]})
            print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(out_dir(cfg, "sensitivity") / "sensitivity_all_runs.csv", index=False)
    summ = []
    for param, g in out.groupby("parameter", sort=False):
        best = g.loc[g["validation_macro_f1"].idxmax()]
        main = g[g["value"] == g["main_setting"]].iloc[0]
        summ.append({"parameter": param, "scenario": best["scenario"],
                     "values_evaluated": ", ".join(f"{x:g}" for x in g["value"]),
                     "main_setting": main["value"], "validation_macro_f1_main": round(main["validation_macro_f1"], 4),
                     "best_observed_value": best["value"], "validation_macro_f1_best": round(best["validation_macro_f1"], 4)})
    pd.DataFrame(summ).to_csv(out_dir(cfg, "tables") / "table7_sensitivity.csv", index=False)
