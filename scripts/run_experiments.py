"""Run the principal experiments (A1-A8) under the chronological and/or stratified protocol.

Outputs
  results/chronological/metrics.csv                 all metrics, validation + test
  results/stratified/metrics_by_seed.csv            40 runs (8 scenarios x 5 seeds)
  results/predictions/<protocol>_<scenario>[_seed].csv.gz   test predictions
  results/tables/table4_chronological_test.csv      Table 4
  results/tables/table5_stratified_summary.csv      Table 5
  results/tables/table6_isolation_forest.csv        Table 6 (+ stratified removal rates)
"""
import pandas as pd

from _common import config_from, out_dir, parser
from rainfall_lgbm.config import SCENARIOS, load_scenario
from rainfall_lgbm.data import add_lags, load_weatheraus
from rainfall_lgbm.pipeline import flatten, run_scenario
from rainfall_lgbm.splits import get_split


def prepare(cfg):
    return add_lags(load_weatheraus(cfg), cfg, cfg["lag_features"]["lags"])


def run_protocol(df, cfg, protocol, scenarios):
    seeds = [cfg["chronological"]["seed"]] if protocol == "chronological" else cfg["stratified"]["seeds"]
    rows = []
    for seed in seeds:
        subset = get_split(df, cfg, protocol, None if protocol == "chronological" else seed)
        for code in scenarios:
            res = run_scenario(df, subset, load_scenario(code), cfg, seed)
            suffix = "" if protocol == "chronological" else f"_seed{seed}"
            res["test_predictions"].to_csv(out_dir(cfg, "predictions") / f"{protocol}_{code}{suffix}.csv.gz",
                                           index=False, compression="gzip")
            rows.append(flatten(res))
            print(f"[{protocol}] {code} seed={seed}  val macroF1={res['validation']['macro_f1']:.4f}  "
                  f"test macroF1={res['test']['macro_f1']:.4f}  ({res['runtime_s']} s)")
    return pd.DataFrame(rows)


def tables(protocol, m, cfg):
    t = out_dir(cfg, "tables")
    if protocol == "chronological":
        cols = ["test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_recall_yes"]
        tab = m[["scenario", "validation_macro_f1"] + cols].round(4)
        tab.to_csv(t / "table4_chronological_test.csv", index=False)
        sel = m.loc[m["validation_macro_f1"].idxmax(), "scenario"]
        print(f"Validation-selected configuration: {sel}")
        f = m[m["scenario"].isin(["A5", "A6"])]
        if len(f):
            r = f.iloc[0]
            pd.DataFrame({
                "component": ["Before filtering", "Removed", "After filtering", "Class-specific removal rate"],
                "No": [r.train_no_before, r.if_removed_no, r.train_no_before - r.if_removed_no,
                       r.if_removed_no / r.train_no_before],
                "Yes": [r.train_yes_before, r.if_removed_yes, r.train_yes_before - r.if_removed_yes,
                        r.if_removed_yes / r.train_yes_before],
                "Total": [r.n_train_before, r.if_removed_total, r.n_train_before - r.if_removed_total,
                          r.if_removed_total / r.n_train_before],
            }).to_csv(t / "table6_isolation_forest.csv", index=False)
    else:
        g = m.groupby("scenario")
        tab = pd.DataFrame({
            "macro_f1_mean": g["test_macro_f1"].mean(), "macro_f1_sd": g["test_macro_f1"].std(ddof=1),
            "recall_yes_mean": g["test_recall_yes"].mean(), "accuracy_mean": g["test_accuracy"].mean(),
            "validation_macro_f1_mean": g["validation_macro_f1"].mean(),
        }).round(4).reset_index()
        tab.to_csv(t / "table5_stratified_summary.csv", index=False)
        f = m[m["scenario"].isin(["A5", "A6"])]
        if len(f):
            pd.DataFrame({"removal_rate_no_mean": [(f.if_removed_no / f.train_no_before).mean()],
                          "removal_rate_yes_mean": [(f.if_removed_yes / f.train_yes_before).mean()]}
                         ).to_csv(t / "table6b_isolation_forest_stratified_rates.csv", index=False)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--protocol", choices=["chronological", "stratified", "all"], default="all")
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    args = p.parse_args()
    cfg = config_from(args)
    df = prepare(cfg)
    protocols = ["chronological", "stratified"] if args.protocol == "all" else [args.protocol]
    for prot in protocols:
        m = run_protocol(df, cfg, prot, args.scenarios)
        name = "metrics.csv" if prot == "chronological" else "metrics_by_seed.csv"
        m.to_csv(out_dir(cfg, prot) / name, index=False)
        tables(prot, m, cfg)
