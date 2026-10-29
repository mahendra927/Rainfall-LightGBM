"""Create and store all partitions as Location + Date keys in data/splits/."""
from _common import config_from, out_dir, parser, write_json
from rainfall_lgbm.data import load_weatheraus
from rainfall_lgbm.splits import chronological_split, save_split, stratified_split

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    cfg = config_from(args)
    df = load_weatheraus(cfg)
    summary = {}
    sub = chronological_split(df, cfg)
    save_split(df, sub, "chronological", cfg)
    summary["chronological"] = {s: {"n": int((sub == s).sum()), "yes": int(df.loc[sub == s, "y"].sum()),
                                    "date_min": str(df.loc[sub == s, "Date"].min().date()),
                                    "date_max": str(df.loc[sub == s, "Date"].max().date())}
                                for s in ("train", "validation", "test")}
    for seed in cfg["stratified"]["seeds"]:
        sub = stratified_split(df, seed, cfg)
        save_split(df, sub, f"stratified_seed{seed}", cfg)
        summary[f"stratified_seed{seed}"] = {s: int((sub == s).sum()) for s in ("train", "validation", "test")}
    write_json(summary, out_dir(cfg, "diagnostics") / "split_summary.json")
    print(summary)
