"""Exploratory paired Wilcoxon signed-rank tests with Holm correction (five stratified seeds)."""
from _common import ROOT, config_from, out_dir, parser
import pandas as pd
from rainfall_lgbm.stats import wilcoxon_holm

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    cfg = config_from(args)
    m = pd.read_csv(out_dir(cfg) / "stratified" / "metrics_by_seed.csv")
    scores = m.pivot(index="seed", columns="scenario", values="test_macro_f1")
    res = wilcoxon_holm(scores, cfg["wilcoxon"]["reference"], cfg["wilcoxon"]["alpha"])
    res.to_csv(out_dir(cfg, "wilcoxon") / "wilcoxon_holm.csv", index=False)
    print(res.round(4).to_string(index=False))
    print(f"\nNote: with n = {len(scores)} pairs the smallest attainable exact two-sided p-value is "
          f"{2 / 2 ** len(scores):.4f}; these tests are exploratory only.")
