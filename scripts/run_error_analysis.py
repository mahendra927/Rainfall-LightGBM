"""False negatives by rainfall intensity (Table 9) and rainfall recall by location."""
import pandas as pd

from _common import ROOT, config_from, out_dir, parser
from rainfall_lgbm.error_analysis import false_negatives_by_intensity, recall_by_location

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    cfg = config_from(args)
    e = cfg["error_analysis"]
    pred = pd.read_csv(out_dir(cfg) / "predictions" / f"chronological_{e['scenario']}.csv.gz")
    t9 = false_negatives_by_intensity(pred, e["intensity_bins"])
    t9.to_csv(out_dir(cfg, "tables") / "table9_false_negatives_by_intensity.csv", index=False)
    loc = recall_by_location(pred)
    loc.to_csv(out_dir(cfg, "tables") / "recall_by_location.csv", index=False)
    print(t9.to_string(index=False))
    print(loc.head(5).to_string(index=False))
