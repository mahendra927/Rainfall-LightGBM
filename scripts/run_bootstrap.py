"""Paired weekly block bootstrap on chronological test predictions (Table 8)."""
import pandas as pd

from _common import ROOT, config_from, out_dir, parser
from rainfall_lgbm.config import SCENARIOS
from rainfall_lgbm.stats import weekly_block_bootstrap

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    cfg = config_from(args)
    b = cfg["bootstrap"]
    ref = b["reference"]
    load = lambda c: pd.read_csv(out_dir(cfg) / "predictions" / f"chronological_{c}.csv.gz")
    pa = load(ref)
    rows = []
    for c in [s for s in SCENARIOS if s != ref and s != "A3"]:
        r = weekly_block_bootstrap(pa, load(c), b["n_resamples"], b["seed"], b["confidence"])
        rows.append({"comparison": f"{ref} − {c}", **r})
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(out_dir(cfg, "bootstrap") / "bootstrap_results.csv", index=False)
    out[["comparison", "difference", "ci_low", "ci_high"]].round(4).to_csv(
        out_dir(cfg, "tables") / "table8_block_bootstrap.csv", index=False)
