"""Verify the downloaded WeatherAUS file and reproduce the dataset audit.

Expected values reported in the manuscript:
  145,460 rows, 23 attributes, 3,261 rows without RainToday, 142,199 labelled rows
  (110,319 No / 31,880 Yes), 49 locations, 2007-11-01 to 2017-06-25,
  0 duplicated Location-Date pairs, 143 intervals longer than one calendar day.
"""
from _common import config_from, out_dir, parser, write_json
from rainfall_lgbm.data import audit

EXPECTED = {"rows_total": 145460, "columns_total": 23, "rows_missing_target": 3261, "rows_labelled": 142199,
            "locations": 49, "duplicated_location_date_pairs": 0, "intervals_longer_than_one_day": 143}

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    cfg = config_from(args)
    res = audit(cfg["data"]["path"], cfg)
    write_json(res, out_dir(cfg, "diagnostics") / "data_audit.json")
    for k, v in res.items():
        flag = ""
        if k in EXPECTED:
            flag = "  OK" if v == EXPECTED[k] else f"  MISMATCH (expected {EXPECTED[k]})"
        print(f"{k:35s} {v}{flag}")
