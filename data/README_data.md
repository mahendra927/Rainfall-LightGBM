# Data

## WeatherAUS (not redistributed)

* Source: Kaggle, "Rain in Australia" (weather-dataset-rattle-package), <https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package>
* File: `weatherAUS.csv`, to be placed in `data/raw/weatherAUS.csv`
* Underlying observations: Australian Bureau of Meteorology. Redistribution terms are set by the data providers, so the raw file is not included in this repository.

## Expected file characteristics

| Property | Value |
|---|---|
| Rows | 145,460 |
| Attributes | 23 |
| Rows without `RainToday` | 3,261 |
| Labelled rows used | 142,199 (110,319 No / 31,880 Yes) |
| Locations | 49 |
| Period | 2007-11-01 to 2017-06-25 |
| Duplicated Location-Date pairs | 0 |
| Intervals longer than one calendar day | 143 |
| SHA-256 | `573fd715cd69fcacc4df32024d823b450ae3edaae7e8ff2eeb623adbed424014` |

Run `python scripts/verify_data.py` to check your copy. A different checksum means the Kaggle file has changed; results may then differ from the manuscript.

## Partitions (`data/splits/`)

Each file lists every labelled observation with its subset, keyed by `Location` and `Date` (not by row number, so the files stay valid regardless of how the CSV is read):

| File | Content |
|---|---|
| `chronological.csv` | train: 2007-11-01 to 2015-01-12 (99,546); validation: 2015-01-13 to 2016-04-08 (21,341); test: 2016-04-09 to 2017-06-25 (21,312) |
| `stratified_seed42.csv` ... `stratified_seed82.csv` | stratified 70:15:15 split (99,539 / 21,330 / 21,330) |

These files are produced by `python scripts/make_splits.py` and should be committed after generation.
