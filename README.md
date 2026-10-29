# Temporal Feature Engineering and Data-Quality Treatments for LightGBM-Based Rainfall Occurrence Classification

Code, data partitions, configurations, and experimental outputs accompanying the manuscript

> P. A. P. Dana Wijaya, G. A. Pradipta, and R. R. Huizen, "Evaluating Temporal Feature Engineering and Data-Quality Treatments for LightGBM-Based Rainfall Occurrence Classification," *Journal of Embedded System Security and Intelligent Systems (JESSI)*, [year], [in press].

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

The study evaluates how much temporal lag features, categorical-aware oversampling (SMOTENC), and three data-quality treatments (IQR capping, Z-score capping, Isolation Forest filtering) contribute to LightGBM performance for **retrospective same-day rainfall occurrence classification** (`RainToday`) on the WeatherAUS dataset. It is not a forecasting study.

| Code | Configuration | Features |
|---|---|---|
| A1 | Baseline LightGBM | 18 |
| A2 | LightGBM + Temporal Lag | 51 |
| A3 | LightGBM + SMOTENC | 18 |
| A4 | LightGBM + Temporal Lag + SMOTENC | 51 |
| A5 | LightGBM + Temporal Lag + Isolation Forest | 51 |
| A6 | LightGBM + Temporal Lag + Isolation Forest + SMOTENC | 51 |
| A7 | LightGBM + Temporal Lag + IQR Capping + SMOTENC | 51 |
| A8 | LightGBM + Temporal Lag + Z-score Capping + SMOTENC | 51 |

Evaluation protocols: a **chronological holdout** (primary) and **five stratified random partitions** (supplementary, seeds 42, 52, 62, 72, 82).

## Repository structure

```
configs/            base.yaml (all settings), scenarios/A1-A8.yaml, sensitivity.yaml
src/rainfall_lgbm/  pipeline implementation
scripts/            command-line entry points (see "Reproducing the results")
data/raw/           place weatherAUS.csv here (not redistributed, see data/README_data.md)
data/splits/        partitions stored as Location + Date keys
notebooks/          run_in_colab.ipynb + the original Colab notebook used for the manuscript
results/            metrics, test predictions, tables, statistical outputs
tests/              unit tests and an end-to-end smoke test on synthetic data
```

## Methodological safeguards

* **No target leakage:** `Rainfall`, `RainToday`, and `RainTomorrow` are excluded from predictors. `Location` and `Date` are metadata only.
* **Lag features on exact calendar days:** lag-1 to lag-3 of 11 variables (33 features) are generated once on the full dataset, sorted by Location and Date, **before partitioning**. A lag whose calendar date is absent stays missing; it is never filled with the previous available row. Only earlier dates are used, so observations at partition boundaries may legitimately use values from the preceding period.
* **Training-only statistics:** median/mode imputation, categorical mappings, IQR/Z-score bounds, Isolation Forest, and SMOTENC are fitted on the training subset only. Isolation Forest removes training rows only; validation and test data are never removed.
* **Validation-based selection:** configurations are compared on validation macro F1. Test data are never used for selection. A7 is the *validation-selected* configuration, not a generally superior one.
* **Four distinct data operations:** anomaly detection, outlier removal, extreme-value capping, and error correction. The last one is not evaluated.

## Installation

Python 3.12 or 3.13. The manuscript used Python 3.13.16 on Google Colab (Linux x86_64).

```bash
git clone https://github.com/<username>/rainfall-lightgbm-temporal.git
cd rainfall-lightgbm-temporal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download `weatherAUS.csv` from Kaggle (<https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package>) and place it in `data/raw/`. Then verify it:

```bash
python scripts/verify_data.py
```

The script checks the file against the audit reported in the manuscript (145,460 rows; 142,199 labelled rows; 49 locations; 143 intervals longer than one day) and prints its SHA-256 checksum. See `data/README_data.md`.

## Reproducing the results

```bash
bash scripts/run_all.sh
```

or step by step:

| Step | Command | Output | Manuscript |
|---|---|---|---|
| 1 | `python scripts/verify_data.py` | `results/diagnostics/data_audit.json` | Dataset section |
| 2 | `python scripts/make_splits.py` | `data/splits/*.csv` | Table 2 |
| 3 | `python scripts/run_experiments.py --protocol all` | metrics, predictions, Tables 4-6 | Tables 4, 5, 6 |
| 4 | `python scripts/run_bootstrap.py` | `results/tables/table8_block_bootstrap.csv` | Table 8 |
| 5 | `python scripts/run_wilcoxon.py` | `results/wilcoxon/wilcoxon_holm.csv` | Statistical reliability |
| 6 | `python scripts/run_sensitivity.py` | `results/tables/table7_sensitivity.csv` | Table 7 |
| 7 | `python scripts/run_error_analysis.py` | Table 9 and recall by location | Table 9 |

The 48 principal runs (8 chronological + 40 stratified) take roughly [x] hours on a 2-core Colab runtime.

### Key results reported in the manuscript

| Scenario | Validation macro F1 | Test macro F1 (chronological) | Mean test macro F1 (stratified) |
|---|---|---|---|
| A1 | – | 0.7727 | 0.7746 ± 0.0026 |
| A2 | – | 0.8225 | 0.8258 ± 0.0024 |
| A6 | – | 0.8229 | 0.8251 ± 0.0016 |
| A7 (validation-selected) | 0.8093 | 0.8211 | 0.8260 ± 0.0018 |

Small numerical differences can occur across library versions and hardware because of floating-point and multithreading effects in LightGBM and SMOTENC.

**Status of this reference implementation.** The scripts were written from the Method section and reproduce the data audit, all partition sizes, and the SMOTENC class counts exactly, and the chronological test macro F1 of A7 within 0.0001. Validation macro F1 values differ from the manuscript by up to about 0.004, so the original Colab notebook in `notebooks/` remains the authoritative record of the reported numbers.

## Statistical notes

* **Paired weekly block bootstrap** (primary uncertainty estimate): 64 calendar-week blocks (Monday-Sunday) of the chronological test period, 1,500 resamples, seed 2026, identical blocks for both models, percentile 95% CI. It does not model spatial dependence between stations or dependence across weeks.
* **Wilcoxon signed-rank tests with Holm correction** on the five stratified partitions are **exploratory**. With five pairs, the smallest attainable exact two-sided p-value is 2/2^5 = 0.0625, so significance at α = 0.05 is unattainable by design. The partitions also overlap and are not independent replications.
* The stratified partitions can place temporally adjacent observations of the same station in both training and test subsets. They describe stability under resampling of the same period, not generalization to future periods.

## Tests

```bash
pytest tests/test_units.py        # lag construction, Wilcoxon minimum p-value, Holm
bash tests/run_smoke_test.sh      # full pipeline on random synthetic data (no real data needed)
```

## Citation

Please cite the article and this archive (see `CITATION.cff`):

> P. A. P. Dana Wijaya, G. A. Pradipta, and R. R. Huizen, "Code and results for: Evaluating Temporal Feature Engineering and Data-Quality Treatments for LightGBM-Based Rainfall Occurrence Classification," Zenodo, v1.0.0, 2026, doi: 10.5281/zenodo.XXXXXXX.

## License

Code: MIT License (`LICENSE`). Results, tables, and partition files: CC BY 4.0 (`results/LICENSE`). The WeatherAUS dataset is not redistributed; its terms are set by its providers.

## Contact

Putu Agus Putra Dana Wijaya, Institut Teknologi dan Bisnis STIKOM Bali, 242012014@stikom-bali.ac.id
