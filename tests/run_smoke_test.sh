#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (does not need the real WeatherAUS file).
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/make_synthetic_data.py
C="--config tests/quick_config.yaml"
python scripts/make_splits.py $C
python scripts/run_experiments.py --protocol all $C
python scripts/run_bootstrap.py $C
python scripts/run_wilcoxon.py $C
python scripts/run_sensitivity.py $C
python scripts/run_error_analysis.py $C
echo "Smoke test finished: outputs in tests/_synthetic/results/"
