#!/usr/bin/env bash
# Full reproduction pipeline. Usage: bash scripts/run_all.sh [--config path/to/override.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/verify_data.py "$@"
python scripts/make_splits.py "$@"
python scripts/run_experiments.py --protocol all "$@"
python scripts/run_bootstrap.py "$@"
python scripts/run_wilcoxon.py "$@"
python scripts/run_sensitivity.py "$@"
python scripts/run_error_analysis.py "$@"
echo "Done. Tables are in results/tables/."
