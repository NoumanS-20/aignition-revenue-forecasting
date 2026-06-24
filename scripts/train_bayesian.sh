#!/usr/bin/env bash
# Build the full hierarchical Bayesian model and verify the scored pipeline.
# Run on Linux/macOS/WSL/Colab (PyMC needs a working C/C++ toolchain).
#
# Usage:
#   bash scripts/train_bayesian.sh [DATA_DIR] [DRAWS]
# Examples:
#   bash scripts/train_bayesian.sh data 1000     # real data in ./data
#   bash scripts/train_bayesian.sh data/sample 500
set -euo pipefail

DATA_DIR="${1:-data}"
DRAWS="${2:-1000}"

echo "==> Installing training dependencies"
python -m pip install --quiet -r requirements-train.txt
python -m pip install --quiet -e .

# If no C compiler is available, fall back to pytensor's pure-Python path (slow).
if ! command -v gcc >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
  echo "==> No C compiler found; disabling pytensor C compilation (slower)"
  export PYTENSOR_FLAGS="cxx="
fi

echo "==> Fitting Bayesian model from ${DATA_DIR} (draws=${DRAWS})"
python train.py --data-dir "$DATA_DIR" --out pickle/model.pkl --method bayesian --draws "$DRAWS"

echo "==> Verifying scored pipeline against the new model"
bash run.sh "$DATA_DIR" pickle/model.pkl output/predictions.csv
head -5 output/predictions.csv

echo "==> Done. Commit the rebuilt pickle/model.pkl:"
echo "    git add pickle/model.pkl && git commit -m 'feat: Bayesian-fitted model'"
