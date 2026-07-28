#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$SCRIPT_DIR"

mkdir -p Data/model Data/hyperpara Data/lstmdraw Data/trainerr Figure/lstmtrain Figure/lstmdraw Figure/flux

shopt -s nullglob
configs=(Data/hyperpara/paras_*.yaml)

if [ ${#configs[@]} -eq 0 ]; then
  echo "No Data/hyperpara/paras_*.yaml config files found."
  exit 1
fi

for config_path in "${configs[@]}"; do
  echo "Training sun imputer with ${config_path}"
  "$PYTHON_BIN" lstm_train_sunpara.py --config "$config_path"
done

"$PYTHON_BIN" lstm_bestmodel_sunpara.py
