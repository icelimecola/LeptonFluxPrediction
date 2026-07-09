#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

mkdir -p jobs JSUB/error JSUB/output JSUB/JSUBs Data/model Data/hyperpara Data/lstmdraw Figure/lstmtrain Figure/lstmdraw Figure/flux

shopt -s nullglob
configs=(Data/hyperpara/paras_*.yaml)

if [ ${#configs[@]} -eq 0 ]; then
  echo "No Data/hyperpara/paras_*.yaml config files found."
  exit 1
fi

for config_path in "${configs[@]}"; do
  config_file=${config_path##*/}
  i=${config_file#paras_}
  i=${i%.yaml}
  sed "s/NUM/$i/g" jsub_train_sun_imputer.sh > JSUB/JSUBs/jsub_train_sun_imputer_$i.sh
  chmod +x JSUB/JSUBs/jsub_train_sun_imputer_$i.sh
  jsub < JSUB/JSUBs/jsub_train_sun_imputer_$i.sh
  echo "Submitted sun imputer paras_$i.yaml"
done

echo "After all jobs finish, run:"
echo "  python best_sun_imputer.py"
