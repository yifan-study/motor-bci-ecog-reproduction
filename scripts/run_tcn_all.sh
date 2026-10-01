#!/usr/bin/env bash
# Train TCN on all 9 Miller patients sequentially.
# Usage: bash scripts/run_tcn_all.sh

set -euo pipefail

cd "$(dirname "$0")/.."

for patient in bp cc ht jc jp mv wc wm zt; do
    echo "=============================================="
    echo "  Training TCN — patient: $patient"
    echo "=============================================="
    /mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python scripts/train.py --config configs/fingerflex_tcn.yaml --patient "$patient" --epochs 40 --gpu 0 --seed 123
    echo ""
done

echo "All 9 patients complete."
