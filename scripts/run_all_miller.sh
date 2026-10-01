#!/usr/bin/env bash
# Train Lomtev U-Net on all 9 Miller patients sequentially.
# Usage: bash scripts/run_all_miller.sh

set -euo pipefail

cd "$(dirname "$0")/.."

for patient in bp cc ht jc jp mv wc wm zt; do
    echo "=============================================="
    echo "  Training patient: $patient"
    echo "=============================================="
    python scripts/train_lomtev_miller.py --patient "$patient" --epochs 40 --gpu 0 --seed 123
    echo ""
done

echo "All 9 patients complete."
