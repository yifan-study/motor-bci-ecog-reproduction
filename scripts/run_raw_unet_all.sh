#!/bin/bash
# Run raw U-Net on all 9 Miller patients across 4 GPUs.
# Usage: bash scripts/run_raw_unet_all.sh

set -e
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python
CONFIG=configs/fingerflex_raw_unet.yaml
SEED=123

PATIENTS=(bp cc ht jc jp mv wc wm zt)
GPUS=(0 1 2 3)
N_GPUS=${#GPUS[@]}

PIDS=()
for i in "${!PATIENTS[@]}"; do
    p=${PATIENTS[$i]}
    gpu=${GPUS[$((i % N_GPUS))]}
    echo "[$(date +%H:%M:%S)] Starting patient $p on GPU $gpu"
    $PYTHON scripts/train.py \
        --config $CONFIG \
        --patient "$p" \
        --gpu "$gpu" \
        --seed $SEED \
        --no-wandb \
        > "results/raw_unet_${p}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All 9 patients launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."

FAILED=0
for i in "${!PATIENTS[@]}"; do
    p=${PATIENTS[$i]}
    if wait ${PIDS[$i]}; then
        echo "[$(date +%H:%M:%S)] Patient $p DONE"
    else
        echo "[$(date +%H:%M:%S)] Patient $p FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All 9 patients completed successfully."
else
    echo "$FAILED patient(s) failed. Check logs in results/raw_unet_*.log"
fi

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for p in "${PATIENTS[@]}"; do
    LOG="results/raw_unet_${p}.log"
    if [ -f "$LOG" ]; then
        echo "--- Patient $p ---"
        grep -A3 "TEST EVALUATION" "$LOG" | tail -3
        echo ""
    fi
done
