#!/usr/bin/env bash
# Depth tuning: L=3 vs L=4 on BCI-IV (3 subjects)
# Base: B_hi (d=128, L=2, heads=4, ff=256, k=1, lr=8.42e-5) → mean 0.444
# 2 variants × 3 subjects = 6 runs, parallelized across 2 GPUs.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python
GPUS=(0 1)
PIDS=()
GPU_IDX=0

run_job() {
    local label="$1" config="$2" subject="$3" gpu="$4"
    echo "[launch] $label — subject=$subject, gpu=$gpu"
    $PYTHON scripts/train.py \
        --config "$config" \
        --subject "$subject" \
        --gpu "$gpu" \
        --seed 123 \
        --no-wandb \
        > "results/tune_depth_${label}_s${subject}.log" 2>&1 &
    PIDS+=($!)
}

wait_for_slot() {
    while [ ${#PIDS[@]} -ge ${#GPUS[@]} ]; do
        local new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        PIDS=("${new_pids[@]}")
        if [ ${#PIDS[@]} -ge ${#GPUS[@]} ]; then
            sleep 10
        fi
    done
    GPU_IDX=$(( (GPU_IDX + 1) % ${#GPUS[@]} ))
}

echo "=============================================="
echo " Depth Tuning: L=3, L=4 on BCI-IV"
echo " Base: B_hi (d=128, L=2) → mean 0.444"
echo " GPUs: ${GPUS[*]}"
echo "=============================================="
echo ""

for subject in 1 2 3; do
    for variant in L3 L4; do
        wait_for_slot
        config="configs/bci4_transformer_B_${variant}.yaml"
        gpu=${GPUS[$GPU_IDX]}
        run_job "$variant" "$config" "$subject" "$gpu"
    done
done

echo ""
echo "All jobs launched. Waiting for completion..."
wait
echo ""
echo "=============================================="
echo " Depth tuning complete. Results:"
echo "=============================================="

echo ""
printf "%-8s %-8s %-10s %-10s %-10s\n" "Variant" "Subject" "Val r" "Test r" "Params"
printf "%-8s %-8s %-10s %-10s %-10s\n" "-------" "-------" "-------" "-------" "-------"

for subject in 1 2 3; do
    for variant in L3 L4; do
        log="results/tune_depth_${variant}_s${subject}.log"
        if [ -f "$log" ]; then
            val_r=$(grep "Best val r:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
            test_r=$(grep "Test r (average):" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
            params=$(grep "Model params:" "$log" 2>/dev/null | awk '{print $NF}' || echo "N/A")
            printf "%-8s %-8s %-10s %-10s %-10s\n" "$variant" "$subject" "$val_r" "$test_r" "$params"
        else
            printf "%-8s %-8s %-10s %-10s %-10s\n" "$variant" "$subject" "MISSING" "MISSING" "MISSING"
        fi
    done
done

echo ""
echo "--- Per-variant mean test r ---"
for variant in L3 L4; do
    sum=0; count=0
    for subject in 1 2 3; do
        log="results/tune_depth_${variant}_s${subject}.log"
        if [ -f "$log" ]; then
            val=$(grep "Test r (average):" "$log" 2>/dev/null | tail -1 | awk '{print $NF}')
            if [ -n "$val" ] && [ "$val" != "N/A" ]; then
                sum=$(echo "$sum + $val" | bc -l)
                count=$((count + 1))
            fi
        fi
    done
    if [ $count -gt 0 ]; then
        mean=$(echo "$sum / $count" | bc -l)
        printf "%-8s mean=%.4f (n=%d)\n" "$variant" "$mean" "$count"
    else
        printf "%-8s no results\n" "$variant"
    fi
done
echo ""
echo "Baseline: B_hi (L=2) mean=0.4444"
