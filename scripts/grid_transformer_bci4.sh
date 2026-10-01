#!/usr/bin/env bash
# Grid search: Transformer variants on BCI-IV (3 subjects)
#
# Variants:
#   A_hi:  d=64,  L=4, k=3, lr=8.42e-5  (match U-Net LR)
#   A_lo:  d=64,  L=4, k=3, lr=3e-5     (lower, typical for transformers)
#   B_hi:  d=128, L=2, k=1, lr=8.42e-5
#   B_lo:  d=128, L=2, k=1, lr=3e-5
#
# 4 variants x 3 subjects = 12 runs, parallelized across 2 GPUs.
# Expected: ~60-90 min total.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python
GPUS=(3)
PIDS=()
GPU_IDX=0

run_job() {
    local label="$1" config="$2" subject="$3" lr="$4" gpu="$5"
    echo "[launch] $label — subject=$subject, lr=$lr, gpu=$gpu"
    $PYTHON scripts/train.py \
        --config "$config" \
        --subject "$subject" \
        --lr "$lr" \
        --gpu "$gpu" \
        --seed 123 \
        --no-wandb \
        > "results/grid_transformer_${label}_s${subject}.log" 2>&1 &
    PIDS+=($!)
}

wait_for_slot() {
    # Wait until at least one GPU slot is free
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
    # Find a free GPU (round-robin)
    GPU_IDX=$(( (GPU_IDX + 1) % ${#GPUS[@]} ))
}

echo "=============================================="
echo " Transformer Grid Search on BCI-IV (3 subjects)"
echo " Variants: A_hi, A_lo, B_hi, B_lo"
echo " GPUs: ${GPUS[*]}"
echo "=============================================="
echo ""

# Launch all 12 runs, 2 at a time
for subject in 1 2 3; do
    for variant in A_hi A_lo B_hi B_lo; do
        wait_for_slot

        case $variant in
            A_hi) config="configs/bci4_transformer_A.yaml"; lr=8.42e-5 ;;
            A_lo) config="configs/bci4_transformer_A.yaml"; lr=3e-5    ;;
            B_hi) config="configs/bci4_transformer_B.yaml"; lr=8.42e-5 ;;
            B_lo) config="configs/bci4_transformer_B.yaml"; lr=3e-5    ;;
        esac

        gpu=${GPUS[$GPU_IDX]}
        run_job "$variant" "$config" "$subject" "$lr" "$gpu"
    done
done

echo ""
echo "All jobs launched. Waiting for completion..."
wait
echo ""
echo "=============================================="
echo " All 12 runs complete. Parsing results..."
echo "=============================================="

# Parse results
echo ""
printf "%-8s %-8s %-10s %-10s %-10s\n" "Variant" "Subject" "Val r" "Test r" "Params"
printf "%-8s %-8s %-10s %-10s %-10s\n" "-------" "-------" "-------" "-------" "-------"

for subject in 1 2 3; do
    for variant in A_hi A_lo B_hi B_lo; do
        log="results/grid_transformer_${variant}_s${subject}.log"
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

# Compute per-variant mean test r
echo ""
echo "--- Per-variant mean test r ---"
for variant in A_hi A_lo B_hi B_lo; do
    sum=0; count=0
    for subject in 1 2 3; do
        log="results/grid_transformer_${variant}_s${subject}.log"
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
