#!/usr/bin/env bash
# Grid search: TCN variants on BCI-IV (3 subjects)
#
# Variants:
#   v1:  blocks=6, lr=8.42e-5  (baseline — same as Miller run)
#   v3a: blocks=6, lr=3e-5     (slower learning)
#   v3b: blocks=4, lr=8.42e-5  (less capacity)
#   v3c: blocks=4, lr=3e-5     (combined)
#
# 4 variants × 3 subjects = 12 runs, parallelized across 2 GPUs.
# Expected: ~30-60 min total.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python
GPUS=(0 1)
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
        > "results/grid_tcn_${label}_s${subject}.log" 2>&1 &
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
echo " TCN Grid Search on BCI-IV (3 subjects)"
echo " Variants: v1, v3a, v3b, v3c"
echo " GPUs: ${GPUS[*]}"
echo "=============================================="
echo ""

# Launch all 12 runs, 2 at a time
for subject in 1 2 3; do
    for variant in v1 v3a v3b v3c; do
        wait_for_slot

        case $variant in
            v1)  config="configs/bci4_tcn.yaml";    lr=8.42e-5 ;;
            v3a) config="configs/bci4_tcn.yaml";    lr=3e-5    ;;
            v3b) config="configs/bci4_tcn_4b.yaml"; lr=8.42e-5 ;;
            v3c) config="configs/bci4_tcn_4b.yaml"; lr=3e-5    ;;
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
    for variant in v1 v3a v3b v3c; do
        log="results/grid_tcn_${variant}_s${subject}.log"
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
