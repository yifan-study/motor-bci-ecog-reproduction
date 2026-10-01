#!/usr/bin/env bash
# Parallel tuning: L=5 (GPU 0) + L=4 ff=512 (GPU 1) on BCI-IV (3 subjects)
# L=5: tests if deeper still helps (trend: L2→L4 was monotonic improvement)
# ff=512: tests wider feedforward MLP with best depth (L=4)

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python
PIDS=()

run_job() {
    local label="$1" config="$2" subject="$3" gpu="$4"
    echo "[launch] $label — subject=$subject, gpu=$gpu"
    $PYTHON scripts/train.py \
        --config "$config" \
        --subject "$subject" \
        --gpu "$gpu" \
        --seed 123 \
        --no-wandb \
        > "results/tune_${label}_s${subject}.log" 2>&1 &
    PIDS+=($!)
}

echo "=============================================="
echo " Parallel tuning: L=5 + ff=512 on BCI-IV"
echo " GPU 0: L=5 (3 subjects sequential)"
echo " GPU 1: L=4+ff512 (3 subjects sequential)"
echo "=============================================="
echo ""

# GPU 0: L=5 subjects sequentially
(
    for subject in 1 2 3; do
        echo "[GPU0] L5 — subject=$subject"
        $PYTHON scripts/train.py \
            --config configs/bci4_transformer_B_L5.yaml \
            --subject "$subject" \
            --gpu 0 \
            --seed 123 \
            --no-wandb \
            > "results/tune_L5_s${subject}.log" 2>&1
    done
) &
PIDS+=($!)

# GPU 1: L=4 ff=512 subjects sequentially
(
    for subject in 1 2 3; do
        echo "[GPU1] L4_ff512 — subject=$subject"
        $PYTHON scripts/train.py \
            --config configs/bci4_transformer_B_L4_ff512.yaml \
            --subject "$subject" \
            --gpu 1 \
            --seed 123 \
            --no-wandb \
            > "results/tune_L4_ff512_s${subject}.log" 2>&1
    done
) &
PIDS+=($!)

echo "All jobs launched. Waiting for completion..."
wait
echo ""
echo "=============================================="
echo " Results:"
echo "=============================================="

echo ""
printf "%-12s %-8s %-10s %-10s %-10s\n" "Variant" "Subject" "Val r" "Test r" "Params"
printf "%-12s %-8s %-10s %-10s %-10s\n" "-----------" "-------" "-------" "-------" "-------"

for variant in L5 L4_ff512; do
    for subject in 1 2 3; do
        log="results/tune_${variant}_s${subject}.log"
        if [ -f "$log" ]; then
            val_r=$(grep "Best val r:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
            test_r=$(grep "Test r (average):" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
            params=$(grep "Model params:" "$log" 2>/dev/null | awk '{print $NF}' || echo "N/A")
            printf "%-12s %-8s %-10s %-10s %-10s\n" "$variant" "$subject" "$val_r" "$test_r" "$params"
        else
            printf "%-12s %-8s %-10s %-10s %-10s\n" "$variant" "$subject" "MISSING" "MISSING" "MISSING"
        fi
    done
done

echo ""
echo "--- Per-variant mean test r ---"
for variant in L5 L4_ff512; do
    sum=0; count=0
    for subject in 1 2 3; do
        log="results/tune_${variant}_s${subject}.log"
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
        printf "%-12s mean=%.4f (n=%d)\n" "$variant" "$mean" "$count"
    else
        printf "%-12s no results\n" "$variant"
    fi
done
echo ""
echo "Baselines: L=4 mean=0.4678, U-Net mean=0.5955"
