#!/usr/bin/env bash
# Dropout tuning: drop=0.2 (GPU 0) + drop=0.3 (GPU 1) on BCI-IV
# Base: d256, ff=512, drop=0.1 (mean 0.517)

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python

echo "=============================================="
echo " Dropout tuning on BCI-IV"
echo " Base: d256, ff=512, drop=0.1 → mean 0.517"
echo " GPU 0: drop=0.2  |  GPU 1: drop=0.3"
echo "=============================================="
echo ""

# GPU 0: dropout=0.2
(
    for subject in 1 2 3; do
        echo "[GPU0] drop02 — subject=$subject"
        $PYTHON scripts/train.py \
            --config configs/bci4_transformer_d256_ff512_drop02.yaml \
            --subject "$subject" \
            --gpu 0 \
            --seed 123 \
            --no-wandb \
            > "results/tune_d256_drop02_s${subject}.log" 2>&1
    done
) &

# GPU 1: dropout=0.3
(
    for subject in 1 2 3; do
        echo "[GPU1] drop03 — subject=$subject"
        $PYTHON scripts/train.py \
            --config configs/bci4_transformer_d256_ff512_drop03.yaml \
            --subject "$subject" \
            --gpu 1 \
            --seed 123 \
            --no-wandb \
            > "results/tune_d256_drop03_s${subject}.log" 2>&1
    done
) &

echo "All jobs launched. Waiting for completion..."
wait
echo ""
echo "=============================================="
echo " Results:"
echo "=============================================="

echo ""
printf "%-14s %-8s %-10s %-10s %-10s\n" "Variant" "Subject" "Val r" "Test r" "Params"
printf "%-14s %-8s %-10s %-10s %-10s\n" "-------------" "-------" "-------" "-------" "-------"

for variant in d256_drop02 d256_drop03; do
    for subject in 1 2 3; do
        log="results/tune_${variant}_s${subject}.log"
        if [ -f "$log" ]; then
            val_r=$(grep "Best val r:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
            test_r=$(grep "Test r (average):" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
            params=$(grep "Model params:" "$log" 2>/dev/null | awk '{print $NF}' || echo "N/A")
            printf "%-14s %-8s %-10s %-10s %-10s\n" "$variant" "$subject" "$val_r" "$test_r" "$params"
        else
            printf "%-14s %-8s %-10s %-10s %-10s\n" "$variant" "$subject" "MISSING" "MISSING" "MISSING"
        fi
    done
done

echo ""
echo "--- Per-variant mean test r ---"
for variant in d256_drop02 d256_drop03; do
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
        printf "%-14s mean=%.4f (n=%d)\n" "$variant" "$mean" "$count"
    else
        printf "%-14s no results\n" "$variant"
    fi
done
echo ""
echo "Baseline: d256 ff512 drop=0.1 mean=0.5168, U-Net mean=0.5955"
