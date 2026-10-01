#!/usr/bin/env bash
# Parallel: drop=0.15 (GPU 1) + hybrid transformer (GPU 2) on BCI-IV

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python

echo "=============================================="
echo " drop=0.15 (GPU 1) + hybrid transformer (GPU 2)"
echo "=============================================="
echo ""

# GPU 1: d256 drop=0.15
(
    for subject in 1 2 3; do
        echo "[GPU1] drop015 — subject=$subject"
        $PYTHON scripts/train.py \
            --config configs/bci4_transformer_d256_ff512_drop015.yaml \
            --subject "$subject" \
            --gpu 1 \
            --seed 123 \
            --no-wandb \
            > "results/tune_d256_drop015_s${subject}.log" 2>&1
    done
) &

# GPU 2: hybrid transformer
(
    for subject in 1 2 3; do
        echo "[GPU2] hybrid — subject=$subject"
        $PYTHON scripts/train.py \
            --config configs/bci4_hybrid_transformer.yaml \
            --subject "$subject" \
            --gpu 2 \
            --seed 123 \
            --no-wandb \
            > "results/tune_hybrid_s${subject}.log" 2>&1
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

for variant in d256_drop015 hybrid; do
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
for variant in d256_drop015 hybrid; do
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
echo "Baselines: d256 drop=0.1 mean=0.517, U-Net mean=0.597"
