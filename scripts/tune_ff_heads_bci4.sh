#!/usr/bin/env bash
# Parallel tuning: ff=1024 (GPU 0) + 8-heads (GPU 1) on BCI-IV
# Base: L=4, ff=512, 4 heads (mean 0.499)

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python

echo "=============================================="
echo " Parallel tuning: ff=1024 + 8-heads on BCI-IV"
echo " Base: L=4, ff=512, 4h → mean 0.499"
echo " GPU 0: ff=1024  |  GPU 1: 8-heads"
echo "=============================================="
echo ""

# GPU 0: ff=1024
(
    for subject in 1 2 3; do
        echo "[GPU0] ff1024 — subject=$subject"
        $PYTHON scripts/train.py \
            --config configs/bci4_transformer_B_L4_ff1024.yaml \
            --subject "$subject" \
            --gpu 0 \
            --seed 123 \
            --no-wandb \
            > "results/tune_L4_ff1024_s${subject}.log" 2>&1
    done
) &

# GPU 1: 8 heads
(
    for subject in 1 2 3; do
        echo "[GPU1] h8 — subject=$subject"
        $PYTHON scripts/train.py \
            --config configs/bci4_transformer_B_L4_ff512_h8.yaml \
            --subject "$subject" \
            --gpu 1 \
            --seed 123 \
            --no-wandb \
            > "results/tune_L4_ff512_h8_s${subject}.log" 2>&1
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

for variant in L4_ff1024 L4_ff512_h8; do
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
for variant in L4_ff1024 L4_ff512_h8; do
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
echo "Baseline: L=4 ff=512 4h mean=0.4990, U-Net mean=0.5955"
