#!/usr/bin/env bash
# d_model=512 tuning on BCI-IV
# Run ff=1024 and ff=2048 sequentially on one GPU
# With 8 heads (head_dim=64, standard)

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python
GPU=2

echo "=============================================="
echo " d_model=512 tuning on BCI-IV (GPU $GPU)"
echo " ff=1024 then ff=2048, all 3 subjects each"
echo "=============================================="
echo ""

for variant in d512_ff1024 d512_ff2048; do
    config="configs/bci4_transformer_${variant}.yaml"
    for subject in 1 2 3; do
        echo "[launch] $variant — subject=$subject"
        $PYTHON scripts/train.py \
            --config "$config" \
            --subject "$subject" \
            --gpu $GPU \
            --seed 123 \
            --no-wandb \
            > "results/tune_${variant}_s${subject}.log" 2>&1
    done
done

echo ""
echo "=============================================="
echo " Results:"
echo "=============================================="

echo ""
printf "%-14s %-8s %-10s %-10s %-10s\n" "Variant" "Subject" "Val r" "Test r" "Params"
printf "%-14s %-8s %-10s %-10s %-10s\n" "-------------" "-------" "-------" "-------" "-------"

for variant in d512_ff1024 d512_ff2048; do
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
for variant in d512_ff1024 d512_ff2048; do
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
echo "Baselines: d256 ff512 mean=0.5168, U-Net mean=0.5955"
