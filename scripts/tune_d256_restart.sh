#!/usr/bin/env bash
# Restart d_model=256 runs killed on previous node
# d256_ff512: S2, S3 needed (S1 done: test r=0.505)
# d256_ff1024: S1, S2, S3 all needed

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python

echo "=============================================="
echo " d_model=256 restart on BCI-IV"
echo " GPU 0: d256_ff512 (S2,S3) | GPU 1: d256_ff1024 (S1,S2,S3)"
echo "=============================================="
echo ""

# GPU 0: d256_ff512 — S2 and S3
(
    for subject in 2 3; do
        echo "[GPU0] d256_ff512 — subject=$subject"
        $PYTHON scripts/train.py \
            --config configs/bci4_transformer_d256_ff512.yaml \
            --subject "$subject" \
            --gpu 0 \
            --seed 123 \
            --no-wandb \
            > "results/tune_d256_ff512_s${subject}.log" 2>&1
    done
) &

# GPU 1: d256_ff1024 — all 3 subjects
(
    for subject in 1 2 3; do
        echo "[GPU1] d256_ff1024 — subject=$subject"
        $PYTHON scripts/train.py \
            --config configs/bci4_transformer_d256_ff1024.yaml \
            --subject "$subject" \
            --gpu 1 \
            --seed 123 \
            --no-wandb \
            > "results/tune_d256_ff1024_s${subject}.log" 2>&1
    done
) &

echo "All jobs launched. Waiting for completion..."
wait
echo ""
echo "=============================================="
echo " Results:"
echo "=============================================="

echo ""
printf "%-16s %-8s %-10s %-10s %-10s\n" "Variant" "Subject" "Val r" "Test r" "Params"
printf "%-16s %-8s %-10s %-10s %-10s\n" "---------------" "-------" "-------" "-------" "-------"

for variant in d256_ff512 d256_ff1024; do
    for subject in 1 2 3; do
        log="results/tune_${variant}_s${subject}.log"
        if [ -f "$log" ]; then
            val_r=$(grep "Best val r:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
            test_r=$(grep "Test r (average):" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
            params=$(grep "Model params:" "$log" 2>/dev/null | awk '{print $NF}' || echo "N/A")
            printf "%-16s %-8s %-10s %-10s %-10s\n" "$variant" "$subject" "$val_r" "$test_r" "$params"
        else
            printf "%-16s %-8s %-10s %-10s %-10s\n" "$variant" "$subject" "MISSING" "MISSING" "MISSING"
        fi
    done
done

echo ""
echo "--- Per-variant mean test r ---"
for variant in d256_ff512 d256_ff1024; do
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
        printf "%-16s mean=%.4f (n=%d)\n" "$variant" "$mean" "$count"
    else
        printf "%-16s no results\n" "$variant"
    fi
done
echo ""
echo "Baselines: d128_ff1024 mean=0.5052, U-Net mean=0.5955"
