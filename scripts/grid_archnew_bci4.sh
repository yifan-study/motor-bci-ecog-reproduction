#!/usr/bin/env bash
# Grid search: spatial bottleneck + multiscale transformer on BCI-IV (3 subjects)
#
# Two new architectures vs L6+h16 baseline (mean=0.621):
#   bottleneck128 : TransformerECoG + 2-stage spatial compression (1920→128→1024)
#   multiscale    : MultiscaleTransformerECoG (T→T/4→T/16 encoder + skip decoder)
#
# Layout: one GPU per subject, run both architectures sequentially on that GPU.
#   GPU 0: bottleneck128 S1 → multiscale S1
#   GPU 1: bottleneck128 S2 → multiscale S2
#   GPU 2: bottleneck128 S3 → multiscale S3
#
# Total: 6 runs, ~5-6 hours depending on early stopping.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python
PIDS=()

run_chain() {
    local subject="$1" gpu="$2"
    echo "[launch] Subject $subject on GPU $gpu"

    # Run sequentially: bottleneck128 then multiscale, on the same GPU
    (
        set -e
        echo "[S${subject}] Starting bottleneck128..."
        $PYTHON scripts/train.py \
            --config configs/bci4_transformer_d1024_ff4096_L6_h16_bottleneck128.yaml \
            --subject "$subject" --gpu "$gpu" --no-wandb \
            > "results/grid_archnew_bottleneck128_s${subject}.log" 2>&1
        echo "[S${subject}] Bottleneck128 done."

        echo "[S${subject}] Starting multiscale..."
        $PYTHON scripts/train.py \
            --config configs/bci4_multiscale_transformer_d1024_ff4096_L6_h16.yaml \
            --subject "$subject" --gpu "$gpu" --no-wandb \
            > "results/grid_archnew_multiscale_s${subject}.log" 2>&1
        echo "[S${subject}] Multiscale done."
    ) &

    PIDS+=($!)
}

echo "============================================================="
echo " Architecture Ablation Grid Search on BCI-IV (3 subjects)"
echo " Architectures: bottleneck128, multiscale_transformer"
echo " Baseline (L6+h16): S1=0.620  S2=0.495  S3=0.748  mean=0.621"
echo "============================================================="
echo ""

run_chain 1 0
run_chain 2 1
run_chain 3 2

echo "All chains launched (PIDs: ${PIDS[*]}). Waiting for completion..."
echo ""
wait
echo ""
echo "============================================================="
echo " All 6 runs complete. Parsing results..."
echo "============================================================="

# Parse results
echo ""
printf "%-16s %-8s %-10s %-10s %-10s\n" "Arch" "Subject" "Val r" "Test r" "Params"
printf "%-16s %-8s %-10s %-10s %-10s\n" "----" "-------" "------" "------" "------"

for arch in bottleneck128 multiscale; do
    for subject in 1 2 3; do
        log="results/grid_archnew_${arch}_s${subject}.log"
        if [ -f "$log" ]; then
            val_r=$(grep "Best val r:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
            test_r=$(grep "Test r (average):" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
            params=$(grep "Model params:" "$log" 2>/dev/null | awk '{print $NF}' || echo "N/A")
            printf "%-16s %-8s %-10s %-10s %-10s\n" "$arch" "$subject" "$val_r" "$test_r" "$params"
        else
            printf "%-16s %-8s %-10s %-10s %-10s\n" "$arch" "$subject" "MISSING" "MISSING" "MISSING"
        fi
    done
done

# Compute per-architecture mean test r
echo ""
echo "--- Per-architecture mean test r (vs baseline L6+h16 mean=0.621) ---"
for arch in bottleneck128 multiscale; do
    sum=0; count=0
    for subject in 1 2 3; do
        log="results/grid_archnew_${arch}_s${subject}.log"
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
        printf "%-16s mean=%.4f (n=%d)\n" "$arch" "$mean" "$count"
    else
        printf "%-16s no results\n" "$arch"
    fi
done
