#!/usr/bin/env bash
# Phase 6 experiments — BCI-IV S1 only (Stage 1 signal check)
#
# Testing 4 untested improvements vs best (L6+h16, test r=0.620):
#   A) sigma_val fix: smooth_sigma_val 6→1 (control, same LR)
#   B) LR=1e-4      : LR sweep (most likely optimal)
#   C) LR=2e-4      : LR sweep (higher)
#   D) LR=3e-4      : LR sweep (aggressive)
#   E) SwiGLU       : gated FFN, ff=2730 (equal params)
#   F) d=1536       : capacity scaling
#
# All use sigma_val=1. All S1 only.
# GPUs available: 0 and 2 (GPU 1 has hardware error).
#
# Ordering: LR sweep first (fastest signal). SwiGLU and d=1536 run in parallel.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python
GPUS=(0 1)   # GPU 1 hardware error: PyTorch sees physical 0→cuda:0, physical 2→cuda:1
PIDS=()
GPU_IDX=0

run_job() {
    local label="$1" config="$2" lr="$3" gpu="$4"
    local logfile="results/phase6_s1_${label}.log"
    echo "[launch] $label — lr=$lr, gpu=$gpu, log=$logfile"
    $PYTHON scripts/train.py \
        --config "$config" \
        --subject 1 \
        --lr "$lr" \
        --gpu "$gpu" \
        --seed 123 \
        --no-wandb \
        > "$logfile" 2>&1 &
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
        PIDS=("${new_pids[@]+"${new_pids[@]}"}")
        [ ${#PIDS[@]} -ge ${#GPUS[@]} ] && sleep 15
    done
    GPU_IDX=$(( (GPU_IDX + 1) % ${#GPUS[@]} ))
}

echo "================================================================"
echo " Phase 6: sigma fix + LR sweep + SwiGLU + d=1536  (BCI-IV S1)"
echo " Baseline: L6+h16, sigma_val=6, lr=8.42e-5 → test r=0.620"
echo " GPUs: ${GPUS[*]}"
echo "================================================================"
echo ""

# Launch all 6 experiments (2 at a time across GPUs 0 and 2)
for entry in \
    "A_sigma_ctrl:bci4_transformer_d1024_ff4096_L6_h16_s1val.yaml:8.42e-5" \
    "B_lr1e4:bci4_transformer_d1024_ff4096_L6_h16_s1val.yaml:1e-4" \
    "C_lr2e4:bci4_transformer_d1024_ff4096_L6_h16_s1val.yaml:2e-4" \
    "D_lr3e4:bci4_transformer_d1024_ff4096_L6_h16_s1val.yaml:3e-4" \
    "E_swiglu:bci4_transformer_d1024_swiglu_L6_h16_s1val.yaml:8.42e-5" \
    "F_d1536:bci4_transformer_d1536_ff4096_L6_h16_s1val.yaml:8.42e-5" ; do
    IFS=':' read -r label config lr <<< "$entry"
    wait_for_slot
    gpu=${GPUS[$GPU_IDX]}
    run_job "$label" "configs/$config" "$lr" "$gpu"
done

echo ""
echo "All jobs launched (PIDs: ${PIDS[*]}). Waiting..."
wait
echo ""
echo "================================================================"
echo " All Phase 6 runs complete. Results:"
echo "================================================================"
echo ""

printf "%-18s %-10s %-10s\n" "Experiment" "Val r" "Test r"
printf "%-18s %-10s %-10s\n" "----------" "------" "------"

for label in A_sigma_ctrl B_lr1e4 C_lr2e4 D_lr3e4 E_swiglu F_d1536; do
    log="results/phase6_s1_${label}.log"
    if [ -f "$log" ]; then
        val_r=$(grep "Best val r:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
        test_r=$(grep "Test r (average):" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
        printf "%-18s %-10s %-10s\n" "$label" "$val_r" "$test_r"
    else
        printf "%-18s %-10s %-10s\n" "$label" "MISSING" "MISSING"
    fi
done

echo ""
echo "Baseline (sigma_val=6, lr=8.42e-5): test r=0.620"
