#!/bin/bash
# Test d_model=1024: ff=4096 (GPU 0) vs ff=2048 (GPU 2)
# S1 first to see if scaling trend continues, then S2, S3

PYTHON="/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python"
SCRIPT="scripts/train.py"
RESULTS="results"

for S in 1 2 3; do
    echo "=== Subject $S: d1024_ff4096 (GPU 0) + d1024_ff2048 (GPU 2) ==="

    $PYTHON $SCRIPT --config configs/bci4_transformer_d1024_ff4096.yaml --subject $S --gpu 0 --no-wandb \
        2>&1 | tee ${RESULTS}/tune_d1024_ff4096_s${S}.log &
    PID1=$!

    $PYTHON $SCRIPT --config configs/bci4_transformer_d1024_ff2048.yaml --subject $S --gpu 2 --no-wandb \
        2>&1 | tee ${RESULTS}/tune_d1024_ff2048_s${S}.log &
    PID2=$!

    wait $PID1
    echo "ff4096 S$S done."
    wait $PID2
    echo "ff2048 S$S done."
done

echo "=== All d1024 runs complete ==="
