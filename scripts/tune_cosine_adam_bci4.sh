#!/bin/bash
# Test cosine annealing with Adam (no weight decay change)
# Cosine 25ep (GPU 0) vs Cosine 30ep (GPU 2) — S1 first, then S2, S3

PYTHON="/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python"
SCRIPT="scripts/train.py"
RESULTS="results"

for S in 1 2 3; do
    echo "=== Subject $S: cosine25 (GPU 0) + cosine30 (GPU 2) ==="

    $PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048_cosine_adam.yaml --subject $S --gpu 0 --no-wandb \
        2>&1 | tee ${RESULTS}/tune_cosine_adam_s${S}.log &
    PID1=$!

    $PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048_cosine30.yaml --subject $S --gpu 2 --no-wandb \
        2>&1 | tee ${RESULTS}/tune_cosine30_s${S}.log &
    PID2=$!

    wait $PID1
    echo "cosine25 S$S done."
    wait $PID2
    echo "cosine30 S$S done."
done

echo "=== All cosine-adam runs complete ==="
