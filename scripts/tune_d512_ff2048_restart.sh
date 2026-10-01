#!/bin/bash
# Restart d512 ff=2048 runs (killed by node switch)
# Run S1 + S2 on GPU 0 and GPU 2 in parallel, then S3

PYTHON="/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python"
SCRIPT="scripts/train.py"
RESULTS="results"

echo "=== d512 ff=2048: S1 (GPU 0) + S2 (GPU 2) in parallel ==="

$PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048.yaml --subject 1 --gpu 0 --no-wandb \
    2>&1 | tee ${RESULTS}/tune_d512_ff2048_s1.log &
PID1=$!

$PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048.yaml --subject 2 --gpu 2 --no-wandb \
    2>&1 | tee ${RESULTS}/tune_d512_ff2048_s2.log &
PID2=$!

wait $PID1
echo "S1 done."
wait $PID2
echo "S2 done."

echo "=== d512 ff=2048: S3 (GPU 0) ==="
$PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048.yaml --subject 3 --gpu 0 --no-wandb \
    2>&1 | tee ${RESULTS}/tune_d512_ff2048_s3.log

echo "=== All d512 ff=2048 runs complete ==="
