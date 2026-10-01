#!/bin/bash
# Test stronger regularization to combat overfitting
# wd=1e-4 on GPU 0, wd=1e-3 on GPU 2 — all 3 subjects each

PYTHON="/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python"
SCRIPT="scripts/train.py"
RESULTS="results"

echo "=== wd=1e-4 (GPU 0) | wd=1e-3 (GPU 2) — S1 in parallel ==="

$PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048_wd4.yaml --subject 1 --gpu 0 --no-wandb \
    2>&1 | tee ${RESULTS}/tune_d512_wd4_s1.log &
PID_WD4=$!

$PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048_wd3.yaml --subject 1 --gpu 2 --no-wandb \
    2>&1 | tee ${RESULTS}/tune_d512_wd3_s1.log &
PID_WD3=$!

wait $PID_WD4
echo "wd4 S1 done."
wait $PID_WD3
echo "wd3 S1 done."

echo "=== S2 in parallel ==="

$PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048_wd4.yaml --subject 2 --gpu 0 --no-wandb \
    2>&1 | tee ${RESULTS}/tune_d512_wd4_s2.log &
PID_WD4=$!

$PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048_wd3.yaml --subject 2 --gpu 2 --no-wandb \
    2>&1 | tee ${RESULTS}/tune_d512_wd3_s2.log &
PID_WD3=$!

wait $PID_WD4
echo "wd4 S2 done."
wait $PID_WD3
echo "wd3 S2 done."

echo "=== S3 in parallel ==="

$PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048_wd4.yaml --subject 3 --gpu 0 --no-wandb \
    2>&1 | tee ${RESULTS}/tune_d512_wd4_s3.log &
PID_WD4=$!

$PYTHON $SCRIPT --config configs/bci4_transformer_d512_ff2048_wd3.yaml --subject 3 --gpu 2 --no-wandb \
    2>&1 | tee ${RESULTS}/tune_d512_wd3_s3.log &
PID_WD3=$!

wait $PID_WD4
echo "wd4 S3 done."
wait $PID_WD3
echo "wd3 S3 done."

echo "=== All regularization tuning runs complete ==="
