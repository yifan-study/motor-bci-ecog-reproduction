#!/bin/bash
# Wait for d1024_ff2048 S3 to finish, then run window=512 experiments on GPU 2
PYTHON="/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python"
RESULTS="results"

echo "Waiting for d1024_ff2048 S3 to finish..."
while kill -0 3377993 2>/dev/null; do
    sleep 30
done
echo "d1024_ff2048 S3 done. Starting window=512 experiments on GPU 2."

for S in 1 2 3; do
    echo "=== window=512: Subject $S (GPU 2) ==="
    $PYTHON scripts/train.py --config configs/bci4_transformer_d1024_ff4096_w512.yaml \
        --subject $S --gpu 2 --no-wandb \
        2>&1 | tee ${RESULTS}/tune_d1024_w512_s${S}.log
    echo "Subject $S done."
done

echo "=== All window=512 runs complete ==="
