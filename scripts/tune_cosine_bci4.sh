#!/bin/bash
# Test cosine annealing + AdamW (standard transformer recipe)
# 20 epochs, warmup 2 epochs, cosine decay to ~0

PYTHON="/mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python"
SCRIPT="scripts/train.py"
RESULTS="results"
CONFIG="configs/bci4_transformer_d512_ff2048_cosine.yaml"

for S in 1 2 3; do
    echo "=== Cosine: Subject $S (GPU 2) ==="
    $PYTHON $SCRIPT --config $CONFIG --subject $S --gpu 2 --no-wandb \
        2>&1 | tee ${RESULTS}/tune_cosine_s${S}.log
done

echo "=== All cosine runs complete ==="
