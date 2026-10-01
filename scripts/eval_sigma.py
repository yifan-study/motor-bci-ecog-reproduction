"""Re-evaluate a saved checkpoint with a different smooth_sigma_test.

Usage:
    python scripts/eval_sigma.py --result_dir results/20260322_185420_dtcnet_1 --sigma 6 --gpu 0
"""
import argparse
import json
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_data
from src.models import build_model
from src.evaluation.metrics import pearson_r_per_channel, smooth_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.result_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Build data (returns a single dict with datasets + metadata)
    data = build_data(config)

    # Build model and load checkpoint
    dataset_info = {
        "n_channels": data["n_channels"],
        "n_input_features": data.get("n_input_features", 1),
    }
    model = build_model(config, dataset_info).to(device)
    ckpt = os.path.join(args.result_dir, "best_model.pt")
    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    model.eval()

    # Get total_stride
    strides = config["model"].get("strides", [1])
    total_stride = 1
    for s in strides:
        total_stride *= s

    # Evaluate on test set (full signal)
    test_ds = data["test"]
    full_input, target = test_ds.get_full_signal()
    inp = full_input.unsqueeze(0).to(device)

    T = inp.shape[-1]
    T_aligned = (T // total_stride) * total_stride
    inp = inp[..., :T_aligned]
    target = target[..., :T_aligned]

    with torch.no_grad():
        pred = model(inp).squeeze(0).cpu().numpy()

    # Apply sigma smoothing
    if args.sigma > 0:
        pred = smooth_predictions(pred, args.sigma)

    r_per_ch = pearson_r_per_channel(pred, target)

    subj = config["data"]["subject"]
    print(f"Subject {subj} | sigma={args.sigma}")
    print(f"  Test r (per ch): {' '.join(f'{r:.4f}' for r in r_per_ch)}")
    print(f"  Test r (average): {r_per_ch.mean():.4f}")
    print(f"  Best finger: {r_per_ch.max():.4f}")


if __name__ == "__main__":
    main()
