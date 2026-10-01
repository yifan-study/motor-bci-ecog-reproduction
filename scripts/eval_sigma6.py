"""Re-evaluate saved checkpoints at sigma=6 (no retraining).

Usage:
    python scripts/eval_sigma6.py --exp_dir results/20260301_232452_transformer_1 --gpu 0

Or evaluate multiple experiment dirs at once:
    python scripts/eval_sigma6.py \
        --exp_dir results/20260301_232452_transformer_1 \
                  results/20260302_010309_transformer_2 \
                  results/20260302_010512_transformer_3 \
        --gpu 0
"""

import sys
import os
import json
import argparse

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import torch
from src.data import build_data
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.evaluation.metrics import pearson_r_per_channel, smooth_predictions


def evaluate_one(exp_dir, device, sigma=6):
    """Load checkpoint from exp_dir and evaluate at given sigma."""
    config_path = os.path.join(exp_dir, "config.yaml")
    ckpt_path = os.path.join(exp_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"  SKIP {exp_dir} — no best_model.pt")
        return None

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    ds = build_data(config)
    model = build_model(config, ds)
    state_dict = torch.load(ckpt_path, weights_only=True, map_location=device)
    # PE buffers are dynamically recomputed — drop them to avoid size mismatch
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".pe")}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Full-signal evaluation
    test_ds = ds["test"]
    full_input, target = test_ds.get_full_signal()
    inp = full_input.unsqueeze(0).to(device)

    # Stride alignment
    strides = config["model"].get("strides", [1])
    total_stride = 1
    for s in strides:
        total_stride *= s
    T = inp.shape[-1]
    T_aligned = (T // total_stride) * total_stride
    inp = inp[..., :T_aligned]
    target = target[..., :T_aligned]

    with torch.no_grad():
        pred = model(inp).squeeze(0).cpu().numpy()

    # Evaluate at multiple sigmas
    results = {}
    for sig in [1, 6]:
        if sig > 0:
            pred_smooth = smooth_predictions(pred, sig)
        else:
            pred_smooth = pred
        r = pearson_r_per_channel(pred_smooth, target)
        results[f"sigma_{sig}"] = {
            "r_per_channel": [float(x) for x in r],
            "r_avg": float(r.mean()),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", nargs="+", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    all_results = {}
    for exp_dir in args.exp_dir:
        name = os.path.basename(exp_dir)
        print(f"Evaluating {name}...")
        results = evaluate_one(exp_dir, device)
        if results is None:
            continue
        all_results[name] = results

        for sig_key, vals in sorted(results.items()):
            r_str = " ".join(f"{r:.4f}" for r in vals["r_per_channel"])
            print(f"  {sig_key}: avg={vals['r_avg']:.4f}  per_ch=[{r_str}]")
        print()

    # Summary table
    print("=" * 70)
    print(f"{'Experiment':<45} {'σ=1':>8} {'σ=6':>8} {'Δ':>8}")
    print("-" * 70)
    for name, results in all_results.items():
        r1 = results["sigma_1"]["r_avg"]
        r6 = results["sigma_6"]["r_avg"]
        delta = r6 - r1
        print(f"{name:<45} {r1:>8.4f} {r6:>8.4f} {delta:>+8.4f}")
    print("=" * 70)

    # Overall mean across subjects
    if len(all_results) > 1:
        mean_s1 = sum(r["sigma_1"]["r_avg"] for r in all_results.values()) / len(all_results)
        mean_s6 = sum(r["sigma_6"]["r_avg"] for r in all_results.values()) / len(all_results)
        print(f"{'MEAN':<45} {mean_s1:>8.4f} {mean_s6:>8.4f} {mean_s6-mean_s1:>+8.4f}")


if __name__ == "__main__":
    main()
