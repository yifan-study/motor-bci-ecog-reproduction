"""Unified training entry point for all models and datasets.

Usage:
    python scripts/train.py --config configs/fingerflex_lomtev.yaml --patient bp --epochs 40
    python scripts/train.py --config configs/fingerflex_raw_unet.yaml --patient bp --epochs 100
    python scripts/train.py --config configs/bci4_lomtev.yaml --subject 1 --epochs 20
"""

import sys
import os
import argparse
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import torch

from src.data import build_data
from src.models import build_model
from src.training.trainer import Trainer
from src.utils.config import load_config, make_experiment_dir
from src.utils.seed import set_seed
from src.utils.wandb_utils import init_wandb, finish


def main():
    parser = argparse.ArgumentParser(description="Unified training script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--patient", type=str, default=None, help="Miller patient ID")
    parser.add_argument("--subject", type=int, default=None, help="BCI-IV subject ID")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    # Load and override config
    config = load_config(args.config)
    if args.patient is not None:
        config["data"]["patient"] = args.patient
    if args.subject is not None:
        config["data"]["subject"] = args.subject
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_wandb:
        config.setdefault("logging", {})["use_wandb"] = False

    set_seed(config.get("seed", 42))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Model:  {config['model']['name']}")

    # Data
    print("Building data pipeline...")
    t0 = time.time()
    ds = build_data(config)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Train: {len(ds['train'])} | Val: {len(ds['val'])} | Test: {len(ds['test'])}")
    print(f"  Channels: {ds['n_channels']} | Input features: {ds.get('n_input_features', '?')}")

    # Model
    model = build_model(config, ds)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Experiment dir + W&B
    exp_dir = make_experiment_dir(config, base_dir="results")
    print(f"  Experiment dir: {exp_dir}")
    init_wandb(config, tags=config.get("logging", {}).get("tags", []))

    # Train
    trainer = Trainer(model, config, ds, device, exp_dir)
    fit_result = trainer.fit()
    print(f"\n  Best val r: {fit_result['best_val_r']:.4f}")

    # Test
    test_result = trainer.test()
    print(f"\nResults saved to {exp_dir}")

    finish()


if __name__ == "__main__":
    main()
