"""Training script for Lomtev U-Net on Miller Library fingerflex patients.

Applies the validated Lomtev (2023) pipeline to 9 Miller patients
to establish an anchor baseline (Milestone 3.0).

Usage:
    python scripts/train_lomtev_miller.py --patient bp --epochs 40 --gpu 0
    python scripts/train_lomtev_miller.py --patient bp --epochs 2 --no-wandb
"""

import sys
import os
import argparse
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d

from src.data.dataset_lomtev import build_lomtev_datasets
from src.models.unet_lomtev import AutoEncoder1D
from src.utils.config import load_config, save_config, make_experiment_dir
from src.utils.seed import set_seed
from src.utils.wandb_utils import init_wandb, log_epoch, log_summary, finish

PATIENTS = ["bp", "cc", "ht", "jc", "jp", "mv", "wc", "wm", "zt"]


# ── Loss ─────────────────────────────────────────────────────────────

def mse_cosine_loss(pred, target, mse_weight=0.5):
    """Lomtev combined loss: 0.5*MSE + 0.5*(1 - cosine_similarity).

    Parameters
    ----------
    pred, target : torch.Tensor, shape (batch, fingers, time)
    mse_weight : float
    """
    mse = F.mse_loss(pred, target)
    # Cosine similarity averaged over batch and fingers
    cos = F.cosine_similarity(pred, target, dim=-1).mean()
    return mse_weight * mse + (1 - mse_weight) * (1 - cos)


# ── Metrics ──────────────────────────────────────────────────────────

def pearson_r_per_finger(pred, target):
    """Pearson correlation per finger.

    Parameters
    ----------
    pred, target : np.ndarray, shape (fingers, time)

    Returns
    -------
    np.ndarray, shape (fingers,) — Pearson r per finger
    """
    rs = []
    for i in range(pred.shape[0]):
        p, t = pred[i], target[i]
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            rs.append(0.0)
        else:
            r = np.corrcoef(p, t)[0, 1]
            rs.append(r if np.isfinite(r) else 0.0)
    return np.array(rs)


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, smooth_sigma=0):
    """Evaluate model on a DataLoader. Returns loss and per-finger Pearson r.

    Parameters
    ----------
    smooth_sigma : float
        Gaussian smoothing sigma applied to predictions before computing
        correlation.  0 = no smoothing.
    """
    model.eval()
    total_loss = 0
    all_pred, all_target = [], []

    for spec_batch, flex_batch in loader:
        spec_batch = spec_batch.to(device)
        flex_batch = flex_batch.to(device)
        pred = model(spec_batch)
        loss = mse_cosine_loss(pred, flex_batch)
        total_loss += loss.item() * spec_batch.size(0)
        all_pred.append(pred.cpu().numpy())
        all_target.append(flex_batch.cpu().numpy())

    n = sum(p.shape[0] for p in all_pred)
    avg_loss = total_loss / n

    # Concatenate along time dimension for correlation
    # Each batch is (batch, fingers, time) → concat batch windows
    pred_cat = np.concatenate(all_pred, axis=0)    # (N, 5, 256)
    target_cat = np.concatenate(all_target, axis=0) # (N, 5, 256)

    # Flatten windows for correlation: (5, N*256)
    pred_flat = pred_cat.transpose(1, 0, 2).reshape(5, -1)
    target_flat = target_cat.transpose(1, 0, 2).reshape(5, -1)

    # Apply Gaussian smoothing to predictions (matches FingerFlex evaluation)
    if smooth_sigma > 0:
        for i in range(pred_flat.shape[0]):
            pred_flat[i] = gaussian_filter1d(pred_flat[i], sigma=smooth_sigma)

    r_per_finger = pearson_r_per_finger(pred_flat, target_flat)
    return avg_loss, r_per_finger


@torch.no_grad()
def evaluate_fullsig(model, dataset, device, smooth_sigma=0):
    """Evaluate on full continuous signal — matches FingerFlex exactly.

    Feeds the entire spectrogram through the model as a single input
    (no windowing), then applies Gaussian smoothing and computes correlation.

    Parameters
    ----------
    dataset : LomtevDataset
        Must have .spec (C, W, T) and .flex (5, T) attributes.
    smooth_sigma : float
        Gaussian smoothing sigma. 0 = no smoothing.

    Returns
    -------
    r_per_finger : np.ndarray, shape (5,)
    """
    model.eval()

    spec = dataset.spec.unsqueeze(0).to(device)  # (1, C, W, T)
    target = dataset.flex.numpy()                  # (5, T)

    # Align to model stride (5 MaxPool1d stride-2 → divisible by 32)
    T = spec.shape[-1]
    T_aligned = (T // 32) * 32
    spec = spec[..., :T_aligned]
    target = target[..., :T_aligned]

    pred = model(spec).squeeze(0).cpu().numpy()    # (5, T_aligned)

    if smooth_sigma > 0:
        for i in range(pred.shape[0]):
            pred[i] = gaussian_filter1d(pred[i], sigma=smooth_sigma)

    return pearson_r_per_finger(pred, target)


# ── Training ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    n_samples = 0

    for spec_batch, flex_batch in loader:
        spec_batch = spec_batch.to(device)
        flex_batch = flex_batch.to(device)

        pred = model(spec_batch)
        loss = mse_cosine_loss(pred, flex_batch)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * spec_batch.size(0)
        n_samples += spec_batch.size(0)

    return total_loss / n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, required=True, choices=PATIENTS,
                        help="Miller patient ID")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8.42e-5)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    args = parser.parse_args()

    # Config
    config = load_config("configs/fingerflex_lomtev.yaml")
    config["data"]["patient"] = args.patient
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["lr"] = args.lr
    if args.no_wandb:
        config.setdefault("logging", {})["use_wandb"] = False
    if args.seed is not None:
        config["seed"] = args.seed

    set_seed(config.get("seed", 42))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print(f"Building Lomtev pipeline for patient {args.patient}...")
    t0 = time.time()
    ds = build_lomtev_datasets(args.patient, config)
    print(f"  Pipeline done in {time.time() - t0:.1f}s")
    print(f"  Train: {len(ds['train'])} | Val: {len(ds['val'])} | Test: {len(ds['test'])}")
    print(f"  Channels: {ds['n_channels']} | Wavelets: {ds['n_wavelets']}")

    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Model
    model_cfg = config["model"]
    model = AutoEncoder1D(
        n_electrodes=ds["n_channels"],
        n_freqs=ds["n_wavelets"],
        n_channels_out=config["data"]["n_targets"],
        channels=model_cfg["channels"],
        kernel_sizes=model_cfg["kernel_sizes"],
        strides=model_cfg["strides"],
        dilation=model_cfg["dilation"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Optimizer & scheduler
    train_cfg = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"],
                                 weight_decay=train_cfg["weight_decay"])
    sched_type = train_cfg.get("scheduler", "reduce_on_plateau")
    if sched_type == "none":
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=train_cfg["scheduler_patience"],
            factor=train_cfg["scheduler_factor"]
        )

    # Experiment directory
    exp_dir = make_experiment_dir(config, base_dir="results")
    print(f"  Experiment dir: {exp_dir}")

    # W&B
    init_wandb(config, tags=config.get("logging", {}).get("tags", []))

    # Smoothing sigmas from config (matches FingerFlex evaluation)
    eval_cfg = config.get("evaluation", {})
    smooth_sigma_val = eval_cfg.get("smooth_sigma_val", 0)
    smooth_sigma_test = eval_cfg.get("smooth_sigma_test", 0)
    print(f"  Smoothing: val sigma={smooth_sigma_val}, test sigma={smooth_sigma_test}")

    # Training loop
    best_val_loss = float("inf")
    best_val_r = -float("inf")
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 15)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val r (avg)':>10} | {'Val r (per finger)':>30} | {'LR':>10} | {'Time':>6}")
    print("-" * 100)

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     grad_clip=train_cfg["grad_clip_max_norm"])
        val_loss, _ = evaluate(model, val_loader, device)  # loss for scheduler
        val_r = evaluate_fullsig(model, ds["val"], device,
                                 smooth_sigma=smooth_sigma_val)
        if scheduler is not None:
            scheduler.step(val_loss)

        elapsed = time.time() - t_start
        lr = optimizer.param_groups[0]["lr"]
        r_avg = val_r.mean()
        r_str = " ".join(f"{r:.3f}" for r in val_r)

        print(f"{epoch:>5} | {train_loss:>10.6f} | {val_loss:>10.6f} | {r_avg:>10.4f} | {r_str:>30} | {lr:>10.2e} | {elapsed:>5.1f}s")

        epoch_metrics = {
            "train/loss": train_loss, "val/loss": val_loss,
            "val/r_avg": float(r_avg), "train/lr": lr,
            "train/epoch_time": elapsed,
        }
        for i, r in enumerate(val_r):
            epoch_metrics[f"val/r_finger_{i}"] = float(r)
        log_epoch(epoch_metrics, step=epoch)

        # Save best model (select by max correlation, not min loss)
        if r_avg > best_val_r:
            best_val_r = r_avg
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Test evaluation with best model
    print("\n" + "=" * 60)
    print("TEST EVALUATION (best val model)")
    print("=" * 60)
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pt"),
                                     weights_only=True))
    test_r = evaluate_fullsig(model, ds["test"], device,
                              smooth_sigma=smooth_sigma_test)
    test_loss, _ = evaluate(model, test_loader, device)  # loss only
    print(f"  Test loss: {test_loss:.6f}")
    print(f"  Test r (per finger): {' '.join(f'{r:.4f}' for r in test_r)}")
    print(f"  Test r (average):    {test_r.mean():.4f}")
    print(f"  Best val r:          {best_val_r:.4f}")

    # Log summary to W&B
    summary = {
        "test/loss": float(test_loss),
        "test/r_avg": float(test_r.mean()),
        "best_val/loss": float(best_val_loss),
        "best_val/r_avg": float(best_val_r),
        "model/n_params": n_params,
        "model/n_channels": ds["n_channels"],
        "data/n_train": len(ds["train"]),
        "data/n_val": len(ds["val"]),
        "data/n_test": len(ds["test"]),
    }
    for i, r in enumerate(test_r):
        summary[f"test/r_finger_{i}"] = float(r)
    log_summary(summary)

    # Save results
    results = {
        "patient": args.patient,
        "best_val_loss": float(best_val_loss),
        "best_val_r": float(best_val_r),
        "test_loss": float(test_loss),
        "test_r_per_finger": [float(r) for r in test_r],
        "test_r_avg": float(test_r.mean()),
        "n_params": n_params,
    }
    import json
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {exp_dir}")

    finish()


if __name__ == "__main__":
    main()
