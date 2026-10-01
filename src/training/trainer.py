"""Shared Trainer class for all models.

Handles training loop, evaluation (windowed + full-signal), checkpointing,
and logging. Does NOT own W&B init/finish — the script handles that.
"""

import csv
import json
import os
import time
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import get_loss_fn
from src.evaluation.metrics import pearson_r_per_channel, smooth_predictions
from src.utils.wandb_utils import log_epoch, log_summary


class Trainer:
    """Unified trainer for ECoG decoding models.

    Parameters
    ----------
    model : nn.Module
    config : dict
        Full YAML config.
    dataset_info : dict
        Return value from build_data() — contains train/val/test datasets
        and metadata.
    device : torch.device
    exp_dir : str
        Path to save checkpoints and results.
    """

    def __init__(self, model, config, dataset_info, device, exp_dir):
        self.model = model.to(device)
        self.config = config
        self.ds = dataset_info
        self.device = device
        self.exp_dir = exp_dir

        train_cfg = config["training"]
        self.loss_fn = get_loss_fn(train_cfg["loss"])
        self.grad_clip = train_cfg.get("grad_clip_max_norm", 1.0)
        self.n_targets = config["data"]["n_targets"]

        # Compute total stride for time alignment in full-signal eval
        strides = config["model"].get("strides", [1])
        self.total_stride = reduce(lambda a, b: a * b, strides, 1)

        # Optimizer
        opt_type = train_cfg.get("optimizer", "adam").lower()
        opt_kwargs = dict(
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), **opt_kwargs)

        # Scheduler
        sched_type = train_cfg.get("scheduler", "reduce_on_plateau")
        self.sched_type = sched_type
        if sched_type == "none":
            self.scheduler = None
        elif sched_type == "cosine":
            warmup_epochs = train_cfg.get("warmup_epochs", 0)
            T_max = train_cfg["epochs"] - warmup_epochs
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(T_max, 1), eta_min=1e-7,
            )
            if warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=warmup_epochs,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = cosine
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min",
                patience=train_cfg.get("scheduler_patience", 5),
                factor=train_cfg.get("scheduler_factor", 0.5),
            )

        # DataLoaders
        batch_size = train_cfg["batch_size"]
        self.train_loader = DataLoader(
            self.ds["train"], batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.ds["val"], batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.ds["test"], batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        # Eval config
        eval_cfg = config.get("evaluation", {})
        self.smooth_sigma_val = eval_cfg.get("smooth_sigma_val", 0)
        self.smooth_sigma_test = eval_cfg.get("smooth_sigma_test", 0)

        # Data augmentation (GPU-side, training only)
        aug_cfg = config.get("augmentation", {})
        self.noise_sigma = aug_cfg.get("gaussian_noise_sigma", 0.0)
        self.amp_scale_range = aug_cfg.get("amplitude_scale_range", None)
        # SpecAugment: time and frequency masking (for 4D spectrogram inputs)
        self.time_mask_param = aug_cfg.get("time_mask_param", 0)
        self.freq_mask_param = aug_cfg.get("freq_mask_param", 0)
        self.n_time_masks = aug_cfg.get("n_time_masks", 1)
        self.n_freq_masks = aug_cfg.get("n_freq_masks", 1)
        self.spec_augment = self.time_mask_param > 0 or self.freq_mask_param > 0
        # Channel dropout: per-sample electrode masking, model-agnostic. Applied
        # here (training only) so EVERY architecture gets it identically as a
        # shared protocol element — not a per-model feature. Supersedes the
        # transformer's internal channel_dropout_prob (leave that unset in configs).
        self.channel_dropout_prob = aug_cfg.get("channel_dropout_prob", 0.0)
        self.augment = (self.noise_sigma > 0 or self.amp_scale_range is not None
                        or self.spec_augment or self.channel_dropout_prob > 0)

        # Per-epoch CSV logger
        self._csv_path = os.path.join(exp_dir, "metrics.csv")
        self._csv_header_written = False

    def _augment_batch(self, x):
        """Apply data augmentation to input batch (GPU-side, in-place)."""
        if self.noise_sigma > 0:
            x = x + torch.randn_like(x) * self.noise_sigma
        if self.amp_scale_range is not None:
            lo, hi = self.amp_scale_range
            # Per-channel scaling: shape (B, C, 1, 1) for 4D or (B, C, 1) for 3D
            shape = [x.size(0), x.size(1)] + [1] * (x.ndim - 2)
            scale = torch.empty(shape, device=x.device).uniform_(lo, hi)
            x = x * scale
        if self.spec_augment and x.ndim == 4:
            x = self._spec_augment(x)
        if self.channel_dropout_prob > 0:
            # Per-sample electrode dropout with inverted-dropout scaling. Masks
            # whole electrodes (dim 1) across all freq bands on 4D (B,C,W,T);
            # identical to the transformer's former internal chdrop, now universal.
            keep = 1.0 - self.channel_dropout_prob
            shape = [x.size(0), x.size(1)] + [1] * (x.ndim - 2)
            mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
            x = x * mask / keep
        return x

    def _spec_augment(self, x):
        """SpecAugment: time and frequency masking on 4D spectrograms.

        x: (B, C, W, T) where W = n_freq_bins, T = n_timesteps.
        Masks are applied per-sample (different random masks per batch element).
        """
        B, C, W, T = x.shape
        # Frequency masking: zero out contiguous frequency bands
        if self.freq_mask_param > 0:
            for _ in range(self.n_freq_masks):
                f = torch.randint(0, self.freq_mask_param + 1, (B,))
                f0 = torch.stack([torch.randint(0, max(W - f[i], 1), (1,)).squeeze()
                                  for i in range(B)])
                for i in range(B):
                    x[i, :, f0[i]:f0[i] + f[i], :] = 0
        # Time masking: zero out contiguous time segments
        if self.time_mask_param > 0:
            for _ in range(self.n_time_masks):
                t = torch.randint(0, self.time_mask_param + 1, (B,))
                t0 = torch.stack([torch.randint(0, max(T - t[i], 1), (1,)).squeeze()
                                  for i in range(B)])
                for i in range(B):
                    x[i, :, :, t0[i]:t0[i] + t[i]] = 0
        return x

    def train_one_epoch(self):
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0
        n_samples = 0

        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if self.augment:
                x_batch = self._augment_batch(x_batch)

            pred = self.model(x_batch)
            loss = self.loss_fn(pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            n_samples += x_batch.size(0)

        return total_loss / n_samples

    @torch.no_grad()
    def evaluate_windowed(self, loader):
        """Evaluate on a DataLoader. Returns (avg_loss, r_per_channel).

        Concatenates windowed predictions, flattens, and computes correlation.
        """
        self.model.eval()
        total_loss = 0
        all_pred, all_target = [], []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            pred = self.model(x_batch)
            loss = self.loss_fn(pred, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            all_pred.append(pred.cpu().numpy())
            all_target.append(y_batch.cpu().numpy())

        n = sum(p.shape[0] for p in all_pred)
        avg_loss = total_loss / n

        # Concat and flatten: (N, C, T) → (C, N*T)
        pred_cat = np.concatenate(all_pred, axis=0)
        target_cat = np.concatenate(all_target, axis=0)
        n_ch = pred_cat.shape[1]
        pred_flat = pred_cat.transpose(1, 0, 2).reshape(n_ch, -1)
        target_flat = target_cat.transpose(1, 0, 2).reshape(n_ch, -1)

        r_per_ch = pearson_r_per_channel(pred_flat, target_flat)
        return avg_loss, r_per_ch

    @torch.no_grad()
    def evaluate_fullsig(self, dataset, smooth_sigma=0):
        """Evaluate on full continuous signal (no windowing).

        Uses dataset.get_full_signal() which works for both LomtevDataset
        and FingerFlexDataset.

        Returns
        -------
        r_per_channel : np.ndarray, shape (n_targets,)
        """
        self.model.eval()

        full_input, target = dataset.get_full_signal()
        # full_input: tensor (for Lomtev: (C,W,T), for raw: (C,T))
        # target: np.ndarray (n_targets, T)

        inp = full_input.unsqueeze(0).to(self.device)  # add batch dim

        # Align time to total_stride
        T = inp.shape[-1]
        T_aligned = (T // self.total_stride) * self.total_stride
        inp = inp[..., :T_aligned]
        target = target[..., :T_aligned]

        pred = self.model(inp).squeeze(0).cpu().numpy()  # (n_targets, T_aligned)

        if smooth_sigma > 0:
            pred = smooth_predictions(pred, smooth_sigma)

        return pearson_r_per_channel(pred, target)

    def fit(self):
        """Full training loop with early stopping. Returns best metrics dict."""
        train_cfg = self.config["training"]
        n_epochs = train_cfg["epochs"]
        patience = train_cfg.get("early_stopping_patience", 15)

        best_val_loss = float("inf")
        best_val_r = -float("inf")
        patience_counter = 0

        print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
              f"{'Val r (avg)':>10} | {'Val r (per ch)':>30} | "
              f"{'LR':>10} | {'Time':>6}")
        print("-" * 105)

        for epoch in range(1, n_epochs + 1):
            t_start = time.time()

            train_loss = self.train_one_epoch()
            val_loss, _ = self.evaluate_windowed(self.val_loader)
            val_r = self.evaluate_fullsig(
                self.ds["val"], smooth_sigma=self.smooth_sigma_val
            )

            if self.scheduler is not None:
                if self.sched_type == "reduce_on_plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            elapsed = time.time() - t_start
            lr = self.optimizer.param_groups[0]["lr"]
            r_avg = val_r.mean()
            r_str = " ".join(f"{r:.3f}" for r in val_r)

            print(f"{epoch:>5} | {train_loss:>10.6f} | {val_loss:>10.6f} | "
                  f"{r_avg:>10.4f} | {r_str:>30} | {lr:>10.2e} | {elapsed:>5.1f}s")

            # W&B logging
            epoch_metrics = {
                "train/loss": train_loss, "val/loss": val_loss,
                "val/r_avg": float(r_avg), "train/lr": lr,
                "train/epoch_time": elapsed,
            }
            for i, r in enumerate(val_r):
                epoch_metrics[f"val/r_ch_{i}"] = float(r)
            log_epoch(epoch_metrics, step=epoch)

            # CSV logging
            csv_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_r_avg": float(r_avg),
                "lr": lr,
                "epoch_time": elapsed,
            }
            for i, r in enumerate(val_r):
                csv_row[f"val_r_ch{i}"] = float(r)
            if not self._csv_header_written:
                self._csv_file = open(self._csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(csv_row.keys()))
                self._csv_writer.writeheader()
                self._csv_header_written = True
            self._csv_writer.writerow(csv_row)
            self._csv_file.flush()

            # Save best model (select by max correlation)
            if r_avg > best_val_r:
                best_val_r = r_avg
                best_val_loss = val_loss
                torch.save(self.model.state_dict(),
                           os.path.join(self.exp_dir, "best_model.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        return {"best_val_r": float(best_val_r), "best_val_loss": float(best_val_loss)}

    def test(self):
        """Evaluate best model on test set. Saves JSON + logs W&B summary.

        Returns
        -------
        dict with test results.
        """
        print("\n" + "=" * 60)
        print("TEST EVALUATION (best val model)")
        print("=" * 60)

        ckpt_path = os.path.join(self.exp_dir, "best_model.pt")
        self.model.load_state_dict(
            torch.load(ckpt_path, weights_only=True, map_location=self.device)
        )

        test_r = self.evaluate_fullsig(
            self.ds["test"], smooth_sigma=self.smooth_sigma_test
        )
        test_loss, _ = self.evaluate_windowed(self.test_loader)

        print(f"  Test loss: {test_loss:.6f}")
        print(f"  Test r (per ch): {' '.join(f'{r:.4f}' for r in test_r)}")
        print(f"  Test r (average): {test_r.mean():.4f}")

        n_params = sum(p.numel() for p in self.model.parameters())

        # W&B summary
        summary = {
            "test/loss": float(test_loss),
            "test/r_avg": float(test_r.mean()),
            "model/n_params": n_params,
            "model/n_channels": self.ds["n_channels"],
            "data/n_train": len(self.ds["train"]),
            "data/n_val": len(self.ds["val"]),
            "data/n_test": len(self.ds["test"]),
        }
        for i, r in enumerate(test_r):
            summary[f"test/r_ch_{i}"] = float(r)
        log_summary(summary)

        # Save JSON
        results = {
            "test_loss": float(test_loss),
            "test_r_per_channel": [float(r) for r in test_r],
            "test_r_avg": float(test_r.mean()),
            "n_params": n_params,
        }
        with open(os.path.join(self.exp_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results
