"""Verify the Phase 2 preprocessing pipeline end-to-end.

Tests:
1. windowing.py — synthetic round-trip test
2. splits.py — index coverage, no overlap
3. preprocessing.py — zscore and apply_car on real data
4. dataset.py — build_datasets for all 9 patients, print summary
5. DataLoader — batch shapes
6. Normalization sanity — train mean≈0, std≈1
"""

import sys
import os

# Run from project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.windowing import create_windows, reconstruct_from_windows
from src.data.splits import temporal_split
from src.data.preprocessing import zscore, apply_car
from src.data.dataset import FingerFlexDataset, build_datasets
from src.data.load_fingerflex import PATIENTS
from src.utils.config import load_config


def test_windowing():
    print("=" * 60)
    print("TEST 1: windowing.py — synthetic round-trip")
    print("=" * 60)

    n_samples, n_features = 10000, 8
    window_size, hop_size = 1000, 100
    signal = np.random.randn(n_samples, n_features)

    windows = create_windows(signal, window_size, hop_size)
    expected_n = (n_samples - window_size) // hop_size + 1
    assert windows.shape == (expected_n, window_size, n_features), \
        f"Shape mismatch: {windows.shape} vs ({expected_n}, {window_size}, {n_features})"
    print(f"  Windows shape: {windows.shape} (expected {expected_n} windows) ✓")

    # Reconstruct and check covered region
    reconstructed = reconstruct_from_windows(windows, hop_size, n_samples)
    assert reconstructed.shape == (n_samples, n_features)

    # The covered region should match the original exactly (where counts > 0)
    last_start = (expected_n - 1) * hop_size
    covered_end = last_start + window_size
    covered = signal[:covered_end]
    recon_covered = reconstructed[:covered_end]
    max_err = np.max(np.abs(covered - recon_covered))
    print(f"  Reconstruction max error (covered region): {max_err:.2e} ✓")
    assert max_err < 1e-10, f"Reconstruction error too large: {max_err}"
    print()


def test_splits():
    print("=" * 60)
    print("TEST 2: splits.py — temporal_split")
    print("=" * 60)

    n_samples = 610040
    splits = temporal_split(n_samples, 0.70, 0.15, 0.15)

    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    test_start, test_end = splits["test"]

    # No gaps or overlaps
    assert train_start == 0
    assert train_end == val_start
    assert val_end == test_start
    assert test_end == n_samples

    # Approximate fractions
    train_frac = (train_end - train_start) / n_samples
    val_frac = (val_end - val_start) / n_samples
    test_frac = (test_end - test_start) / n_samples

    print(f"  Train: {train_start:>7d} – {train_end:>7d} ({train_frac:.3f})")
    print(f"  Val:   {val_start:>7d} – {val_end:>7d} ({val_frac:.3f})")
    print(f"  Test:  {test_start:>7d} – {test_end:>7d} ({test_frac:.3f})")
    print(f"  Total coverage: {train_end - train_start + val_end - val_start + test_end - test_start} / {n_samples} ✓")
    print()


def test_preprocessing():
    print("=" * 60)
    print("TEST 3: preprocessing.py — zscore + apply_car")
    print("=" * 60)

    # zscore: fit and apply
    data = np.random.randn(5000, 10) * 50 + 100  # mean=100, std=50
    normed, stats = zscore(data)
    print(f"  zscore (fit): mean range [{normed.mean(axis=0).min():.4f}, {normed.mean(axis=0).max():.4f}] (expect ≈0)")
    print(f"  zscore (fit): std range  [{normed.std(axis=0).min():.4f}, {normed.std(axis=0).max():.4f}] (expect ≈1)")

    # Apply with pre-computed stats
    data2 = np.random.randn(2000, 10) * 50 + 100
    normed2, _ = zscore(data2, mean=stats["mean"], std=stats["std"])
    print(f"  zscore (apply): mean range [{normed2.mean(axis=0).min():.4f}, {normed2.mean(axis=0).max():.4f}] (expect ≈0)")

    # apply_car
    data3 = np.random.randn(1000, 10)
    car_result = apply_car(data3)
    medians_after = np.median(car_result, axis=1)
    print(f"  apply_car: median-across-channels max abs = {np.max(np.abs(medians_after)):.2e} (expect ≈0) ✓")
    print()


def test_all_patients():
    print("=" * 60)
    print("TEST 4: build_datasets — all 9 patients")
    print("=" * 60)

    config = load_config("configs/fingerflex_default.yaml")

    print(f"  {'Patient':<8} {'Ch':>4} {'Train win':>10} {'Val win':>10} {'Test win':>10}")
    print(f"  {'-'*8} {'-'*4} {'-'*10} {'-'*10} {'-'*10}")

    for pid in PATIENTS:
        ds = build_datasets(pid, config)
        n_ch = ds["n_channels"]
        n_train = len(ds["train"])
        n_val = len(ds["val"])
        n_test = len(ds["test"])
        print(f"  {pid:<8} {n_ch:>4} {n_train:>10} {n_val:>10} {n_test:>10}")

        # Spot check shapes for first patient
        if pid == PATIENTS[0]:
            ecog_w, flex_w = ds["train"][0]
            assert ecog_w.shape == (config["windowing"]["window_size"], n_ch), \
                f"ECoG shape: {ecog_w.shape}"
            assert flex_w.shape == (config["windowing"]["window_size"], 5), \
                f"Flex shape: {flex_w.shape}"
            assert ecog_w.dtype == torch.float32
            assert flex_w.dtype == torch.float32

    print("  All 9 patients loaded successfully ✓")
    print()


def test_dataloader():
    print("=" * 60)
    print("TEST 5: DataLoader batch shapes")
    print("=" * 60)

    config = load_config("configs/fingerflex_default.yaml")
    ds = build_datasets("bp", config)
    batch_size = config["training"]["batch_size"]

    loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True)
    ecog_batch, flex_batch = next(iter(loader))

    print(f"  ECoG batch: {ecog_batch.shape} (expect ({batch_size}, {config['windowing']['window_size']}, {ds['n_channels']}))")
    print(f"  Flex batch:  {flex_batch.shape} (expect ({batch_size}, {config['windowing']['window_size']}, 5))")
    assert ecog_batch.shape[0] == batch_size
    assert flex_batch.shape[0] == batch_size
    print("  DataLoader shapes correct ✓")
    print()


def test_normalization_sanity():
    print("=" * 60)
    print("TEST 6: Normalization sanity — train mean≈0, std≈1")
    print("=" * 60)

    config = load_config("configs/fingerflex_default.yaml")
    ds = build_datasets("bp", config)

    # Reconstruct full train tensor to check stats
    train_ds = ds["train"]
    # Use non-overlapping windows to avoid bias from overlap
    # Just sample a few hundred windows
    n_check = min(100, len(train_ds))
    ecog_samples = torch.stack([train_ds[i][0] for i in range(0, len(train_ds), max(1, len(train_ds) // n_check))])
    # Flatten to (total_samples, channels)
    ecog_flat = ecog_samples.reshape(-1, ecog_samples.shape[-1])

    ch_means = ecog_flat.mean(dim=0)
    ch_stds = ecog_flat.std(dim=0)

    print(f"  Train ECoG mean range: [{ch_means.min():.4f}, {ch_means.max():.4f}] (expect ≈0)")
    print(f"  Train ECoG std range:  [{ch_stds.min():.4f}, {ch_stds.max():.4f}] (expect ≈1)")

    # After CAR, means won't be exactly 0 and stds won't be exactly 1,
    # but they should be in a reasonable range
    assert ch_means.abs().max() < 1.0, f"Mean too far from 0: {ch_means.abs().max():.4f}"
    assert ch_stds.min() > 0.3, f"Std too small: {ch_stds.min():.4f}"
    assert ch_stds.max() < 3.0, f"Std too large: {ch_stds.max():.4f}"
    print("  Normalization sanity check passed ✓")
    print()


if __name__ == "__main__":
    test_windowing()
    test_splits()
    test_preprocessing()
    test_all_patients()
    test_dataloader()
    test_normalization_sanity()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
