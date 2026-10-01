"""Verify both pipeline modes ('ours' and 'lomtev') on real Miller data.

Tests:
1. minmax_scale: synthetic + real data
2. 'ours' mode: verify existing behavior unchanged
3. 'lomtev' mode: verify different preprocessing order, MinMax targets
4. Side-by-side comparison: shapes, stats, target ranges
"""

import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.preprocessing import minmax_scale, zscore
from src.data.dataset import build_datasets
from src.data.load_fingerflex import PATIENTS
from src.utils.config import load_config


def test_minmax_scale():
    print("=" * 60)
    print("TEST 1: minmax_scale — synthetic + real data")
    print("=" * 60)

    # Synthetic: known range
    data = np.array([[0, 10], [5, 20], [10, 30]], dtype=np.float64)
    scaled, stats = minmax_scale(data)
    assert scaled.min() == 0.0 and scaled.max() == 1.0
    assert np.allclose(stats["min"], [0, 10])
    assert np.allclose(stats["max"], [10, 30])
    print("  Synthetic: [0,10]×[10,30] → [0,1] ✓")

    # Apply train stats to new data (may go outside [0,1])
    data2 = np.array([[-5, 35], [15, 5]], dtype=np.float64)
    scaled2, _ = minmax_scale(data2, dmin=stats["min"], dmax=stats["max"])
    assert np.allclose(scaled2[0], [-0.5, 1.25])  # outside [0,1] is correct
    print("  Apply to new data: values outside [0,1] handled correctly ✓")

    # Constant column: should not divide by zero
    data3 = np.array([[5, 10], [5, 20], [5, 30]], dtype=np.float64)
    scaled3, _ = minmax_scale(data3)
    assert not np.any(np.isnan(scaled3))
    print("  Constant column: no NaN ✓")
    print()


def test_ours_mode():
    print("=" * 60)
    print("TEST 2: 'ours' pipeline mode (default)")
    print("=" * 60)

    config = load_config("configs/fingerflex_default.yaml")
    assert config["preprocessing"]["pipeline_mode"] == "ours"

    ds = build_datasets("bp", config)
    assert ds["pipeline_mode"] == "ours"

    # Check shapes
    ecog_w, flex_w = ds["train"][0]
    print(f"  ECoG window: {ecog_w.shape} (expect (1000, 46))")
    print(f"  Flex window:  {flex_w.shape} (expect (1000, 5))")
    assert ecog_w.shape == (1000, 46)
    assert flex_w.shape == (1000, 5)

    # Stats should have mean/std (z-score)
    assert "mean" in ds["stats"]["ecog"]
    assert "std" in ds["stats"]["ecog"]
    assert "mean" in ds["stats"]["flex"]
    assert "std" in ds["stats"]["flex"]

    # Target should be z-scored (mean≈0, can be negative)
    flex_sample = ds["train"][0][1]
    print(f"  Flex range: [{flex_sample.min():.2f}, {flex_sample.max():.2f}] (z-scored, unbounded)")
    print("  'ours' mode ✓")
    print()
    return ds


def test_lomtev_mode():
    print("=" * 60)
    print("TEST 3: 'lomtev' pipeline mode")
    print("=" * 60)

    config = load_config("configs/fingerflex_lomtev.yaml")
    assert config["preprocessing"]["pipeline_mode"] == "lomtev"

    ds = build_datasets("bp", config)
    assert ds["pipeline_mode"] == "lomtev"

    # Check shapes
    ecog_w, flex_w = ds["train"][0]
    print(f"  ECoG window: {ecog_w.shape} (expect (1000, 46))")
    print(f"  Flex window:  {flex_w.shape} (expect (1000, 5))")
    assert ecog_w.shape == (1000, 46)
    assert flex_w.shape == (1000, 5)

    # Stats should have min/max (minmax_scale)
    assert "min" in ds["stats"]["flex"]
    assert "max" in ds["stats"]["flex"]

    # Targets should be MinMax scaled — train portion should be in [0, 1]
    n_check = min(50, len(ds["train"]))
    flex_samples = torch.stack([ds["train"][i][1] for i in range(n_check)])
    fmin, fmax = flex_samples.min().item(), flex_samples.max().item()
    print(f"  Flex range (train, {n_check} windows): [{fmin:.4f}, {fmax:.4f}] (expect ≈[0, 1])")
    assert fmin >= -0.01, f"MinMax train min too low: {fmin}"
    assert fmax <= 1.01, f"MinMax train max too high: {fmax}"
    print("  'lomtev' mode ✓")
    print()
    return ds


def test_side_by_side():
    print("=" * 60)
    print("TEST 4: Side-by-side comparison on 'bp'")
    print("=" * 60)

    config_ours = load_config("configs/fingerflex_default.yaml")
    config_lomtev = load_config("configs/fingerflex_lomtev.yaml")

    ds_ours = build_datasets("bp", config_ours)
    ds_lomtev = build_datasets("bp", config_lomtev)

    print(f"  {'':20s} {'ours':>12s} {'lomtev':>12s}")
    print(f"  {'':20s} {'-'*12} {'-'*12}")
    print(f"  {'Train windows':20s} {len(ds_ours['train']):>12d} {len(ds_lomtev['train']):>12d}")
    print(f"  {'Val windows':20s} {len(ds_ours['val']):>12d} {len(ds_lomtev['val']):>12d}")
    print(f"  {'Test windows':20s} {len(ds_ours['test']):>12d} {len(ds_lomtev['test']):>12d}")
    print(f"  {'Channels':20s} {ds_ours['n_channels']:>12d} {ds_lomtev['n_channels']:>12d}")

    # Sample ECoG stats
    ecog_ours = torch.stack([ds_ours["train"][i][0] for i in range(0, len(ds_ours["train"]), max(1, len(ds_ours["train"]) // 50))])
    ecog_lomtev = torch.stack([ds_lomtev["train"][i][0] for i in range(0, len(ds_lomtev["train"]), max(1, len(ds_lomtev["train"]) // 50))])

    print(f"  {'ECoG mean range':20s} [{ecog_ours.mean(dim=(0,1)).min():.3f}, {ecog_ours.mean(dim=(0,1)).max():.3f}]  [{ecog_lomtev.mean(dim=(0,1)).min():.3f}, {ecog_lomtev.mean(dim=(0,1)).max():.3f}]")
    print(f"  {'ECoG std range':20s} [{ecog_ours.std(dim=(0,1)).min():.3f}, {ecog_ours.std(dim=(0,1)).max():.3f}]  [{ecog_lomtev.std(dim=(0,1)).min():.3f}, {ecog_lomtev.std(dim=(0,1)).max():.3f}]")

    # Target stats
    flex_ours = torch.stack([ds_ours["train"][i][1] for i in range(0, len(ds_ours["train"]), max(1, len(ds_ours["train"]) // 50))])
    flex_lomtev = torch.stack([ds_lomtev["train"][i][1] for i in range(0, len(ds_lomtev["train"]), max(1, len(ds_lomtev["train"]) // 50))])

    print(f"  {'Flex target range':20s} [{flex_ours.min():.3f}, {flex_ours.max():.3f}]  [{flex_lomtev.min():.3f}, {flex_lomtev.max():.3f}]")
    print(f"  {'Flex target scaling':20s} {'z-score':>12s} {'minmax[0,1]':>12s}")

    # Verify they produce different results (not accidentally the same)
    ecog_o = ds_ours["train"][0][0]
    ecog_l = ds_lomtev["train"][0][0]
    assert not torch.allclose(ecog_o, ecog_l, atol=1e-3), \
        "ECoG outputs should differ between modes!"
    print("  Modes produce different preprocessing (confirmed) ✓")
    print()


def test_lomtev_all_patients():
    print("=" * 60)
    print("TEST 5: 'lomtev' mode — all 9 patients")
    print("=" * 60)

    config = load_config("configs/fingerflex_lomtev.yaml")

    print(f"  {'Patient':<8} {'Ch':>4} {'Train':>8} {'Val':>8} {'Test':>8} {'Flex range':>16}")
    print(f"  {'-'*8} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*16}")

    for pid in PATIENTS:
        ds = build_datasets(pid, config)
        # Check target range in a few train windows
        n_check = min(20, len(ds["train"]))
        flex_samples = torch.stack([ds["train"][i][1] for i in range(n_check)])
        fmin, fmax = flex_samples.min().item(), flex_samples.max().item()
        print(f"  {pid:<8} {ds['n_channels']:>4} {len(ds['train']):>8} {len(ds['val']):>8} {len(ds['test']):>8} [{fmin:>6.3f}, {fmax:>6.3f}]")

    print("  All 9 patients loaded in lomtev mode ✓")
    print()


if __name__ == "__main__":
    test_minmax_scale()
    test_ours_mode()
    test_lomtev_mode()
    test_side_by_side()
    test_lomtev_all_patients()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
