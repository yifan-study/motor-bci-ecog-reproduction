"""Verify BCI Competition IV data loading and Lomtev pipeline integration.

Tests:
1. Data loading: all 3 subjects, shapes, dtypes
2. Interpolation: 25 Hz → 100 Hz finger data
3. Full pipeline: build_bci4_lomtev_datasets on subject 1
4. DataLoader + model forward pass
5. All 3 subjects: no crashes, print summary
"""

import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.load_bci4 import load_bci4, BCI4_SUBJECTS
from src.data.preprocessing_lomtev import interpolate_fingerflex
from src.data.dataset_lomtev import build_bci4_lomtev_datasets, LomtevDataset
from src.models.unet_lomtev import AutoEncoder1D
from src.utils.config import load_config


def test_data_loading():
    print("=" * 60)
    print("TEST 1: BCI-IV data loading — all 3 subjects")
    print("=" * 60)

    for sub in BCI4_SUBJECTS:
        raw = load_bci4(sub, data_dir="data/bci4")
        print(f"  Subject {sub}:")
        print(f"    Train ECoG: {raw['train_ecog'].shape} {raw['train_ecog'].dtype}")
        print(f"    Train Flex: {raw['train_flex'].shape} {raw['train_flex'].dtype}")
        print(f"    Test ECoG:  {raw['test_ecog'].shape} {raw['test_ecog'].dtype}")
        print(f"    Test Flex:  {raw['test_flex'].shape} {raw['test_flex'].dtype}")
        print(f"    Channels: {raw['n_channels']}, SR: {raw['sr']}, Flex SR: {raw['flex_sr']}")

        assert raw["train_ecog"].shape[0] == 400000
        assert raw["test_ecog"].shape[0] == 200000
        assert raw["train_ecog"].shape[1] == raw["n_channels"]
        assert raw["train_flex"].shape[1] == 5
        print()

    print("  Data loading ✓")
    print()


def test_interpolation():
    print("=" * 60)
    print("TEST 2: Finger data interpolation (25 Hz → 100 Hz)")
    print("=" * 60)

    raw = load_bci4(1, data_dir="data/bci4")
    flex = raw["train_flex"].T  # (5, time @ 1000)
    print(f"  Input shape: {flex.shape} (5 fingers, {flex.shape[1]} samples @ 1000 Hz)")

    flex_interp = interpolate_fingerflex(flex, cur_fs=1000, true_fs=25, needed_hz=100)
    print(f"  Output shape: {flex_interp.shape} (expect ~(5, {flex.shape[1] // 10}))")

    # Expected: 400000 / 1000 * 100 = 40000 (approximately)
    expected_len = flex.shape[1] // 10
    assert abs(flex_interp.shape[-1] - expected_len) < 10, \
        f"Length mismatch: got {flex_interp.shape[-1]}, expected ~{expected_len}"
    assert flex_interp.shape[0] == 5

    # Check interpolation quality in a region with actual movement (middle of recording)
    mid = flex_interp.shape[-1] // 2
    diffs = np.abs(np.diff(flex_interp[0, mid:mid+100]))
    nonzero_diffs = np.sum(diffs > 1e-10)
    print(f"  Non-zero diffs (middle 100 samples): {nonzero_diffs}/99")
    assert nonzero_diffs > 50, "Interpolation seems to have zero-order hold artifacts"

    print(f"  Flex range: [{flex_interp.min():.4f}, {flex_interp.max():.4f}]")
    print("  Interpolation ✓")
    print()


def test_full_pipeline():
    print("=" * 60)
    print("TEST 3: Full BCI-IV Lomtev pipeline — subject 1")
    print("=" * 60)

    config = load_config("configs/bci4_lomtev.yaml")
    ds = build_bci4_lomtev_datasets(1, config)

    print(f"  Pipeline mode: {ds['pipeline_mode']}")
    print(f"  Channels: {ds['n_channels']}, Wavelets: {ds['n_wavelets']}")
    print(f"  Downsample: {ds['downsample_fs']} Hz")
    print(f"  Train: {len(ds['train'])} windows")
    print(f"  Val:   {len(ds['val'])} windows")
    print(f"  Test:  {len(ds['test'])} windows")

    # Check shapes
    spec_w, flex_w = ds["train"][0]
    print(f"  Spec window: {spec_w.shape} (expect ({ds['n_channels']}, {ds['n_wavelets']}, 256))")
    print(f"  Flex window: {flex_w.shape} (expect (5, 256))")
    assert spec_w.shape == (ds['n_channels'], ds['n_wavelets'], 256)
    assert flex_w.shape == (5, 256)
    assert spec_w.dtype == torch.float32

    # Flex should be MinMax scaled [0, 1] in train
    n_check = min(20, len(ds["train"]))
    flex_all = torch.stack([ds["train"][i][1] for i in range(n_check)])
    print(f"  Flex range (train, {n_check} win): [{flex_all.min():.4f}, {flex_all.max():.4f}]")

    print("  Full pipeline ✓")
    print()
    return ds


def test_dataloader_and_model(ds):
    print("=" * 60)
    print("TEST 4: DataLoader + model forward pass")
    print("=" * 60)

    loader = DataLoader(ds["train"], batch_size=16, shuffle=True)
    spec_batch, flex_batch = next(iter(loader))
    print(f"  Spec batch: {spec_batch.shape}")
    print(f"  Flex batch: {flex_batch.shape}")

    model = AutoEncoder1D(
        n_electrodes=ds["n_channels"], n_freqs=ds["n_wavelets"],
        n_channels_out=5,
        channels=[32, 32, 64, 64, 128, 128],
        kernel_sizes=[7, 7, 5, 5, 5],
        strides=[2, 2, 2, 2, 2],
        dilation=[1, 1, 1, 1, 1],
    )
    with torch.no_grad():
        pred = model(spec_batch)
    print(f"  Model output: {pred.shape} (expect (16, 5, 256))")
    assert pred.shape == (16, 5, 256)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    print("  DataLoader + model ✓")
    print()


def test_all_subjects():
    print("=" * 60)
    print("TEST 5: All 3 BCI-IV subjects")
    print("=" * 60)

    config = load_config("configs/bci4_lomtev.yaml")

    print(f"  {'Subject':<8} {'Ch':>4} {'Wv':>4} {'Train':>8} {'Val':>8} {'Test':>8} {'Spec shape':>20}")
    print(f"  {'-'*8} {'-'*4} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

    for sub in BCI4_SUBJECTS:
        ds = build_bci4_lomtev_datasets(sub, config)
        spec_w, _ = ds["train"][0]
        shape_str = f"({spec_w.shape[0]}, {spec_w.shape[1]}, {spec_w.shape[2]})"
        print(f"  {sub:<8} {ds['n_channels']:>4} {ds['n_wavelets']:>4} "
              f"{len(ds['train']):>8} {len(ds['val']):>8} {len(ds['test']):>8} "
              f"{shape_str:>20}")

    print("  All 3 subjects loaded ✓")
    print()


if __name__ == "__main__":
    test_data_loading()
    test_interpolation()
    ds = test_full_pipeline()
    test_dataloader_and_model(ds)
    test_all_subjects()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
