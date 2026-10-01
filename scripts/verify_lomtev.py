"""Verify the ported Lomtev spectrogram pipeline on Miller data.

Tests:
1. Model: AutoEncoder1D forward pass with variable channel counts
2. Preprocessing: spectrogram shapes for one patient
3. Full pipeline: build_lomtev_datasets on one patient
4. DataLoader: batch shapes match model expectations
5. All 9 patients: no crashes, print summary
"""

import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.unet_lomtev import AutoEncoder1D
from src.data.dataset_lomtev import build_lomtev_datasets, LomtevDataset
from src.data.load_fingerflex import PATIENTS
from src.utils.config import load_config


def test_model_forward():
    print("=" * 60)
    print("TEST 1: AutoEncoder1D forward pass")
    print("=" * 60)

    # Default Lomtev config (62 channels, 40 wavelets)
    hp = dict(
        n_electrodes=62, n_freqs=40, n_channels_out=5,
        channels=[32, 32, 64, 64, 128, 128],
        kernel_sizes=[7, 7, 5, 5, 5],
        strides=[2, 2, 2, 2, 2],
        dilation=[1, 1, 1, 1, 1],
    )
    model = AutoEncoder1D(**hp)
    x = torch.randn(4, 62, 40, 256)
    y = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape} (expect (4, 5, 256))")
    assert y.shape == (4, 5, 256)

    # Variable channels (46 — patient bp)
    hp2 = {**hp, "n_electrodes": 46}
    model2 = AutoEncoder1D(**hp2)
    x2 = torch.randn(4, 46, 40, 256)
    y2 = model2(x2)
    print(f"  46-ch:  {x2.shape} → {y2.shape} (expect (4, 5, 256))")
    assert y2.shape == (4, 5, 256)

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params (62ch): {n_params:,}")
    n_params2 = sum(p.numel() for p in model2.parameters())
    print(f"  Params (46ch): {n_params2:,}")
    print("  Model forward pass ✓")
    print()


def test_one_patient():
    print("=" * 60)
    print("TEST 2: Full pipeline — patient bp")
    print("=" * 60)

    config = load_config("configs/fingerflex_lomtev.yaml")
    ds = build_lomtev_datasets("bp", config)

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

    print("  One patient pipeline ✓")
    print()
    return ds


def test_dataloader(ds):
    print("=" * 60)
    print("TEST 3: DataLoader batch shapes + model compatibility")
    print("=" * 60)

    loader = DataLoader(ds["train"], batch_size=16, shuffle=True)
    spec_batch, flex_batch = next(iter(loader))
    print(f"  Spec batch: {spec_batch.shape} (expect (16, {ds['n_channels']}, {ds['n_wavelets']}, 256))")
    print(f"  Flex batch: {flex_batch.shape} (expect (16, 5, 256))")

    # Feed through model
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
    print("  DataLoader + model ✓")
    print()


def test_all_patients():
    print("=" * 60)
    print("TEST 4: All 9 patients")
    print("=" * 60)

    config = load_config("configs/fingerflex_lomtev.yaml")

    print(f"  {'Patient':<8} {'Ch':>4} {'Wv':>4} {'Train':>8} {'Val':>8} {'Test':>8} {'Spec shape':>20}")
    print(f"  {'-'*8} {'-'*4} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

    for pid in PATIENTS:
        ds = build_lomtev_datasets(pid, config)
        spec_w, _ = ds["train"][0]
        shape_str = f"({spec_w.shape[0]}, {spec_w.shape[1]}, {spec_w.shape[2]})"
        print(f"  {pid:<8} {ds['n_channels']:>4} {ds['n_wavelets']:>4} {len(ds['train']):>8} {len(ds['val']):>8} {len(ds['test']):>8} {shape_str:>20}")

    print("  All 9 patients loaded ✓")
    print()


if __name__ == "__main__":
    test_model_forward()
    ds = test_one_patient()
    test_dataloader(ds)
    test_all_patients()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
