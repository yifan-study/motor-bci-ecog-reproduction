# Motor-BCI ECoG Decoder Reproduction

Reproduction code and a unified evaluation protocol for the ECoG finger-trajectory
decoders analyzed in:

> Y. Yu, P. Billion-Polak, T. M. Khoshgoftaar.
> *A Review of Machine Learning Applications in Motor Brain-Computer Interfaces.* (under review)

The review reimplements four published deep-learning decoders for the
**BCI Competition IV Dataset 4** finger-flexion benchmark and evaluates them under a
single protocol, to test which reported correlations survive independent replication.

## What this reproduces

| Decoder | Architecture | Reported *r* | Reproduced *r* |
|---|---|---:|---:|
| FingerFlex (Lomtev et al. 2022) | Wavelet + U-Net | 0.67 | 0.60 |
| DTCNet (Wang et al. 2025) | Dilated-transposed CNN | 0.69 | 0.68 |
| DeepFingerNet (Tao et al. 2025) | Nested U-Net | 0.69 | 0.43 |
| BC4D4 (Jangir et al. 2025) | Hybrid CNN-DNN | 0.86 | n/r |

Values are the mean per-finger Pearson correlation on BCI Competition IV Dataset 4, all
obtained under the one protocol below. `n/r` = not reproducible on the full test signal
under a standard protocol.

## Unified protocol

Every decoder is trained and scored identically; only the model changes.

- **Preprocessing:** Morlet wavelet features, 40–300 Hz bandpass with a 60 Hz notch,
  downsampled to 100 Hz.
- **Windowing:** 256-sample windows.
- **Split:** a contiguous temporal train/validation/test split that respects neural
  autocorrelation (no shuffling across the recording).
- **Optimization:** identical Adam settings, loss, and schedule for every model, with a
  single fixed random seed.
- **Metric:** per-finger Pearson correlation averaged over the five fingers, scored on
  the full (Gaussian-smoothed) test signal. No method is credited for evaluating on a
  filtered subset.

## Repository layout

- `src/models/` — decoder implementations (U-Net, nested U-Net, DTCNet, TCN, transformer,
  conformer, state-space).
- `src/data/` — BCI Competition IV Dataset 4 loaders and the wavelet preprocessing pipeline.
- `src/training/`, `src/evaluation/` — the shared trainer and metrics.
- `configs/` — one YAML per decoder and setting (e.g. `bci4_dtcnet.yaml`,
  `bci4_deepfingernet.yaml`).
- `scripts/` — training entry points and result-collection utilities.
- `download_dataset.py` — fetches the public datasets (see below).

## Reproducing a decoder

```bash
pip install -r requirements.txt
python download_dataset.py                 # fetches the public datasets into data/
python scripts/train_lomtev_bci4.py --config configs/bci4_dtcnet.yaml
```

Swap the `--config` for another `configs/bci4_*.yaml` to reproduce a different decoder.
Trained weights and results are written under `results/` (git-ignored). Data are
downloaded locally and are never committed to this repository.

---

## Datasets

The reproduction uses the public **BCI Competition IV Dataset 4** (Schalk et al.), a
finger-flexion benchmark with three epileptic subjects. `download_dataset.py` also fetches
the **Miller ECoG library** (Kai J. Miller, *A library of human electrocorticographic data
and analyses*, Nature Human Behaviour, 2019; Stanford Digital Repository,
<https://purl.stanford.edu/zk881ps0522>), which the broader benchmark draws on.

```bash
python download_dataset.py                     # download + extract everything (~7.5 GB)
python download_dataset.py --output /path/to/data
python download_dataset.py --select            # interactively pick files
python download_dataset.py --verify-only       # check integrity of prior downloads
```

Re-running is safe — already-downloaded files are skipped. The library contains 204 ECoG
datasets from 34 patients across 16 behavioral experiments; the finger-flexion subset used
here is `BCI_Competion4_dataset4_data_fingerflexions` (derived from the `fingerflex`
experiment, 9 patients).

## Citation

If you use this code, please cite the review and the underlying data library:

```bibtex
@article{miller_library_2019,
  title   = {A library of human electrocorticographic data and analyses},
  author  = {Miller, Kai J.},
  journal = {Nature Human Behaviour},
  volume  = {3}, number = {11}, pages = {1225--1235}, year = {2019},
  doi     = {10.1038/s41562-019-0678-3},
  url     = {https://purl.stanford.edu/zk881ps0522}
}
```

## License

- **This code:** see [`LICENSE`](LICENSE).
- **Miller ECoG dataset:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
  (Stanford Digital Repository).
