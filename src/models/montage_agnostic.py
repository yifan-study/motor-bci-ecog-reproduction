"""Montage-agnostic ECoG decoder for cross-patient transfer.

The problem: Miller patients have different electrode montages (38-64 channels, at
different 3-D positions), so a standard fixed-input decoder can't transfer from one
patient to another. This model is invariant to electrode count and order:

    1. A SHARED temporal encoder embeds each electrode's feature time-series
       (applied identically to every electrode -> works for any channel count).
    2. Each electrode's 3-D coordinate is embedded (MLP) and added to its encoding,
       so the model knows WHERE each electrode sits.
    3. Attention pooling aggregates across electrodes -> a fixed-size representation
       regardless of how many electrodes there are (permutation- and count-invariant).
    4. A temporal decoder maps that to the targets.

Trained with channel dropout (random electrodes zeroed), it learns not to depend on
any specific montage, which is what lets it decode a held-out patient's ECoG.

Forward:  x (B, C, F, T), coords (C, 3)  ->  (B, n_targets, T).  C may vary per batch.
Length-preserving in time (T_out == T), so it plugs into full-signal Pearson-r eval.
"""

import torch
import torch.nn as nn


class MontageAgnosticDecoder(nn.Module):
    def __init__(self, n_input_features, n_targets, d_model=96, n_enc_layers=4,
                 kernel_size=7, dropout=0.1, n_regions=16):
        super().__init__()
        self.n_targets = n_targets

        # (1) shared per-electrode temporal encoder (length-preserving convs)
        enc = []
        c_in = n_input_features
        for _ in range(n_enc_layers):
            enc += [
                nn.Conv1d(c_in, d_model, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            c_in = d_model
        self.enc = nn.Sequential(*enc)

        # (2) electrode-position embedding from 3-D coordinates
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        # (2b) anatomical-region embedding — region codes are shared across patients,
        # giving a functional coordinate frame (better for transfer than raw geometry).
        self.region_emb = nn.Embedding(n_regions, d_model)

        # (3) attention pooling over electrodes (count-invariant)
        self.score = nn.Linear(d_model, 1)

        # (4) temporal decoder -> targets (length-preserving)
        self.dec = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, n_targets, 1),
        )

    def forward(self, x, coords, regions=None):
        """x: (B,C,F,T); coords: (C,3) xyz; regions: (C,) anatomical codes (optional)."""
        B, C, Fdim, T = x.shape
        h = self.enc(x.reshape(B * C, Fdim, T))        # (B*C, d, T)
        d = h.shape[1]
        h = h.reshape(B, C, d, T)
        pos = self.pos_mlp(coords.to(h.dtype))          # (C, d)
        h = h + pos[None, :, :, None]                   # inject 3-D position per electrode
        if regions is not None:                          # inject shared anatomical region
            h = h + self.region_emb(regions)[None, :, :, None]
        scores = self.score(h.mean(dim=-1)).squeeze(-1)  # (B, C) attention logits
        w = torch.softmax(scores, dim=1)                 # (B, C)
        pooled = (h * w[:, :, None, None]).sum(dim=1)    # (B, d, T) count-invariant
        return self.dec(pooled)                          # (B, n_targets, T)
