"""Conv-frontend Transformer for ECoG finger flexion decoding.

Combines a dilated-conv encoder (DTCNet-inspired) for local temporal feature
extraction with transformer layers for global temporal context. The key insight
is that DTCNet's inductive bias (dilated convs, multi-scale) is what makes it
beat a pure transformer at equal or lower param count. This hybrid gives:

1. **Local temporal features** via dilated convolutions (captures oscillations,
   transients, and short-range dependencies efficiently)
2. **Global temporal context** via self-attention (captures long-range
   dependencies across the full window)

Architecture:
    Input (B, C*W, T) at 100 Hz
    → Conv stem: 3 dilated conv blocks (similar to DTCNet encoder stages 0-2)
      [C*W → 128 → 128 → 256] with MaxPool, reducing T → T/4
    → Transformer: L layers of self-attention at T/4 resolution
      (4× cheaper attention than full-res, larger effective receptive field)
    → Upsample: ConvTranspose1d T/4 → T
    → Output projection: Conv1d → n_targets

Input:  (batch, n_electrodes, n_freqs, time) — wavelet spectrograms
Output: (batch, n_channels_out, time)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LN1d(nn.Module):
    """Channel-wise LayerNorm for (B, C, T) tensors."""

    def __init__(self, n_ch):
        super().__init__()
        self.ln = nn.LayerNorm(n_ch)

    def forward(self, x):
        return self.ln(x.transpose(-2, -1)).transpose(-2, -1)


class _ConvBlock(nn.Module):
    """Conv1d (same-pad) → LN → GELU → Dropout → optional MaxPool."""

    def __init__(self, in_ch, out_ch, kernel, dilation=1, dropout=0.1, pool=1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation,
                              padding=pad, bias=False)
        self.ln = _LN1d(out_ch)
        self.drop = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x):
        return self.pool(self.drop(F.gelu(self.ln(self.conv(x)))))


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding with dynamic expansion."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)
        self.register_buffer("pe", torch.zeros(1, 1, d_model))

    def _expand_pe(self, length):
        pe = torch.zeros(length, self.d_model, device=self.div_term.device)
        position = torch.arange(length, dtype=torch.float, device=self.div_term.device).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            self._expand_pe(x.size(1))
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConvTransformerECoG(nn.Module):
    """Conv-frontend Transformer for ECoG decoding.

    Parameters
    ----------
    n_channels_in : int
        Flattened input features (n_electrodes * n_freqs).
    n_channels_out : int
        Number of output targets (e.g. 5 fingers).
    d_model : int
        Transformer hidden dimension.
    n_layers : int
        Number of transformer encoder layers.
    n_heads : int
        Number of attention heads.
    dim_feedforward : int
        FFN hidden dimension.
    conv_channels : list of int
        Channel dims for conv stem stages (default [128, 128, 256]).
    conv_kernels : list of int
        Kernel sizes for conv stem (default [7, 7, 5]).
    conv_dilations : list of int
        Dilations for conv stem (default [1, 2, 3]).
    dropout : float
        Dropout probability.
    eval_window : int
        Sliding-window size for inference.
    channel_dropout_prob : float
        Per-electrode dropout probability.
    """

    def __init__(self, n_channels_in, n_channels_out, d_model=512,
                 n_layers=6, n_heads=8, dim_feedforward=2048,
                 conv_channels=None, conv_kernels=None, conv_dilations=None,
                 dropout=0.1, eval_window=256, channel_dropout_prob=0.0):
        super().__init__()
        self.eval_window = eval_window
        self.channel_dropout_prob = channel_dropout_prob

        if conv_channels is None:
            conv_channels = [128, 128, 256]
        if conv_kernels is None:
            conv_kernels = [7, 7, 5]
        if conv_dilations is None:
            conv_dilations = [1, 2, 3]

        # ── Conv stem (DTCNet-inspired) ─────────────────────────────────────
        # Stage 0: feature reduction, no pool
        stem_layers = [_ConvBlock(n_channels_in, conv_channels[0],
                                  conv_kernels[0], conv_dilations[0],
                                  dropout, pool=1)]
        # Stages 1+: with MaxPool(2) each
        for i in range(1, len(conv_channels)):
            stem_layers.append(
                _ConvBlock(conv_channels[i - 1], conv_channels[i],
                           conv_kernels[i], conv_dilations[i],
                           dropout, pool=2)
            )
        self.conv_stem = nn.ModuleList(stem_layers)
        self.n_pools = len(conv_channels) - 1  # number of MaxPool(2) stages
        self.downsample_factor = 2 ** self.n_pools

        # Project conv output dim → d_model
        conv_out_dim = conv_channels[-1]
        self.proj = nn.Sequential(
            nn.Conv1d(conv_out_dim, d_model, kernel_size=1, bias=False),
            _LN1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Transformer at reduced temporal resolution ──────────────────────
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Upsample back to full resolution ────────────────────────────────
        # Use a series of ConvTranspose1d(stride=2) to undo MaxPool(2) stages
        up_layers = []
        for _ in range(self.n_pools):
            up_layers.extend([
                nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2,
                                   padding=1, bias=False),
                _LN1d(d_model),
                nn.GELU(),
            ])
        self.upsample = nn.Sequential(*up_layers)

        # ── Output projection ───────────────────────────────────────────────
        self.output_conv = nn.Conv1d(d_model, n_channels_out, kernel_size=1)

    def _forward_core(self, x):
        """Conv stem → project → PE → transformer → upsample → output.

        x: (B, C_in, T). Returns: (B, n_out, T).
        """
        T_in = x.shape[-1]

        # Conv stem
        for block in self.conv_stem:
            x = block(x)

        # Project to d_model
        x = self.proj(x)  # (B, d_model, T_down)

        # Transformer at reduced resolution
        x = x.transpose(1, 2)  # (B, T_down, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # (B, d_model, T_down)

        # Upsample back
        x = self.upsample(x)
        # Trim to original length (ConvTranspose can overshoot by 1)
        x = x[:, :, :T_in]

        return self.output_conv(x)

    def forward(self, x):
        # Handle 4D spectrogram input: (B, C, W, T) → (B, C*W, T)
        if x.ndim == 4:
            b, c, w, t = x.shape
            if self.training and self.channel_dropout_prob > 0:
                mask = torch.bernoulli(
                    torch.full((b, c, 1, 1), 1.0 - self.channel_dropout_prob,
                               device=x.device)
                )
                x = x * mask / (1.0 - self.channel_dropout_prob)
            x = x.reshape(b, c * w, t)

        T = x.shape[-1]
        if T <= self.eval_window or self.training:
            return self._forward_core(x)
        return self._sliding_window_forward(x)

    def _sliding_window_forward(self, x):
        """50%-overlap sliding window for long sequences."""
        B, C, T = x.shape
        W = self.eval_window
        stride = W // 2

        out_sum = x.new_zeros(B, self.output_conv.out_channels, T)
        counts = x.new_zeros(1, 1, T)

        for start in range(0, T, stride):
            end = min(start + W, T)
            chunk = x[:, :, start:end]

            pad_len = W - chunk.shape[-1]
            if pad_len > 0:
                chunk = F.pad(chunk, (0, pad_len))

            pred = self._forward_core(chunk)

            if pad_len > 0:
                pred = pred[:, :, :W - pad_len]

            out_sum[:, :, start:end] += pred
            counts[:, :, start:end] += 1

        return out_sum / counts
