"""Temporal Convolutional Network for ECoG → finger flexion decoding.

TCN (Bai et al., 2018) adapted for sequence-to-sequence regression:
  - Acausal (padding='same') dilated convolutions — fair comparison with U-Net
  - LayerNorm + GELU + Dropout — matches U-Net ConvBlock pattern
  - No downsampling — preserves temporal resolution (time in = time out)
  - Residual connections in each block

Input:  (batch, n_electrodes, n_freqs, time) — wavelet spectrograms
Output: (batch, n_channels_out, time)         — predicted finger flexion
"""

import torch
import torch.nn as nn


class _TransposedLayerNorm(nn.Module):
    """LayerNorm that works on (B, C, T) by transposing to (B, T, C)."""

    def __init__(self, n_features):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C) -> norm -> (B, C, T)
        return self.norm(x.transpose(-2, -1)).transpose(-2, -1)


class TCNBlock(nn.Module):
    """Single TCN residual block: two dilated convolutions with skip connection.

    Each conv path: Conv1d → LayerNorm → GELU → Dropout
    Output: block(x) + residual(x)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding="same", dilation=dilation, bias=False,
        )
        self.norm1 = _TransposedLayerNorm(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding="same", dilation=dilation, bias=False,
        )
        self.norm2 = _TransposedLayerNorm(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        out = self.drop(self.act(self.norm1(self.conv1(x))))
        out = self.drop(self.act(self.norm2(self.conv2(out))))
        return out + self.residual(x)


class TCN(nn.Module):
    """Temporal Convolutional Network for ECoG decoding.

    Parameters
    ----------
    n_channels_in : int
        Flattened input features (n_electrodes * n_freqs).
    n_channels_out : int
        Number of output targets (e.g. 5 fingers).
    hidden_channels : int
        Uniform channel width for all TCN blocks.
    kernel_size : int
        Convolution kernel size in each block.
    n_blocks : int
        Number of TCN residual blocks.
    dilation_base : int
        Dilation grows as dilation_base**i for block i.
    dropout : float
        Dropout probability.
    """

    def __init__(self, n_channels_in, n_channels_out, hidden_channels=64,
                 kernel_size=5, n_blocks=6, dilation_base=2, dropout=0.1):
        super().__init__()

        # Spatial reduction: flatten (ch * freq) → hidden_channels
        self.spatial_reduce = nn.Sequential(
            nn.Conv1d(n_channels_in, hidden_channels, kernel_size=3,
                      padding="same", bias=False),
            _TransposedLayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stack of TCN blocks with exponentially growing dilation
        self.blocks = nn.ModuleList([
            TCNBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation_base ** i,
                dropout=dropout,
            )
            for i in range(n_blocks)
        ])

        # Output projection
        self.output_conv = nn.Conv1d(hidden_channels, n_channels_out, kernel_size=1)

    def forward(self, x):
        # Handle 4D spectrogram input: (B, C, W, T) → (B, C*W, T)
        if x.ndim == 4:
            b, c, w, t = x.shape
            x = x.reshape(b, c * w, t)

        x = self.spatial_reduce(x)
        for block in self.blocks:
            x = block(x)
        return self.output_conv(x)
