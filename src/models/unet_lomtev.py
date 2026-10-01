"""Lomtev (2023) U-Net encoder-decoder for ECoG → finger flexion.

Ported from ../FingerFlex/Lightning_BCI-autoencoder.ipynb.
Architecture is kept identical; only n_electrodes and n_freqs are
configurable to handle variable channel counts across Miller patients.

Input:  (batch, n_electrodes, n_freqs, time)  — wavelet spectrograms
Output: (batch, n_channels_out, time)          — predicted finger flexion
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d (no bias) → LayerNorm → GELU → Dropout → MaxPool1d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, p_conv_drop=0.1):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                bias=False, padding='same')
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=p_conv_drop)
        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.conv1d(x)
        x = torch.transpose(x, -2, -1)
        x = self.norm(x)
        x = torch.transpose(x, -2, -1)
        x = self.activation(x)
        x = self.drop(x)
        x = self.downsample(x)
        return x


class UpConvBlock(nn.Module):
    """ConvBlock → Upsample (linear interpolation)."""

    def __init__(self, scale, **kwargs):
        super().__init__()
        self.conv_block = ConvBlock(**kwargs)
        self.upsample = nn.Upsample(scale_factor=scale, mode='linear',
                                    align_corners=False)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.upsample(x)
        return x


class AutoEncoder1D(nn.Module):
    """Encoder-decoder with skip connections (Lomtev, 2023).

    Default hyperparameters match the published FingerFlex model:
        channels  = [32, 32, 64, 64, 128, 128]
        kernels   = [7, 7, 5, 5, 5]
        strides   = [2, 2, 2, 2, 2]
    """

    def __init__(self,
                 n_electrodes=62,
                 n_freqs=40,
                 n_channels_out=5,
                 channels=(32, 32, 64, 64, 128, 128),
                 kernel_sizes=(7, 7, 5, 5, 5),
                 strides=(2, 2, 2, 2, 2),
                 dilation=(1, 1, 1, 1, 1)):
        super().__init__()

        channels = list(channels)
        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        self.n_inp_features = n_freqs * n_electrodes
        self.n_channels_out = n_channels_out
        self.model_depth = len(channels) - 1

        # Spatial reduction: flatten (ch × freq) → channels[0]
        self.spatial_reduce = ConvBlock(self.n_inp_features, channels[0],
                                       kernel_size=3)

        # Encoder
        self.downsample_blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1], kernel_sizes[i],
                      stride=strides[i], dilation=dilation[i])
            for i in range(self.model_depth)
        ])

        # Decoder (reverse order, skip-concat doubles input channels)
        dec_channels = channels[:-1] + channels[-1:]
        self.upsample_blocks = nn.ModuleList([
            UpConvBlock(
                scale=strides[i],
                in_channels=(dec_channels[i + 1] if i == self.model_depth - 1
                             else dec_channels[i + 1] * 2),
                out_channels=dec_channels[i],
                kernel_size=kernel_sizes[i],
            )
            for i in range(self.model_depth - 1, -1, -1)
        ])

        # Final 1×1 conv: (channels[0]*2) → n_channels_out
        self.conv1x1_one = nn.Conv1d(channels[0] * 2, n_channels_out,
                                     kernel_size=1, padding='same')

    def forward(self, x):
        batch, elec, n_freq, time = x.shape
        x = x.reshape(batch, -1, time)
        x = self.spatial_reduce(x)

        skips = []
        for block in self.downsample_blocks:
            skips.append(x)
            x = block(x)

        for i, block in enumerate(self.upsample_blocks):
            x = block(x)
            x = torch.cat((x, skips[-(i + 1)]), dim=1)

        x = self.conv1x1_one(x)
        return x
