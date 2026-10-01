"""Raw ECoG U-Net encoder-decoder.

Same architecture as AutoEncoder1D (Lomtev, 2023) but operates directly
on raw ECoG time series instead of wavelet spectrograms.

Input:  (batch, n_channels, time)   — raw ECoG (channels-first)
Output: (batch, n_targets, time)    — predicted finger flexion
"""

import torch
import torch.nn as nn

from src.models.unet_lomtev import ConvBlock, UpConvBlock


class AutoEncoder1DRaw(nn.Module):
    """U-Net for raw ECoG signals (no spectrogram preprocessing).

    Key differences from AutoEncoder1D:
    - Input is (batch, n_channels, time), no reshape needed
    - spatial_reduce maps n_channels (~50) to channels[0], not
      n_electrodes*n_freqs (~2000)

    Parameters
    ----------
    n_channels_in : int
        Number of ECoG electrodes (input channels).
    n_channels_out : int
        Number of output channels (e.g. 5 fingers).
    channels, kernel_sizes, strides, dilation : tuples
        Architecture hyperparameters (same as AutoEncoder1D).
    """

    def __init__(self,
                 n_channels_in=62,
                 n_channels_out=5,
                 channels=(32, 32, 64, 64, 128, 128),
                 kernel_sizes=(7, 7, 5, 5, 5),
                 strides=(2, 2, 2, 2, 2),
                 dilation=(1, 1, 1, 1, 1)):
        super().__init__()

        channels = list(channels)
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.model_depth = len(channels) - 1

        # Spatial reduction: n_channels_in → channels[0]
        self.spatial_reduce = ConvBlock(n_channels_in, channels[0],
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

        # Final 1x1 conv: (channels[0]*2) → n_channels_out
        self.conv1x1_one = nn.Conv1d(channels[0] * 2, n_channels_out,
                                     kernel_size=1, padding='same')

    def forward(self, x):
        # x: (batch, n_channels_in, time) — no reshape needed
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
