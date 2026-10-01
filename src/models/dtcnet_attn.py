"""DTCNet + bottleneck self-attention.

Subclasses DTCNet (Wang et al. 2025) and inserts 1+ pre-LN transformer encoder
layers between the encoder bottleneck (512-dim, T/32) and the decoder. At
training window T=256 the bottleneck is only 8 timesteps — trivially cheap for
self-attention.

Default n_attn_layers=2 adds ~6.3M params on top of the ~5.2M baseline.
"""

import torch
import torch.nn as nn

from src.models.dtcnet import DTCNet
from src.models.transformer import SinusoidalPositionalEncoding


class DTCNetAttn(DTCNet):
    """DTCNet with self-attention at the bottleneck."""

    def __init__(self, n_channels_in: int, n_channels_out: int = 5,
                 dropout: float = 0.1,
                 n_attn_layers: int = 2, n_attn_heads: int = 8,
                 attn_dim_feedforward: int = 2048):
        super().__init__(n_channels_in, n_channels_out, dropout)

        self.bottleneck_pe = SinusoidalPositionalEncoding(512, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=n_attn_heads,
            dim_feedforward=attn_dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-LN for stable training
        )
        self.bottleneck_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=n_attn_layers
        )
        self.bottleneck_norm = nn.LayerNorm(512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, F, T = x.shape
            x = x.reshape(B, C * F, T)

        # Encoder (inherited)
        x0 = self.fr(x)
        x1 = self.enc[0](x0)
        x2 = self.enc[1](x1)
        x3 = self.enc[2](x2)
        x4 = self.enc[3](x3)
        x5 = self.enc[4](x4)      # (B, 512, T//32) — bottleneck

        # Self-attention at bottleneck
        z = x5.transpose(1, 2)            # (B, T/32, 512)
        z = self.bottleneck_pe(z)
        z = self.bottleneck_attn(z)
        z = self.bottleneck_norm(z)
        x5 = z.transpose(1, 2)            # (B, 512, T/32)

        # Decoder (inherited structure)
        d = self.initial_up(x5)
        d = d[..., :x4.shape[-1]]
        d = self.dec[0](d, x4)
        d = self.dec[1](d, x3)
        d = self.dec[2](d, x2)
        d = self.dec[3](d, x1)
        d = self.dec[4](d, x0)

        return self.head(d)
