"""DTCNet: Dilated-Transposed CNN for ECoG → finger flexion.

Reproduces DTCNet (Wang et al., 2025, Frontiers in Computational Neuroscience).
Architecture: encoder-decoder with dilated 1D convolutions + MaxPool downsampling,
sawtooth dilation pattern (1,2,3,1,2), ConvTranspose1d upsampling in decoder,
concatenative skip connections.

Key design (confirmed from paper text Sec 2.3.4 + Figure 4):
- Encoder: Conv1d + LayerNorm + GELU + Dropout + MaxPool (one block per stage)
- Decoder: cat(skip) + Conv1d + LN + GELU + Drop + ConvTranspose1d
  "Skip concatenation combines the low-level detail information obtained during
   the downsampling phase with the high-level abstract features. The concatenated
   feature map doubles the size of the feature in its second dimension, which is
   then processed by a convolution module using conventional 1D convolution.
   Subsequently, [...] transposed convolution to recover the temporal resolution."
- Skip connections: "After the downsampling step of each encoder layer, a pooled
  copy is saved for model skip connections."

Input:  (batch, N_ch*40, T)  — flattened Morlet spectrogram from pipeline,
        OR (batch, N_ch, 40, T) — 4D pipeline format (auto-reshaped)
Output: (batch, n_channels_out, T)

Encoder stages (dims = [n_in, 64, 64, 128, 256, 512, 512]):
  Stage 0  FR:  Conv(n_in→64,  k=3, d=1) + LN + GELU + Drop   [no MaxPool]
  Stage 1  Enc: Conv(64→64,   k=7, d=1) + LN + GELU + Drop + MaxPool(2)
  Stage 2  Enc: Conv(64→128,  k=7, d=2) + LN + GELU + Drop + MaxPool(2)
  Stage 3  Enc: Conv(128→256, k=5, d=3) + LN + GELU + Drop + MaxPool(2)
  Stage 4  Enc: Conv(256→512, k=5, d=1) + LN + GELU + Drop + MaxPool(2)
  Stage 5  Enc: Conv(512→512, k=5, d=2) + LN + GELU + Drop + MaxPool(2) [bottleneck]

Decoder (paper ordering: cat → conv → transposed conv):
  Initial upsample: ConvTranspose(512→512, k=2, s=2) brings bottleneck@T/32 to T/16
  dec[0]: cat(512+512)→256 @T/16, ConvTranspose→T/8  (skip from enc4)
  dec[1]: cat(256+256)→128 @T/8,  ConvTranspose→T/4  (skip from enc3)
  dec[2]: cat(128+128)→64  @T/4,  ConvTranspose→T/2  (skip from enc2)
  dec[3]: cat(64+64)→64    @T/2,  ConvTranspose→T    (skip from enc1)
  dec[4]: cat(64+64)→64    @T     [no upsample]      (skip from FR)

Note on parameter count: paper claims 550-790K but this is inconsistent with the
stated channel dimensions [64,64,128,256,512,512]. Our implementation faithfully
uses the paper's dims/kernels/dilations, yielding ~4.5M params. The paper's count
appears to be an error.

Reference:
    Wang et al. "DTCNet: Dilated-Transposed Convolutional Neural Network for
    Motor Decoding from Electrocorticography"
    Frontiers in Computational Neuroscience, 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LN1d(nn.Module):
    """Channel-wise LayerNorm for (B, C, T) tensors."""
    def __init__(self, n_ch: int):
        super().__init__()
        self.ln = nn.LayerNorm(n_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(-2, -1)).transpose(-2, -1)


class _EncBlock(nn.Module):
    """Encoder block: Conv1d (same-pad) → LN → GELU → Dropout → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 dilation: int = 1, dropout: float = 0.1, pool: int = 2):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation,
                              padding=pad, bias=False)
        self.ln   = _LN1d(out_ch)
        self.drop = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.drop(F.gelu(self.ln(self.conv(x)))))


class _DecBlock(nn.Module):
    """Decoder block (paper ordering): cat(skip) → Conv → LN → GELU → Drop → ConvTranspose.

    Paper Sec 2.3.4: "Skip concatenation [...] The concatenated feature map
    doubles the size of the feature in its second dimension, which is then
    processed by a convolution module using conventional 1D convolution.
    Subsequently, [...] transposed convolution to recover the temporal resolution."

    The last decoder block (at full temporal resolution) omits the ConvTranspose.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 kernel: int, dilation: int = 1, dropout: float = 0.1,
                 upsample: bool = True):
        super().__init__()
        cat_ch = in_ch + skip_ch
        pad    = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(cat_ch, out_ch, kernel, dilation=dilation,
                              padding=pad, bias=False)
        self.ln   = _LN1d(out_ch)
        self.drop = nn.Dropout(p=dropout)
        # ConvTranspose at END doubles temporal resolution (paper ordering)
        self.up = (nn.ConvTranspose1d(out_ch, out_ch, kernel_size=2, stride=2)
                   if upsample else nn.Identity())

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Trim to match (MaxPool rounding can cause ±1 mismatch)
        min_t = min(x.shape[-1], skip.shape[-1])
        x    = x[..., :min_t]
        skip = skip[..., :min_t]
        x = torch.cat([x, skip], dim=1)          # cat with skip
        x = self.drop(F.gelu(self.ln(self.conv(x))))  # Conv → LN → GELU → Drop
        return self.up(x)                         # ConvTranspose (upsample)


class DTCNet(nn.Module):
    """Dilated-Transposed CNN encoder-decoder for continuous ECoG decoding."""

    _DIMS    = [64,  64,  128, 256, 512, 512]   # output ch per stage
    _KERNELS = [3,   7,   7,   5,   5,   5  ]   # conv kernel per stage
    _DILATS  = [1,   1,   2,   3,   1,   2  ]   # dilation per stage

    def __init__(self, n_channels_in: int, n_channels_out: int = 5,
                 dropout: float = 0.1):
        super().__init__()

        dims    = self._DIMS
        kernels = self._KERNELS
        dilats  = self._DILATS

        # ---- Encoder ----
        in_chs = [n_channels_in] + dims[:-1]
        # Stage 0 (Feature Reduction): no MaxPool
        self.fr  = _EncBlock(in_chs[0], dims[0], kernels[0], dilats[0], dropout, pool=1)
        # Stages 1-5: each with MaxPool(2)
        self.enc = nn.ModuleList([
            _EncBlock(in_chs[i], dims[i], kernels[i], dilats[i], dropout, pool=2)
            for i in range(1, 6)
        ])

        # ---- Decoder (paper ordering: cat → conv → ConvTranspose) ----
        # Initial upsample: brings bottleneck from T/32 to T/16 to match enc4 skip
        self.initial_up = nn.ConvTranspose1d(512, 512, kernel_size=2, stride=2)

        # dec[0]: cat(initial_up(512) + enc4(512))=1024 → 256, upsample → T/8
        # dec[1]: cat(dec0(256) + enc3(256))=512 → 128, upsample → T/4
        # dec[2]: cat(dec1(128) + enc2(128))=256 → 64,  upsample → T/2
        # dec[3]: cat(dec2(64)  + enc1(64))=128  → 64,  upsample → T
        # dec[4]: cat(dec3(64)  + fr(64))=128    → 64,  NO upsample (at full T)
        dec_in   = [512, 256, 128,  64,  64]   # input from prev stage
        dec_skip = [512, 256, 128,  64,  64]   # skip from enc stages
        dec_out  = [256, 128,  64,  64,  64]   # output channels
        dec_k    = [  5,   5,   7,   7,   3]   # mirror of enc kernels 4→0
        dec_d    = [  1,   3,   2,   1,   1]   # mirror of enc dilations 4→0

        self.dec = nn.ModuleList([
            _DecBlock(dec_in[j], dec_skip[j], dec_out[j], dec_k[j], dec_d[j],
                      dropout, upsample=(j < 4))
            for j in range(5)
        ])

        # ---- Head ----
        self.head = nn.Conv1d(64, n_channels_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, F, T = x.shape
            x = x.reshape(B, C * F, T)

        # Encoder
        x0 = self.fr(x)           # (B, 64, T)
        x1 = self.enc[0](x0)      # (B, 64,  T//2)
        x2 = self.enc[1](x1)      # (B, 128, T//4)
        x3 = self.enc[2](x2)      # (B, 256, T//8)
        x4 = self.enc[3](x3)      # (B, 512, T//16)
        x5 = self.enc[4](x4)      # (B, 512, T//32)  — bottleneck

        # Decoder (paper ordering: cat → conv → ConvTranspose)
        d = self.initial_up(x5)            # 512@T/32 → 512@T/16
        d = d[..., :x4.shape[-1]]          # trim ±1 from MaxPool rounding
        d = self.dec[0](d, x4)    # cat(512+512)@T/16 → 256, up → T/8
        d = self.dec[1](d, x3)    # cat(256+256)@T/8  → 128, up → T/4
        d = self.dec[2](d, x2)    # cat(128+128)@T/4  → 64,  up → T/2
        d = self.dec[3](d, x1)    # cat(64+64)@T/2    → 64,  up → T
        d = self.dec[4](d, x0)    # cat(64+64)@T      → 64   (no up)

        return self.head(d)        # (B, n_out, T)
