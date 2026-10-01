"""NestedUNet: 3-level UNet++ for ECoG → finger flexion.

Reproduces DeepFingerNet (Tao et al., 2025, IEEE TIM).
Architecture: 3 nested U-Nets with dense skip connections (UNet++ style).

Encoder: Conv→LN→GELU→Dropout per level (k=3), channels [32, 64, 128, 256]
Decoder: Pre-norm (LN→GELU→Conv→Drop→LN→GELU→Conv) per node
Upsampling: ConvTranspose1d (first/direct decoder path); Upsample (nested paths)

v3 fixes (from re-reading paper Fig. 2):
  - Decoder kernel_size configurable (default 3, was hardcoded 1 — k=1 gives
    zero temporal receptive field in reconstruction path)
  - Dense output: head receives cat(e0, d01, d02, d03) = 4×base_ch = 128ch,
    matching the paper diagram's "128 → Output" (was 32 → Output)

Input:  (batch, N, T)           — raw ECoG, N electrodes at 100 Hz
        OR (batch, N, W, T)     — 4-D spectrogram (n_wavelets>0)
Output: (batch, n_channels_out, T)

Reference:
    Tao et al. "DeepFingerNet: A Nested U-Net for ECoG-based Finger Decoding"
    IEEE Transactions on Instrumentation and Measurement, 2025.
    DOI: 10.1109/TIM.2025.3644562
"""

import torch
import torch.nn as nn


class _LN(nn.Module):
    """LayerNorm on channel dim for (B, C, T) tensors."""

    def __init__(self, n_ch):
        super().__init__()
        self.ln = nn.LayerNorm(n_ch)

    def forward(self, x):
        return self.ln(x.transpose(-2, -1)).transpose(-2, -1)


# ── Encoder block: Conv→LN→GELU→Dropout (paper Fig. 2 encoder) ──────────

class _EncoderBlock(nn.Module):
    """Single encoder level: Conv1d(k) → LN → GELU → Dropout."""

    def __init__(self, in_ch, out_ch, k=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding="same", bias=True)
        self.ln = _LN(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.ln(self.conv(x))))


# ── Decoder block: pre-norm (LN→GELU→Conv→Drop) × 2 (paper Fig. 2 decoder) ──

class _DecoderBlock(nn.Module):
    """Two pre-norm conv blocks: (LN → GELU → Conv1d(k) → Dropout) × 2.

    Paper Fig. 2 decoder: LN→GELU→Conv1→Drop→LN→GELU→Conv2→Upsample.
    Upsample is handled externally; this block does the two conv sub-blocks.
    """

    def __init__(self, in_ch, out_ch, dropout=0.1, kernel_size=3):
        super().__init__()
        # Sub-block 1: in_ch → out_ch
        self.ln1 = _LN(in_ch)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                               padding="same", bias=True)
        self.drop1 = nn.Dropout(dropout)
        # Sub-block 2: out_ch → out_ch
        self.ln2 = _LN(out_ch)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size,
                               padding="same", bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop1(self.conv1(self.act1(self.ln1(x))))
        x = self.drop2(self.conv2(self.act2(self.ln2(x))))
        return x


class NestedUNet(nn.Module):
    """3-level UNet++ for ECoG decoding (DeepFingerNet, Tao et al. 2025).

    Parameters
    ----------
    n_channels_in : int
        Number of input channels. For faithful DeepFingerNet reproduction
        this is N_electrodes (62 for BCI-IV), NOT N_electrodes × N_freqs.
    n_channels_out : int
        Number of output fingers (5).
    base_ch : int
        Base channel width. Encoder channels = [base_ch, 2x, 4x, 8x].
    kernel_size : int
        Kernel size for encoder convolutions (k=3 in paper).
    dropout : float
        Dropout rate (paper Fig. 2 shows Drop layers in both encoder/decoder).
    """

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int = 5,
        base_ch: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
        decoder_kernel_size: int = 3,
        dense_output: bool = True,
    ):
        super().__init__()
        C = [base_ch * (2**i) for i in range(4)]  # [32, 64, 128, 256]
        k = kernel_size  # encoder kernel size (k=3)
        dk = decoder_kernel_size
        self.dense_output = dense_output

        # ── Encoder: Conv→LN→GELU→Dropout per level ────────────────────────
        self.sr   = _EncoderBlock(n_channels_in, C[0], k, dropout)  # N → 32
        self.enc1 = _EncoderBlock(C[0], C[1], k, dropout)           # 32 → 64
        self.enc2 = _EncoderBlock(C[1], C[2], k, dropout)           # 64 → 128
        self.enc3 = _EncoderBlock(C[2], C[3], k, dropout)           # 128 → 256
        self.pool = nn.MaxPool1d(2)

        # ── First (direct) decoder path: ConvTranspose + DecoderBlock ────────
        self.ct21  = nn.ConvTranspose1d(C[3], C[2], kernel_size=2, stride=2)
        self.dec21 = _DecoderBlock(C[2] + C[2], C[2], dropout, dk)   # 256 → 128

        self.ct11  = nn.ConvTranspose1d(C[2], C[1], kernel_size=2, stride=2)
        self.dec11 = _DecoderBlock(C[1] + C[1], C[1], dropout, dk)   # 128 → 64

        self.ct01  = nn.ConvTranspose1d(C[1], C[0], kernel_size=2, stride=2)
        self.dec01 = _DecoderBlock(C[0] + C[0], C[0], dropout, dk)   # 64  → 32

        # ── Nested decoder paths: Upsample + DecoderBlock ────────────────────
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        # up(X21=128) + enc1(64) + X11(64) = 256  (Dec5)
        self.dec12 = _DecoderBlock(C[2] + C[1] + C[1], C[1], dropout, dk)   # 256 → 64

        # up(X11=64) + enc0(32) + X01(32) = 128  (Dec3)
        self.dec02 = _DecoderBlock(C[1] + C[0] + C[0], C[0], dropout, dk)   # 128 → 32

        # up(X12=64) + enc0(32) + X01(32) + X02(32) = 160  (Dec6; paper: 192)
        self.dec03 = _DecoderBlock(C[1] + C[0] * 3, C[0], dropout, dk)      # 160 → 32

        # ── Output head ─────────────────────────────────────────────────────
        # Dense output: cat(e0, d01, d02, d03) = 4×base_ch = 128ch
        # Matches paper diagram "128 → Output"
        head_ch = C[0] * 4 if dense_output else C[0]
        self.head = nn.Conv1d(head_ch, n_channels_out, kernel_size=1)

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Trim or zero-pad x along time dim to match ref's length (off-by-1 fix)."""
        diff = ref.shape[-1] - x.shape[-1]
        if diff == 0:
            return x
        if diff > 0:
            return torch.nn.functional.pad(x, (0, diff))
        return x[..., :ref.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept 4-D input (B, N, 1, T) from n_wavelets=0 pipeline → (B, N, T)
        if x.dim() == 4:
            B, C, F, T = x.shape
            x = x.reshape(B, C * F, T)

        # ── Encode ──────────────────────────────────────────────────────────
        e0 = self.sr(x)                         # (B, 32,  T)
        e1 = self.enc1(self.pool(e0))            # (B, 64,  T//2)
        e2 = self.enc2(self.pool(e1))            # (B, 128, T//4)
        e3 = self.enc3(self.pool(e2))            # (B, 256, T//8)

        # ── Direct decoder path (ConvTranspose) ─────────────────────────────
        d21 = self.dec21(torch.cat([self._match(self.ct21(e3), e2), e2], dim=1))
        d11 = self.dec11(torch.cat([self._match(self.ct11(e2), e1), e1], dim=1))
        d01 = self.dec01(torch.cat([self._match(self.ct01(e1), e0), e0], dim=1))

        # ── Nested decoder paths (Upsample, preserves exact length) ─────────
        d12 = self.dec12(torch.cat([self._match(self.up(d21), e1), e1,  d11       ], dim=1))
        d02 = self.dec02(torch.cat([self._match(self.up(d11), e0), e0,  d01       ], dim=1))
        d03 = self.dec03(torch.cat([self._match(self.up(d12), e0), e0,  d01,  d02 ], dim=1))

        # Dense output: cat all level-0 features (paper: 32+32+32+32=128→Output)
        if self.dense_output:
            out = torch.cat([e0, d01, d02, d03], dim=1)  # (B, 128, T)
            return self.head(out)
        return self.head(d03)                    # (B, 32, T)
