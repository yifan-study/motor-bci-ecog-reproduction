"""Transformer encoder for ECoG → finger flexion decoding.

Classes
-------
TransformerECoG
    Flat single-scale transformer with optional spatial bottleneck.
MultiscaleTransformerECoG
    3-scale hierarchical transformer (T → T/4 → T/16) with skip connections.
HybridTransformerECoG
    Legacy 2-scale hybrid (kept for reference).

Input:  (batch, n_electrodes, n_freqs, time) — wavelet spectrograms
Output: (batch, n_channels_out, time)         — predicted finger flexion
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Dynamically expands the buffer if the input sequence is longer than
    the current buffer, so it works for both 256-step training windows
    and full-signal evaluation (up to ~20k steps).
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # Register div_term as buffer so it moves with device
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)
        # Start with a small buffer; will be expanded on first forward
        self.register_buffer("pe", torch.zeros(1, 1, d_model))

    def _expand_pe(self, length):
        """Recompute PE buffer to cover at least `length` positions."""
        pe = torch.zeros(length, self.d_model, device=self.div_term.device)
        position = torch.arange(length, dtype=torch.float, device=self.div_term.device).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        self.pe = pe.unsqueeze(0)  # (1, length, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        if x.size(1) > self.pe.size(1):
            self._expand_pe(x.size(1))
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class _TransposedLayerNorm(nn.Module):
    """LayerNorm that works on (B, C, T) by transposing to (B, T, C)."""

    def __init__(self, n_features):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)

    def forward(self, x):
        return self.norm(x.transpose(-2, -1)).transpose(-2, -1)


class _SwiGLUTransformerEncoderLayer(nn.Module):
    """Pre-LN TransformerEncoderLayer with SwiGLU feedforward.

    SwiGLU FFN: out = W3( SiLU(W1(x)) ⊙ W2(x) )
    Uses 3 weight matrices instead of 2, so for equal parameter count use
    dim_feedforward ≈ 2/3 × standard_dim_feedforward (e.g. 2730 vs 4096).

    Drop-in replacement for nn.TransformerEncoderLayer(norm_first=True, batch_first=True).
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, bias=False
        )
        # SwiGLU: two gate matrices + one output matrix
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w3 = nn.Linear(dim_feedforward, d_model, bias=False)
        # Pre-LN norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # Pre-LN self-attention
        normed = self.norm1(src)
        attn_out, _ = self.self_attn(normed, normed, normed,
                                      attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask,
                                      need_weights=False)
        src = src + self.attn_drop(attn_out)
        # Pre-LN SwiGLU FFN
        normed = self.norm2(src)
        src = src + self.ffn_drop(self.w3(F.silu(self.w1(normed)) * self.w2(normed)))
        return src


class TransformerECoG(nn.Module):
    """Transformer encoder for ECoG decoding.

    Parameters
    ----------
    n_channels_in : int
        Flattened input features (n_electrodes * n_freqs).
    n_channels_out : int
        Number of output targets (e.g. 5 fingers).
    d_model : int
        Transformer hidden dimension.
    n_layers : int
        Number of TransformerEncoderLayer blocks.
    n_heads : int
        Number of attention heads.
    dim_feedforward : int
        Feedforward hidden dimension in each layer.
    spatial_kernel_size : int
        Kernel size for the spatial reduction Conv1d.
    spatial_bottleneck_dim : int
        If > 0, use a 2-stage spatial compression:
        n_channels_in → spatial_bottleneck_dim → d_model.
        Mimics U-Net's aggressive spatial bottleneck (e.g. 1920→128→1024).
        If 0 (default), compress directly n_channels_in → d_model.
    ffn_type : str
        Feedforward type: 'gelu' (default) or 'swiglu'.
        SwiGLU uses 3 weight matrices; for equal params use
        dim_feedforward ≈ 2/3 × standard value (e.g. 2730 vs 4096).
    dropout : float
        Dropout probability.
    """

    def __init__(self, n_channels_in, n_channels_out, d_model=64,
                 n_layers=4, n_heads=4, dim_feedforward=256,
                 spatial_kernel_size=3, spatial_bottleneck_dim=0,
                 ffn_type='gelu', dropout=0.1, eval_window=256,
                 channel_dropout_prob=0.0):
        super().__init__()
        self.eval_window = eval_window
        self.channel_dropout_prob = channel_dropout_prob

        # Spatial reduction: flatten (ch * freq) → d_model
        # Optional 2-stage bottleneck: n_in → bottleneck → d_model
        if spatial_bottleneck_dim > 0:
            self.spatial_reduce = nn.Sequential(
                nn.Conv1d(n_channels_in, spatial_bottleneck_dim,
                          kernel_size=spatial_kernel_size, padding="same", bias=False),
                _TransposedLayerNorm(spatial_bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(spatial_bottleneck_dim, d_model, kernel_size=1, bias=False),
                _TransposedLayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.spatial_reduce = nn.Sequential(
                nn.Conv1d(n_channels_in, d_model, kernel_size=spatial_kernel_size,
                          padding="same", bias=False),
                _TransposedLayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder stack (pre-LN, bidirectional)
        if ffn_type == 'swiglu':
            self.transformer = nn.ModuleList([
                _SwiGLUTransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ])
        else:
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

        # Output projection
        self.output_conv = nn.Conv1d(d_model, n_channels_out, kernel_size=1)

    def _forward_core(self, x):
        """Run spatial reduction → PE → transformer → output projection.

        x: (B, C_in, T) with T <= eval_window.
        Returns: (B, n_out, T)
        """
        x = self.spatial_reduce(x)
        x = x.transpose(1, 2)       # (B, T, d_model)
        x = self.pos_enc(x)
        if isinstance(self.transformer, nn.ModuleList):
            for layer in self.transformer:
                x = layer(x)
        else:
            x = self.transformer(x)
        x = x.transpose(1, 2)       # (B, d_model, T)
        return self.output_conv(x)

    def forward(self, x):
        # Handle 4D spectrogram input: (B, C, W, T) → (B, C*W, T)
        if x.ndim == 4:
            b, c, w, t = x.shape
            # Channel dropout: zero entire electrodes (all wavelet features)
            if self.training and self.channel_dropout_prob > 0:
                mask = torch.bernoulli(
                    torch.full((b, c, 1, 1), 1.0 - self.channel_dropout_prob, device=x.device)
                )
                x = x * mask / (1.0 - self.channel_dropout_prob)
            x = x.reshape(b, c * w, t)

        T = x.shape[-1]

        # Short sequences (training windows): direct forward
        if T <= self.eval_window or self.training:
            return self._forward_core(x)

        # Long sequences (full-signal eval): sliding-window inference
        # to match the attention context the model was trained on.
        return self._sliding_window_forward(x)

    def _sliding_window_forward(self, x):
        """Sliding-window inference for sequences longer than eval_window.

        Uses 50% overlap and averages predictions in overlapping regions.
        This ensures the transformer sees the same attention context length
        it was trained on (eval_window), with consistent positional encoding.
        """
        B, C, T = x.shape
        W = self.eval_window
        stride = W // 2  # 50% overlap

        # Accumulate predictions and counts for averaging
        out_sum = x.new_zeros(B, self.output_conv.out_channels, T)
        counts = x.new_zeros(1, 1, T)

        for start in range(0, T, stride):
            end = min(start + W, T)
            chunk = x[:, :, start:end]

            # Pad last chunk to eval_window if needed
            pad_len = W - chunk.shape[-1]
            if pad_len > 0:
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

            pred = self._forward_core(chunk)

            # Remove padding from prediction
            if pad_len > 0:
                pred = pred[:, :, :W - pad_len]

            out_sum[:, :, start:end] += pred
            counts[:, :, start:end] += 1

        return out_sum / counts


class MultiscaleTransformerECoG(nn.Module):
    """3-scale hierarchical Transformer for ECoG decoding.

    Encoder: full-res (T) → T/4 → T/16, with transformer blocks at each scale.
    Decoder: T/16 → T/4 → T with U-Net-style skip connections.

    This directly addresses the flat transformer's single-resolution limitation.
    Each scale sees the same temporal duration (2.56 s for T=256) but at
    different temporal resolutions, giving the model multi-scale context.

    Parameters
    ----------
    n_channels_in : int
        Flattened input features (n_electrodes * n_freqs).
    n_channels_out : int
        Number of output targets (e.g. 5 fingers).
    d_model : int
        Hidden dimension shared across all scales.
    n_layers : int
        Total transformer layers, split evenly across 3 scales.
        Must be divisible by 3 for equal split; remainder goes to bottleneck.
    n_heads : int
        Number of attention heads (same at all scales).
    dim_feedforward : int
        FFN hidden dimension (same at all scales).
    spatial_kernel_size : int
        Kernel size for the initial spatial reduction.
    dropout : float
        Dropout probability.
    downsample_factor : int
        Temporal stride for each downsampling step (default 4: T→T/4→T/16).
    """

    def __init__(self, n_channels_in, n_channels_out, d_model=1024,
                 n_layers=6, n_heads=16, dim_feedforward=4096,
                 spatial_kernel_size=1, dropout=0.1,
                 downsample_factor=4, eval_window=256):
        super().__init__()
        self.eval_window = eval_window
        self.downsample_factor = downsample_factor

        # Split layers across 3 scales: [n//3, n//3, remaining]
        n_s1 = n_layers // 3
        n_s2 = n_layers // 3
        n_s3 = n_layers - 2 * (n_layers // 3)

        # ── Spatial reduction (same as flat transformer) ────────────────────
        self.spatial_reduce = nn.Sequential(
            nn.Conv1d(n_channels_in, d_model, kernel_size=spatial_kernel_size,
                      padding="same", bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Scale 1 encoder: full resolution T ─────────────────────────────
        self.pe1 = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.enc1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", norm_first=True, batch_first=True,
            ),
            num_layers=n_s1,
        )

        # Downsample 1: T → T/factor  (kernel=2*factor, stride=factor, pad=factor//2)
        f = downsample_factor
        self.down1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=2 * f, stride=f,
                      padding=f // 2, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
        )

        # ── Scale 2 encoder: T/4 ───────────────────────────────────────────
        self.pe2 = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.enc2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", norm_first=True, batch_first=True,
            ),
            num_layers=n_s2,
        )

        # Downsample 2: T/4 → T/16
        self.down2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=2 * f, stride=f,
                      padding=f // 2, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
        )

        # ── Scale 3 encoder: T/16 (bottleneck) ─────────────────────────────
        self.pe3 = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.enc3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", norm_first=True, batch_first=True,
            ),
            num_layers=n_s3,
        )

        # ── Decoder: upsample + skip connections ────────────────────────────
        # Up1: T/16 → T/4 + skip from enc2
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, kernel_size=2 * f, stride=f,
                               padding=f // 2, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
        )
        self.merge1 = nn.Sequential(
            nn.Conv1d(2 * d_model, d_model, kernel_size=1, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Up2: T/4 → T + skip from enc1
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, kernel_size=2 * f, stride=f,
                               padding=f // 2, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
        )
        self.merge2 = nn.Sequential(
            nn.Conv1d(2 * d_model, d_model, kernel_size=1, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output projection
        self.output_conv = nn.Conv1d(d_model, n_channels_out, kernel_size=1)

    def _encode(self, x, pe, enc):
        """PE + transformer encoder. x: (B, d_model, T) → (B, d_model, T)."""
        x = pe(x.transpose(1, 2))   # (B, T, d_model)
        x = enc(x)                   # (B, T, d_model)
        return x.transpose(1, 2)    # (B, d_model, T)

    def _forward_core(self, x):
        """x: (B, C_in, T). Returns: (B, n_out, T)."""
        x = self.spatial_reduce(x)             # (B, d_model, T)

        # Encoder
        x = self._encode(x, self.pe1, self.enc1)  # (B, d_model, T)
        skip1 = x
        x = self.down1(x)                          # (B, d_model, T/4)

        x = self._encode(x, self.pe2, self.enc2)  # (B, d_model, T/4)
        skip2 = x
        x = self.down2(x)                          # (B, d_model, T/16)

        x = self._encode(x, self.pe3, self.enc3)  # (B, d_model, T/16)

        # Decoder
        x = self.up1(x)[:, :, :skip2.shape[-1]]   # (B, d_model, T/4)
        x = self.merge1(torch.cat([x, skip2], dim=1))

        x = self.up2(x)[:, :, :skip1.shape[-1]]   # (B, d_model, T)
        x = self.merge2(torch.cat([x, skip1], dim=1))

        return self.output_conv(x)

    def forward(self, x):
        if x.ndim == 4:
            b, c, w, t = x.shape
            x = x.reshape(b, c * w, t)

        T = x.shape[-1]
        if T <= self.eval_window or self.training:
            return self._forward_core(x)
        return self._sliding_window_forward(x)

    def _sliding_window_forward(self, x):
        """Sliding-window inference for sequences longer than eval_window."""
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
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

            pred = self._forward_core(chunk)

            if pad_len > 0:
                pred = pred[:, :, :W - pad_len]

            out_sum[:, :, start:end] += pred
            counts[:, :, start:end] += 1

        return out_sum / counts


class HybridTransformerECoG(nn.Module):
    """Transformer with single-level downsample/upsample + skip connection.

    Adds multi-scale inductive bias inspired by U-Net's success:
      - Spatial reduce to d_model at full resolution
      - Save skip connection
      - Downsample by 2 (strided conv) → transformer at half resolution
      - Upsample back (transposed conv) + add skip
      - Output projection

    The transformer operates at T/2, halving attention cost and doubling
    effective receptive field, while skip connection preserves fine detail.
    """

    def __init__(self, n_channels_in, n_channels_out, d_model=256,
                 n_layers=4, n_heads=4, dim_feedforward=512,
                 spatial_kernel_size=1, dropout=0.1,
                 eval_window=256):
        super().__init__()
        self.eval_window = eval_window

        # Spatial reduction: (ch * freq) → d_model at full resolution
        self.spatial_reduce = nn.Sequential(
            nn.Conv1d(n_channels_in, d_model, kernel_size=spatial_kernel_size,
                      padding="same", bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Downsample: stride=2, T → T/2
        self.downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
        )

        # Positional encoding + transformer at half resolution
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

        # Upsample: T/2 → T
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
        )

        # Merge skip + upsampled (2*d_model → d_model)
        self.skip_merge = nn.Sequential(
            nn.Conv1d(2 * d_model, d_model, kernel_size=1, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output projection
        self.output_conv = nn.Conv1d(d_model, n_channels_out, kernel_size=1)

    def _forward_core(self, x):
        """x: (B, C_in, T). Returns: (B, n_out, T)."""
        # Spatial reduce at full resolution
        x = self.spatial_reduce(x)  # (B, d_model, T)
        skip = x

        # Downsample → transformer → upsample
        x = self.downsample(x)      # (B, d_model, T/2)
        T_down = x.shape[-1]

        x = x.transpose(1, 2)       # (B, T/2, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)       # (B, d_model, T/2)

        x = self.upsample(x)        # (B, d_model, T')
        # Match skip length exactly (handles odd-length rounding)
        x = x[:, :, :skip.shape[-1]]

        # Merge skip connection
        x = self.skip_merge(torch.cat([x, skip], dim=1))  # (B, d_model, T)

        return self.output_conv(x)

    def forward(self, x):
        if x.ndim == 4:
            b, c, w, t = x.shape
            x = x.reshape(b, c * w, t)

        T = x.shape[-1]
        if T <= self.eval_window or self.training:
            return self._forward_core(x)
        return self._sliding_window_forward(x)

    def _sliding_window_forward(self, x):
        """Sliding-window inference for long sequences."""
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
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

            pred = self._forward_core(chunk)

            if pad_len > 0:
                pred = pred[:, :, :W - pad_len]

            out_sum[:, :, start:end] += pred
            counts[:, :, start:end] += 1

        return out_sum / counts
