"""Conformer for ECoG finger flexion decoding.

Combines multi-head self-attention (global temporal context) with
depthwise convolutions (local temporal patterns), using Rotary
Position Embeddings (RoPE) instead of sinusoidal PE.

Architecture per block (Gulati et al., 2020):
    x = x + 0.5 * FFN(x)
    x = x + MHSA(x)          # with RoPE
    x = x + ConvModule(x)    # depthwise temporal conv
    x = x + 0.5 * FFN(x)
    x = LayerNorm(x)

Key differences from vanilla TransformerECoG:
1. Depthwise conv module captures local temporal patterns (oscillations, transients)
2. RoPE gives relative positional bias (better generalization than sinusoidal)
3. Conv stem (k=3) replaces pointwise projection (adds temporal context to embedding)
4. GLU gating in conv module adds multiplicative interaction

References:
- Gulati et al., "Conformer: Convolution-augmented Transformer
  for Speech Recognition", Interspeech 2020
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position
  Embedding", 2021
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ──────────────────────────────────────────────────────────────────

class _TransposedLayerNorm(nn.Module):
    """LayerNorm on (B, C, T) by transposing to (B, T, C)."""

    def __init__(self, n_features):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)

    def forward(self, x):
        return self.norm(x.transpose(-2, -1)).transpose(-2, -1)


# ── Rotary Position Embedding ────────────────────────────────────────────────

class RotaryPositionalEmbedding(nn.Module):
    """Precompute cos/sin tables for RoPE."""

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)        # (T, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)       # (T, dim)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to Q and K.  cos/sin: (T, d_head)."""
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, d_head)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ── Multi-Head Attention with RoPE ──────────────────────────────────────────

class RoPEMultiHeadAttention(nn.Module):
    """Self-attention with Rotary Position Embeddings.

    Uses F.scaled_dot_product_attention (Flash Attention when available).
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.d_head)
        self.attn_dropout = dropout

    def forward(self, x):
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ── Conformer Sub-modules ────────────────────────────────────────────────────

class ConformerFeedForward(nn.Module):
    """Half-step FFN: LN → Linear → Swish → Drop → Linear → Drop."""

    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = F.silu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ConformerConvModule(nn.Module):
    """Convolution module: LN → PW(d→2d) → GLU → DW(k) → BN → Swish → PW(d) → Drop."""

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d)
        x = self.norm(x)
        x = x.transpose(1, 2)          # (B, d, T)
        x = self.pointwise_conv1(x)    # (B, 2d, T)
        x = self.glu(x)                # (B, d, T)
        x = self.depthwise_conv(x)     # (B, d, T)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)    # (B, d, T)
        x = self.dropout(x)
        return x.transpose(1, 2)       # (B, T, d)


# ── Conformer Block ──────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """FFN → MHSA → ConvModule → FFN → LN (Gulati et al., 2020)."""

    def __init__(self, d_model, n_heads, dim_feedforward,
                 conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn_dropout(self.attn(self.attn_norm(x)))
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        x = self.final_norm(x)
        return x


# ── Full Model ───────────────────────────────────────────────────────────────

class ConformerECoG(nn.Module):
    """Conformer for ECoG → finger flexion decoding.

    Parameters
    ----------
    n_channels_in : int
        Flattened input features (n_electrodes * n_freqs).
    n_channels_out : int
        Number of output targets (e.g. 5 fingers).
    d_model : int
        Hidden dimension.
    n_layers : int
        Number of Conformer blocks.
    n_heads : int
        Number of attention heads.
    dim_feedforward : int
        FFN expansion dimension.
    conv_kernel_size : int
        Depthwise conv kernel size in Conformer blocks (default 31 = 310 ms @ 100 Hz).
    stem_kernel_size : int
        Temporal kernel size in the conv stem.
    dropout : float
        Dropout probability.
    eval_window : int
        Window size for sliding-window inference.
    channel_dropout_prob : float
        Per-electrode dropout probability (applied in 4D before flatten).
    """

    def __init__(self, n_channels_in, n_channels_out, d_model=512,
                 n_layers=6, n_heads=8, dim_feedforward=2048,
                 conv_kernel_size=31, stem_kernel_size=3,
                 dropout=0.1, eval_window=256,
                 channel_dropout_prob=0.0):
        super().__init__()
        self.eval_window = eval_window
        self.channel_dropout_prob = channel_dropout_prob

        # Conv stem: adds temporal context to embedding (vs pointwise k=1)
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels_in, d_model, kernel_size=stem_kernel_size,
                      padding=stem_kernel_size // 2, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=stem_kernel_size,
                      padding=stem_kernel_size // 2, bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Conformer blocks (RoPE inside attention — no separate PE needed)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dim_feedforward,
                           conv_kernel_size, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_conv = nn.Conv1d(d_model, n_channels_out, kernel_size=1)

    def _forward_core(self, x):
        """Stem → Conformer blocks → output.  x: (B, C_in, T)."""
        x = self.stem(x)               # (B, d, T)
        x = x.transpose(1, 2)          # (B, T, d)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)          # (B, d, T)
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
        """50%-overlap sliding window for sequences longer than eval_window."""
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
