"""TransformerV2 — vanilla transformer + RoPE + conv module.

Targeted improvement over TransformerECoG: keeps the well-tuned
single-FFN architecture (78M → ~97M params) and adds only:

1. **RoPE** (Rotary Position Embeddings) replacing sinusoidal PE.
   Zero extra params, better relative position generalization.

2. **Depthwise conv module** per block for local temporal patterns.
   Adds ~19M params for d=1024 (controlled vs Conformer's +77M).

Block structure (vs vanilla):
    Vanilla:   LN → MHSA → residual  →  LN → FFN → residual
    V2:        LN → MHSA(RoPE) → residual → ConvModule → residual → LN → FFN → residual

This avoids the full Conformer's parameter doubling (2x FFN) that caused
massive overfitting at d=1024 (155M vs 78M vanilla on 33K training samples).
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


# ── RoPE ─────────────────────────────────────────────────────────────────────

class RotaryPositionalEmbedding(nn.Module):
    """Precompute cos/sin tables for Rotary Position Embeddings."""

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)         # (T, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (T, dim)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ── RoPE Multi-Head Attention ────────────────────────────────────────────────

class RoPEMultiHeadAttention(nn.Module):
    """Self-attention with RoPE. Uses F.scaled_dot_product_attention."""

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


# ── Depthwise Conv Module ────────────────────────────────────────────────────

class ConvModule(nn.Module):
    """LN → PW(d→2d) → GLU → DW-Conv(k) → BN → Swish → PW(d) → Drop.

    Captures local temporal patterns (oscillations, transients) that
    attention alone misses. Only ~3M params per layer at d=1024.
    """

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


# ── Transformer V2 Block ────────────────────────────────────────────────────

class TransformerV2Block(nn.Module):
    """Pre-LN attention + conv module + pre-LN FFN.

    Like vanilla TransformerEncoderLayer but with:
    - RoPE attention instead of absolute PE + standard attention
    - Conv module between attention and FFN
    """

    def __init__(self, d_model, n_heads, dim_feedforward,
                 conv_kernel_size=31, dropout=0.1):
        super().__init__()
        # Attention with RoPE
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Conv module
        self.conv_module = ConvModule(d_model, conv_kernel_size, dropout)

        # FFN (same as vanilla transformer — single FFN, not doubled)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-LN attention with RoPE
        x = x + self.attn_dropout(self.attn(self.attn_norm(x)))
        # Conv module (residual)
        x = x + self.conv_module(x)
        # Pre-LN FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── Full Model ───────────────────────────────────────────────────────────────

class TransformerV2ECoG(nn.Module):
    """TransformerV2 for ECoG → finger flexion decoding.

    Vanilla transformer architecture + RoPE + conv module per block.
    ~97M params at d=1024 (vs 78M vanilla, 155M full Conformer).

    Parameters
    ----------
    n_channels_in : int
        Flattened input features (n_electrodes * n_freqs).
    n_channels_out : int
        Number of output targets (e.g. 5 fingers).
    d_model : int
        Hidden dimension.
    n_layers : int
        Number of transformer blocks.
    n_heads : int
        Number of attention heads.
    dim_feedforward : int
        FFN expansion dimension.
    spatial_kernel_size : int
        Kernel size for spatial reduction conv.
    conv_kernel_size : int
        Kernel size for depthwise conv module in each block.
    dropout : float
        Dropout probability.
    eval_window : int
        Window size for sliding-window inference.
    channel_dropout_prob : float
        Per-electrode dropout probability.
    """

    def __init__(self, n_channels_in, n_channels_out, d_model=1024,
                 n_layers=6, n_heads=16, dim_feedforward=4096,
                 spatial_kernel_size=1, conv_kernel_size=31,
                 dropout=0.1, eval_window=256,
                 channel_dropout_prob=0.0):
        super().__init__()
        self.eval_window = eval_window
        self.channel_dropout_prob = channel_dropout_prob

        # Spatial reduction (same as vanilla — keep what works)
        self.spatial_reduce = nn.Sequential(
            nn.Conv1d(n_channels_in, d_model, kernel_size=spatial_kernel_size,
                      padding="same", bias=False),
            _TransposedLayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer V2 blocks (RoPE inside — no separate PE needed)
        self.blocks = nn.ModuleList([
            TransformerV2Block(d_model, n_heads, dim_feedforward,
                               conv_kernel_size, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_conv = nn.Conv1d(d_model, n_channels_out, kernel_size=1)

    def _forward_core(self, x):
        x = self.spatial_reduce(x)     # (B, d, T)
        x = x.transpose(1, 2)          # (B, T, d)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)          # (B, d, T)
        return self.output_conv(x)

    def forward(self, x):
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
