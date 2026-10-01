"""Efficient state-space (S4D) decoder for continuous ECoG regression.

Tests the paper's Finding-1 frontier: does an efficient *state-space* inductive bias
match or beat DTCNet at iso-parameter budget? Adapts the Cortical-SSM idea (S4/S5, no
patchification; S4-family chosen over Mamba for continuous signals) to seq2seq
regression. Each block: a diagonal S4D layer (Gu et al., 2022) mixing over time +
GLU channel mixing, residual. Length-preserving -> plugs into full-signal Pearson-r eval.

Input:  (B, C, F, T) spectrogram (reshaped to (B, C*F, T))  ->  (B, n_targets, T).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DKernel(nn.Module):
    """Diagonal state-space convolution kernel (S4D)."""

    def __init__(self, d_model, d_state=64):
        super().__init__()
        H, N = d_model, d_state
        log_dt = torch.rand(H) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)
        self.log_dt = nn.Parameter(log_dt)                                   # (H,)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N)))    # stable (Re A < 0)
        self.A_imag = nn.Parameter(math.pi * torch.arange(N).float().unsqueeze(0).repeat(H, 1))
        self.C = nn.Parameter(torch.randn(H, N, 2) / math.sqrt(N))           # complex as (...,2)

    def forward(self, L):
        dt = torch.exp(self.log_dt)                                # (H,)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag          # (H, N) complex
        C = torch.view_as_complex(self.C)                          # (H, N)
        dtA = dt.unsqueeze(-1) * A                                  # (H, N)
        ell = torch.arange(L, device=dt.device)
        K = torch.exp(dtA.unsqueeze(-1) * ell)                     # (H, N, L)
        return 2.0 * torch.einsum("hn,hnl->hl", C, K).real          # (H, L) real kernel


class S4DBlock(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.kernel = S4DKernel(d_model, d_state)
        self.D = nn.Parameter(torch.randn(d_model))
        self.mix = nn.Conv1d(d_model, 2 * d_model, 1)  # -> GLU channel mixing
        self.drop = nn.Dropout(dropout)

    def _ssm(self, u):  # u: (B, H, L)
        L = u.shape[-1]
        K = self.kernel(L)                            # (H, L)
        n = 2 * L
        Kf = torch.fft.rfft(K, n=n)                   # (H, n//2+1)
        uf = torch.fft.rfft(u, n=n)                   # (B, H, n//2+1)
        y = torch.fft.irfft(uf * Kf, n=n)[..., :L]    # (B, H, L) causal conv
        return y + u * self.D.unsqueeze(-1)

    def forward(self, x):  # (B, H, L)
        u = self.norm(x.transpose(1, 2)).transpose(1, 2)
        y = F.gelu(self._ssm(u))
        y = self.drop(F.glu(self.mix(y), dim=1))
        return x + y


class SSMDecoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, d_model=256, d_state=64,
                 n_layers=6, dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_channels_in, d_model, 1)
        self.blocks = nn.ModuleList([S4DBlock(d_model, d_state, dropout) for _ in range(n_layers)])
        self.out = nn.Conv1d(d_model, n_channels_out, 1)

    def forward(self, x):
        if x.dim() == 4:                     # (B, C, F, T) -> (B, C*F, T)
            B, C, Fdim, T = x.shape
            x = x.reshape(B, C * Fdim, T)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)
