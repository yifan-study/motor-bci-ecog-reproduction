#!/usr/bin/env python
"""Verify DTCNet + bottleneck self-attention implementation.

Checks:
1. DTCNet unchanged: original DTCNet has no extra params
2. Param count: DTCNetAttn with n_attn_layers=2 → ~11.5M
3. Output shapes: (B,5,256), (B,5,1024), and 4D input
4. Gradient flow: attention layers and encoder both receive gradients
5. Numerical identity: DTCNetAttn produces same encoder/decoder as DTCNet base
"""

import sys
sys.path.insert(0, ".")

import torch
from src.models.dtcnet import DTCNet
from src.models.dtcnet_attn import DTCNetAttn


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_base_unchanged():
    """Original DTCNet should have no attention params."""
    n_in = 62 * 40
    m = DTCNet(n_in, 5, dropout=0.1)
    p = count_params(m)
    assert not hasattr(m, "bottleneck_attn"), "DTCNet should not have bottleneck_attn"
    print(f"  [PASS] DTCNet unchanged: {p:,} params, no attention attrs")
    return p


def test_param_count(baseline_params):
    """DTCNetAttn with 2 layers should add ~6.3M params."""
    n_in = 62 * 40
    m = DTCNetAttn(n_in, 5, dropout=0.1, n_attn_layers=2, n_attn_heads=8,
                   attn_dim_feedforward=2048)
    p = count_params(m)
    delta = p - baseline_params
    print(f"  [PASS] DTCNetAttn: {p:,} params (+{delta:,} over baseline)")
    assert 5_000_000 < delta < 8_000_000, f"Delta {delta} outside expected range"


def test_output_shapes():
    """Check output shape for various input shapes."""
    n_in = 62 * 40
    m = DTCNetAttn(n_in, 5, dropout=0.1, n_attn_layers=2)
    m.eval()

    # Training window: (B, C*F, 256)
    x1 = torch.randn(2, n_in, 256)
    y1 = m(x1)
    assert y1.shape == (2, 5, 256), f"Expected (2,5,256), got {y1.shape}"
    print(f"  [PASS] Shape (B,C*F,256) → {y1.shape}")

    # Longer input: (B, C*F, 1024)
    x2 = torch.randn(2, n_in, 1024)
    y2 = m(x2)
    assert y2.shape == (2, 5, 1024), f"Expected (2,5,1024), got {y2.shape}"
    print(f"  [PASS] Shape (B,C*F,1024) → {y2.shape}")

    # 4D input: (B, C, F, T)
    x3 = torch.randn(2, 62, 40, 256)
    y3 = m(x3)
    assert y3.shape == (2, 5, 256), f"Expected (2,5,256), got {y3.shape}"
    print(f"  [PASS] Shape (B,C,F,T) 4D → {y3.shape}")


def test_gradient_flow():
    """Verify gradients flow through both attention and encoder."""
    n_in = 62 * 40
    m = DTCNetAttn(n_in, 5, dropout=0.1, n_attn_layers=2)
    m.train()

    x = torch.randn(2, n_in, 256)
    y = m(x)
    loss = y.sum()
    loss.backward()

    # Check encoder conv gets gradient
    enc_grad = m.enc[0].conv.weight.grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0, "No gradient in encoder"

    # Check attention layer gets gradient
    attn_param = next(m.bottleneck_attn.parameters())
    assert attn_param.grad is not None and attn_param.grad.abs().sum() > 0, \
        "No gradient in attention"

    # Check bottleneck_norm gets gradient
    norm_grad = m.bottleneck_norm.weight.grad
    assert norm_grad is not None and norm_grad.abs().sum() > 0, \
        "No gradient in bottleneck_norm"

    print("  [PASS] Gradients flow through encoder, attention, and norm")


def test_numerical_identity():
    """DTCNetAttn should share encoder/decoder weights with DTCNet when loaded."""
    n_in = 62 * 40
    torch.manual_seed(123)
    m_base = DTCNet(n_in, 5, dropout=0.1)

    # Load base weights into DTCNetAttn (strict=False to skip attention params)
    torch.manual_seed(999)  # different seed — attn params will differ
    m_attn = DTCNetAttn(n_in, 5, dropout=0.1, n_attn_layers=2)
    m_attn.load_state_dict(m_base.state_dict(), strict=False)

    # Verify shared encoder/decoder weights are identical
    for name, p_base in m_base.named_parameters():
        p_attn = dict(m_attn.named_parameters())[name]
        assert torch.equal(p_base, p_attn), f"Weight mismatch in {name}"

    print("  [PASS] Encoder/decoder weights transfer correctly from DTCNet")


if __name__ == "__main__":
    print("1. Base DTCNet unchanged")
    bp = test_base_unchanged()

    print("\n2. Param count with attention")
    test_param_count(bp)

    print("\n3. Output shapes")
    test_output_shapes()

    print("\n4. Gradient flow")
    test_gradient_flow()

    print("\n5. Weight compatibility")
    test_numerical_identity()

    print("\n=== ALL CHECKS PASSED ===")
