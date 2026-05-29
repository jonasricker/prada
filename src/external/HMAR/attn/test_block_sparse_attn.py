# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import pytest
import torch

from .block_sparse_attn_interface import BlockSparseAttention

TORCH_MANUAL_SEED = 42

pns = {
    680: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
    2240: [1, 2, 3, 4, 6, 9, 13, 18, 24, 32],
    9451: [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64],
}

DEVICE = "cuda"

test_cfgs = [
    (B, H, N_CTX, HEAD_DIM, SPARSITY_PATTERN)
    for B in [1, 2]
    for H in [16, 20, 24]
    for N_CTX in [680, 2240]
    for HEAD_DIM in [64, 128]
    for SPARSITY_PATTERN in ["block_diagonal", "block_causal"]
]


@pytest.mark.parametrize("B, H, N_CTX, HEAD_DIM, SPARSITY_PATTERN", test_cfgs)
def test_op(B, H, N_CTX, HEAD_DIM, SPARSITY_PATTERN, dtype=torch.float16):
    torch.manual_seed(TORCH_MANUAL_SEED)

    q = (
        torch.empty((B, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((B, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((B, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    sm_scale = 0.5
    dout = torch.randn_like(q)

    # setup block diagonal sizes
    patch_nums = pns[N_CTX]
    patch_nums_sq = [pn * pn for pn in patch_nums]

    L = sum(pn * pn for pn in patch_nums)
    d = torch.cat([torch.full((pn * pn,), i) for i, pn in enumerate(patch_nums)]).view(
        L, 1
    )
    dT = d.transpose(0, 1)  # dT: 11L

    if SPARSITY_PATTERN == "block_diagonal":
        M = torch.where(d == dT, 1, 0)
    elif SPARSITY_PATTERN == "block_causal":
        M = torch.where(d >= dT, 1, 0)

    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()

    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    # custom implementation
    attention = BlockSparseAttention(
        block_sizes=patch_nums_sq, sparsity_pattern=SPARSITY_PATTERN
    ).to(DEVICE)

    tri_out = attention(q, k, v, sm_scale).half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None

    # compare
    rtol = 1e-2
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)

    torch.cuda.empty_cache()
