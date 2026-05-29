# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import argparse

import prettytable as pt
import torch
import triton

from attn.block_sparse_attn_interface import BlockSparseAttention

TORCH_MANUAL_SEED = 42

pns = {
    680: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
    2240: [1, 2, 3, 4, 6, 9, 13, 18, 24, 32],
    9451: [1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64],
}

DEVICE = "cuda"

pt_fwd = pt.PrettyTable()
pt_bwd = pt.PrettyTable()
pt_fwd.field_names = ["N_CTX", "B", "H", "Torch", "Triton", "speedup"]
pt_bwd.field_names = ["N_CTX", "B", "H", "Torch", "Triton", "speedup"]


torch.manual_seed(TORCH_MANUAL_SEED)
torch.cuda.manual_seed(TORCH_MANUAL_SEED)

depths = [20, 24, 30, 36]
batch_sizes = [4]


def benchmark(sparsity_pattern="block_diagonal"):
    for N in pns.keys():
        for batch in batch_sizes:
            for H in depths:

                B, H, N_CTX, HEAD_DIM = (batch, H, N, 64)
                dtype = torch.float16

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

                # reference implementation
                patch_nums = pns[N]
                patch_nums_sq = [pn * pn for pn in patch_nums]

                L = sum(pn * pn for pn in patch_nums)
                d = torch.cat(
                    [torch.full((pn * pn,), i) for i, pn in enumerate(patch_nums)]
                ).view(L, 1)
                dT = d.transpose(0, 1)  # dT: 11L

                # reference implementation call torch.sdpa which should call mem efficient attention under the hood, performance should be similar to using xformers
                if sparsity_pattern == "block_diagonal":
                    M = torch.where(d == dT, True, False).to(DEVICE)
                elif sparsity_pattern == "block_causal":
                    M = torch.where(d >= dT, True, False).to(DEVICE)

                fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=M, scale=sm_scale
                )
                fn()
                ms_torch_fwd = triton.testing.do_bench(fn)
                out_torch = fn().half()

                fn = lambda: out_torch.backward(dout, retain_graph=True)
                ms_torch_bwd = triton.testing.do_bench(fn)

                attention = BlockSparseAttention(
                    block_sizes=patch_nums_sq, sparsity_pattern=sparsity_pattern
                ).to(DEVICE)
                _ = attention(q, k, v, sm_scale).half()
                fn = lambda: attention(q, k, v, sm_scale)
                fn()
                ms_triton_fwd = triton.testing.do_bench(fn)
                out_triton = fn().half()

                fn = lambda: out_triton.backward(dout, retain_graph=True)
                ms_triton_bwd = triton.testing.do_bench(fn)

                pt_fwd.add_row(
                    [
                        N_CTX,
                        B,
                        H,
                        ms_torch_fwd,
                        ms_triton_fwd,
                        ms_torch_fwd / ms_triton_fwd,
                    ],
                    divider=batch == batch_sizes[-1] and H == depths[-1],
                )
                pt_bwd.add_row(
                    [
                        N_CTX,
                        B,
                        H,
                        ms_torch_bwd,
                        ms_triton_bwd,
                        ms_torch_bwd / ms_triton_bwd,
                    ],
                    divider=batch == batch_sizes[-1] and H == depths[-1],
                )

                torch.cuda.empty_cache()

    pt_fwd.float_format = "0.4"
    pt_bwd.float_format = "0.4"

    pt_fwd.title = "Forward Benchmark Results"
    print(pt_fwd.get_string())

    pt_bwd.title = "Backward Benchmark Results"
    print(pt_bwd.get_string())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark attention kernels")
    parser.add_argument(
        "-sp",
        "--sparsity_pattern",
        default="block_diagonal",
        choices=["block_diagonal", "block_causal"],
        help="Sparsity pattern for block sparse attention",
        type=str,
    )

    args = parser.parse_args()

    benchmark(args.sparsity_pattern)
