# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

from typing import List

import torch
from torch import nn

from .block_sparse_attn_triton import block_sparse_attn_func


class BlockSparseAttention(nn.Module):
    def __init__(
        self, block_sizes=List[int], device="cuda", sparsity_pattern="block_diagonal"
    ):
        super().__init__()

        self.N_CTX = sum(block_sizes)

        bs_cum_sums = torch.cumsum(torch.tensor([0] + block_sizes), 0)

        self.row_ends = torch.cat(
            [
                torch.full((block_size,), bs_cum_sums[i + 1])
                for i, block_size in enumerate(block_sizes)
            ]
        ).to(device=device, dtype=torch.int32)
        
        self.cum_sums = torch.cat(
            [
                torch.full((block_size,), bs_cum_sums[i])
                for i, block_size in enumerate(block_sizes)
            ]
        ).to(device=device, dtype=torch.int32)

        self.sparsity_pattern = sparsity_pattern

        if sparsity_pattern == "block_diagonal":
            self.row_starts = torch.cat(
                [
                    torch.full((block_size,), bs_cum_sums[i])
                    for i, block_size in enumerate(block_sizes)
                ]
            ).to(device=device, dtype=torch.int32)
        elif sparsity_pattern == "block_causal":
            self.row_starts = torch.zeros_like(self.row_ends)
        else:
            raise ValueError(f"Unknown sparsity pattern: {sparsity_pattern}")
        
    def forward(self, q, k, v, sm_scale):
        N_CTX = q.shape[2]
        assert (
            self.N_CTX == N_CTX
        ), f"Sum of block sizes ({self.N_CTX}) must equal N_CTX ({N_CTX})"

        return block_sparse_attn_func(
            q,
            k,
            v,
            self.row_starts,
            self.row_ends,
            self.cum_sums,
            sm_scale,
            self.sparsity_pattern,
        )
