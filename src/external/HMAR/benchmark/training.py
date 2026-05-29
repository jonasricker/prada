# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import prettytable as pt
import torch
import triton

import dist
from models import build_vae_nsp
from utils import arg_util

TORCH_MANUAL_SEED = 42

reso_pns = {
    256: (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    512: (1, 2, 3, 4, 6, 9, 13, 18, 24, 32),
    1024: (1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64),
}

num_classes = 1000

torch.manual_seed(TORCH_MANUAL_SEED)
torch.cuda.manual_seed(TORCH_MANUAL_SEED)


def build_nsp(args: arg_util.Args, using_block_sparse_attn=False):
    _, nsp = build_vae_nsp(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=dist.get_device(),
        patch_nums=args.patch_nums,
        num_classes=num_classes,
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini,
        using_block_sparse_attn=using_block_sparse_attn,
    )

    return nsp


results = pt.PrettyTable()

results.field_names = ["Resolution", "B", "H", "Torch", "Triton", "speedup"]

args: arg_util.Args = arg_util.init_dist_and_get_args(
    init_dist=False, validate_args=False
)

depths = [20, 24, 30, 36]
batch_sizes = [2, 4, 8]

for reso in reso_pns.keys():
    for depth in depths:
        for batch in batch_sizes:
            patch_nums = reso_pns[reso]
            L = sum(pn * pn for pn in patch_nums)
            args.patch_nums = patch_nums
            args.depth = depth

            try:
                labels = torch.randint(0, num_classes, (batch,))
                x = torch.randn(
                    (batch, L - 1, 32),
                    device=dist.get_device(),
                    requires_grad=True,
                    dtype=torch.float16,
                )

                do = torch.rand(
                    (batch, L, 4096), device=dist.get_device(), dtype=torch.float16
                )

                labels = labels.to(dist.get_device())

                nsp_torch = build_nsp(args, using_block_sparse_attn=False)
                nsp_torch = nsp_torch.to(dist.get_device())
                nsp_torch = args.compile_model(nsp_torch, args.tfast)

                def torch_attn(labels, x, do):
                    with torch.amp.autocast("cuda"):
                        nsp_torch(labels, x).backward(do, retain_graph=True)

                fn = lambda: torch_attn(labels, x, do)
                torch_ms = triton.testing.do_bench(fn)

            except torch.cuda.OutOfMemoryError as e:
                torch_ms = "OOM"
            finally:
                del x, do, labels, nsp_torch
                torch.cuda.empty_cache()

            try:
                labels = torch.randint(0, num_classes, (batch,))
                x = torch.randn(
                    (batch, L - 1, 32),
                    device=dist.get_device(),
                    requires_grad=True,
                    dtype=torch.float16,
                )
                do = torch.rand(
                    (batch, L, 4096), device=dist.get_device(), dtype=torch.float16
                )
                labels = labels.to(dist.get_device())

                nsp_triton = build_nsp(args, using_block_sparse_attn=True)
                nsp_triton = nsp_triton.to(dist.get_device())
                nsp_triton = args.compile_model(nsp_triton, args.tfast)

                def triton_attn(labels, x, do):
                    with torch.amp.autocast("cuda", enabled=True):
                        nsp_triton(labels, x).backward(do, retain_graph=True)

                fn = lambda: triton_attn(labels, x, do)
                triton_ms = triton.testing.do_bench(fn)

            except torch.cuda.OutOfMemoryError as e:
                triton_ms = "OOM"
            finally:
                del x, do, labels, nsp_triton
                torch.cuda.empty_cache()

            speedup = torch_ms / triton_ms if triton_ms != "OOM" and torch_ms != "OOM" else "N/A"

            results.add_row([f"{reso}", batch, depth, torch_ms, triton_ms, speedup])
        results.add_divider()
       
results.title = "End-to-End Benchmark Results"
results.float_format = "0.4"
print(results.get_string())
