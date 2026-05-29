# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import itertools

import prettytable as pt
import torch
from triton.testing import do_bench

from models import HMAR
from utils.benchmark import benchmark_memory_usage
from utils.sampling_arg_util import Args, get_args

device = "cuda"

reso_mask_schedules = {
    256: [[1], [1, 3], [1, 3, 5], [16], [25], [36], [64], [100], [169], [256]],
    512: [[1], [1, 3], [1, 1, 2, 5], [5, 11], [36], [81], [169], [324], [576], [1024]],
    1024: [[1], [1, 3], [1, 1, 2, 5], [3, 5, 8], [25], [49], [81], [144], [256], [441], [729], [1296], [2304], [4096]]
}

reso_pns = {
    256: (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    512: (1, 2, 3, 4, 6, 9, 13, 18, 24, 32),
    1024: (1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64),
}

def build_everything(args: Args):
    from models import VQVAE, build_vae_hmar

    vae_local, hmar = build_vae_hmar(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=args.patch_nums,
        num_classes=1000,
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
    )

    # make sure that the vqvae is actually initialized
    # without this we weird nan errors
    for _, param in vae_local.named_parameters():
        param.data = torch.rand_like(param.data)

    # make sure that the hmar is actually initialized
    # without this we get weird nan errors
    for _, param in hmar.named_parameters():
        param.data = torch.rand_like(param.data)

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    hmar: HMAR = args.compile_model(hmar, args.tfast)

    return hmar


def _benchmark_generation(benchmark_fn, args, bsz, mask, mask_schedule, kv_cache):
    try:
        hmar = build_everything(args)
        t_or_m = benchmark_fn(
            lambda: hmar.generate(
                bsz,
                None,
                cfg=1.5,
                top_p=0.96,
                top_k=900,
                more_smooth=False,
                mask=mask,
                mask_schedule=mask_schedule,
                kv_cache=kv_cache,
            )
        )
        del hmar
    except torch.cuda.OutOfMemoryError as e:
        t_or_m = "OOM"
    finally:
        torch.cuda.empty_cache()
        
    return t_or_m


def benchmark_generation(benchmark_fn, resolution, args, bszs, depths, results):

    for depth, bsz in itertools.product(depths, bszs):

        with torch.inference_mode():
            args.depth = depth

            x_var = _benchmark_generation(
                benchmark_fn,
                args,
                bsz,
                mask=False,
                mask_schedule=None,
                kv_cache=True
            )
            
            x_hmar = _benchmark_generation(
                benchmark_fn,
                args,
                bsz,
                mask=True,
                mask_schedule=args.mask_schedule,
                kv_cache=False,
            )
            
            gain = x_var/x_hmar if x_var != 'OOM' and x_hmar != 'OOM' else 'N/A'
                
            results.add_row([resolution, depth, bsz, x_var, x_hmar, gain])

    return results


def create_results_table(unit, gain_label, float_format="0.2"):
    results = pt.PrettyTable(float_format=float_format)
    results.field_names = [
        "Resolution",
        "Depth",
        "Batch size",
        f"VAR ({unit})",
        f"HMAR ({unit})",
        gain_label
    ]
    return results
    
if __name__ == "__main__":
    args: Args = get_args()
    depths = [20, 24, 30, 36]
    bszs = [4, 8, 16]

    results = create_results_table("GB", "savings")
    results.title = "Memory Usage"
    
    # memory usage
    for reso in reso_pns.keys():
        args.patch_nums = reso_pns[reso]
        args.mask_schedule = reso_mask_schedules[reso]
    
        results = benchmark_generation(
            benchmark_memory_usage, reso, args, bszs, depths, results
        )
        results.add_divider()
    
    print(results.get_string())

    results = create_results_table("ms", "speedup")
    results.title = "Runtime"
    
    # runtime
    for reso in reso_pns.keys():
        args.patch_nums = reso_pns[reso]
        args.mask_schedule = reso_mask_schedules[reso]
    
        results = benchmark_generation(
            do_bench, reso, args, bszs, depths, results
        )
        results.add_divider()
    
    print(results.get_string())