# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import os
import shutil
import torch

from models import HMAR
from utils import misc
from utils.sampling_arg_util import Args, get_args
from utils.evaluation import generate_50k_samples


def build_everything(args: Args):
    from models import VQVAE, build_vae_hmar

    vae_local, hmar = build_vae_hmar(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        device="cuda",
        patch_nums=args.patch_nums,
        num_classes=1000,
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
    )

    vae_ckpt = os.path.join(os.getcwd(), "vae_ch160v4096z32.pth")
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location="cpu", weights_only=True), strict=True)

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    hmar: HMAR = args.compile_model(hmar, args.tfast)

    return hmar


if __name__ == "__main__":
    args: Args = get_args(cfg_folder='evaluate')
    hmar = build_everything(args)
    torch.set_default_device(f"cuda")

    hmar.load_state_dict(torch.load("hmar-d16.pth", map_location="cpu", weights_only=True))
    hmar.eval()

    sample_folder = os.path.join(os.getcwd(), f"samples-{args.checkpoint}")
    
    generate_50k_samples(
        hmar, sample_folder=sample_folder, num_samples=1, top_p=args.top_p, top_k=args.top_k, cfg= args.cfg, mask=args.mask, mask_schedule=args.mask_schedule
    )
    
    sample_npz = os.path.join(os.getcwd(), f"{sample_folder}.npz")

    print("Converting samples to npz...")

    if os.path.exists(sample_npz):
        os.remove(sample_npz)
    misc.create_npz_from_sample_folder(f"{sample_folder}")

    shutil.rmtree(sample_folder)
