# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import os
import torch
from utils.sampling_arg_util import Args, get_args
from models import HMAR
import torch

device = 'cuda'

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

    vae_ckpt = os.path.join(".", "vae_ch160v4096z32.pth")
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location="cpu", weights_only=True), strict=True)

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    hmar: HMAR = args.compile_model(hmar, args.tfast)

    return hmar


if __name__ == "__main__":
    args: Args = get_args(cfg_folder='sample')
    hmar = build_everything(args)
    torch.set_default_device(device)

    hmar.eval()
    hmar.load_state_dict(torch.load(f"{args.checkpoint}.pth", map_location="cpu", weights_only=True))
    
    class_id = 3
    b = 8
    seed = 13
    
    with torch.inference_mode():
        imgs = hmar.generate(
            b,
            class_id,
            g_seed=seed,
            num_samples=1,
            top_k=args.top_k,
            top_p=args.top_p,
            cfg=args.cfg,
            more_smooth=args.more_smooth,
            mask=args.mask,
            mask_schedule=args.mask_schedule
        )
        from torchvision.utils import save_image

    save_image(imgs, "sample_hmar.png", nrow=4)
