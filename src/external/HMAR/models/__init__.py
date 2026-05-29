# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

from typing import Tuple
import torch.nn as nn

from .nsp import NextScalePrediction
from .mp import MaskedPrediction
from .hmar import HMAR
from .transformer import Transformer
from .vqvae import VQVAE, VectorQuantizer2


def build_vae_nsp(
    device,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
    V=4096,
    Cvae=32,
    ch=160,
    share_quant_resi=4,
    num_classes=1000,
    depth=16,
    shared_aln=False,
    attn_l2_norm=True,
    flash_if_available=True,
    fused_if_available=True,
    init_adaln=0.5,
    init_adaln_gamma=1e-5,
    init_head=0.02,
    init_std=-1,  # init_std < 0: automated
    test_mode=True,
    using_block_sparse_attn=True,
) -> Tuple[VQVAE, NextScalePrediction]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    # disable built-in initialization for speed
    for clz in (
        nn.Linear,
        nn.LayerNorm,
        nn.BatchNorm2d,
        nn.SyncBatchNorm,
        nn.Conv1d,
        nn.Conv2d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
    ):
        setattr(clz, "reset_parameters", lambda self: None)

    # build models
    vae_local = VQVAE(
        vocab_size=V,
        z_channels=Cvae,
        ch=ch,
        test_mode=test_mode,
        share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums,
    ).to(device)

    nsp_wo_ddp = NextScalePrediction(
        vae_local=vae_local,
        num_classes=num_classes,
        depth=depth,
        embed_dim=width,
        num_heads=heads,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=dpr,
        norm_eps=1e-6,
        shared_aln=shared_aln,
        cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available,
        fused_if_available=fused_if_available,
        using_block_sparse_attn=using_block_sparse_attn,
    ).to(device)
    nsp_wo_ddp.init_weights(
        init_adaln=init_adaln,
        init_adaln_gamma=init_adaln_gamma,
        init_head=init_head,
        init_std=init_std,
    )

    return vae_local, nsp_wo_ddp


def build_vae_mp(
    device,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
    V=4096,
    Cvae=32,
    ch=160,
    share_quant_resi=4,
    num_classes=1000,
    depth=16,
    shared_aln=False,
    attn_l2_norm=True,
    flash_if_available=True,
    fused_if_available=True,
    init_adaln=0.5,
    init_adaln_gamma=1e-5,
    init_head=0.02,
    init_std=-1,  # init_std < 0: automated
    test_mode=True,
    n_layers_train=2,
    using_block_sparse_attn=True,
) -> Tuple[VQVAE, MaskedPrediction]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    # disable built-in initialization for speed
    for clz in (
        nn.Linear,
        nn.LayerNorm,
        nn.BatchNorm2d,
        nn.SyncBatchNorm,
        nn.Conv1d,
        nn.Conv2d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
    ):
        setattr(clz, "reset_parameters", lambda self: None)

    # build models
    vae_local = VQVAE(
        vocab_size=V,
        z_channels=Cvae,
        ch=ch,
        test_mode=test_mode,
        share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums,
    ).to(device)

    mp_wo_ddp = MaskedPrediction(
        vae_local=vae_local,
        num_classes=num_classes,
        depth=depth,
        embed_dim=width,
        num_heads=heads,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=dpr,
        norm_eps=1e-6,
        shared_aln=shared_aln,
        cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available,
        fused_if_available=fused_if_available,
        n_layers_train=n_layers_train,
        using_block_sparse_attn=using_block_sparse_attn,
    ).to(device)
    mp_wo_ddp.init_weights(
        init_adaln=init_adaln,
        init_adaln_gamma=init_adaln_gamma,
        init_head=init_head,
        init_std=init_std,
    )

    return vae_local, mp_wo_ddp


def build_vae_hmar(
    device,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
    V=4096,
    Cvae=32,
    ch=160,
    share_quant_resi=4,
    num_classes=1000,
    depth=16,
    shared_aln=False,
    attn_l2_norm=True,
    flash_if_available=True,
    fused_if_available=True,
    test_mode=True,
    n_layers_train=8,
    using_block_sparse_attn=False,
) -> Tuple[VQVAE, MaskedPrediction]:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    # disable built-in initialization for speed
    for clz in (
        nn.Linear,
        nn.LayerNorm,
        nn.BatchNorm2d,
        nn.SyncBatchNorm,
        nn.Conv1d,
        nn.Conv2d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
    ):
        setattr(clz, "reset_parameters", lambda self: None)

    # build models
    vae_local = VQVAE(
        vocab_size=V,
        z_channels=Cvae,
        ch=ch,
        test_mode=test_mode,
        share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums,
    ).to(device)

    hmar_wo_ddp = HMAR(
        vae_local=vae_local,
        num_classes=num_classes,
        depth=depth,
        embed_dim=width,
        num_heads=heads,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=dpr,
        norm_eps=1e-6,
        shared_aln=shared_aln,
        cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available,
        fused_if_available=fused_if_available,
        n_layers_train=n_layers_train,
    ).to(device)

    return vae_local, hmar_wo_ddp
