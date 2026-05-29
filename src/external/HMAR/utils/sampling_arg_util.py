# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import yaml
from .arg_util import _compile_model, _get_yaml_loader, _seed_everything, _set_tf32
from tap import Tap


class Args(Tap):
    checkpoint = "hmar-d16"
    # VAE
    vfast: int = 0  # torch.compile VAE; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'

    # HMAR
    tfast: int = 0  # torch.compile HMAR; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    depth: int = 16  # HMAR depth

    # other hps
    saln: bool = False  # whether to use shared adaln
    anorm: bool = True  # whether to use L2 normalized attention
    fuse: bool = True  # whether to use fused op like flash attn, xformers, fused MLP, fused LayerNorm, etc.
    
    # data
    pn: str = "1_2_3_4_5_6_8_10_13_16"
    patch_size: int = 16
    patch_nums: tuple = None  # [automatically set; don't specify this] = tuple(map(int, args.pn.replace('-', '_').split('_')))
    resos: tuple = None  # [automatically set; don't specify this] = tuple(pn * args.patch_size for pn in args.patch_nums)

    tf32: bool = True  # whether to use TensorFloat32
    seed: int = 42  # seed

    cfg: float = 1.5
    top_k: int = 900
    top_p: float = 0.96
    more_smooth: bool = False
    mask: bool = True
    mask_schedule = [[1], [1, 3], [1, 1, 2, 5], [16], [25], [36], [64], [100], [169], [256]]

    def seed_everything(self, benchmark: bool):
        _seed_everything(self.seed, benchmark)

    def compile_model(self, m, fast):
        return _compile_model(m, fast)

    @staticmethod
    def set_tf32(tf32: bool):
        _set_tf32(tf32)


def get_args(cfg_folder: str = None) -> Args:
    args = Args(explicit_bool=True).parse_args(known_only=True)
    loader = _get_yaml_loader()

    if cfg_folder != None:
        try:
            with open(f"config/{cfg_folder}/{args.checkpoint}.yaml", "r") as file:
                config = yaml.load(file, Loader=loader)
                for key, value in config.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
        except FileNotFoundError:
            exit(f'{"*"*40}  please specify a valid checkpoint {"*"*40}')

    args.patch_nums = tuple(map(int, args.pn.replace('-', '_').split('_')))
    
    # set env
    args.set_tf32(args.tf32)
    args.seed_everything(benchmark=True)

    return args