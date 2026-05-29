# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.nn.parallel import DistributedDataParallel as DDP

from models import VQVAE, Transformer, VectorQuantizer2
from utils.amp_sc import AmpOptimizer

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class Trainer(object):
    def __init__(
        self,
        device,
        patch_nums: Tuple[int, ...],
        resos: Tuple[int, ...],
        vae_local: VQVAE,
        transformer_wo_ddp: Transformer,
        transformer: DDP,
        optimizer: AmpOptimizer,
        label_smooth: float,
        reweight_loss: bool = False,
        loss_reweight_type: str = "equal",
    ):
        super(Trainer, self).__init__()
        self.transformer, self.vae_local, self.quantize_local = (
            transformer,
            vae_local,
            vae_local.quantize,
        )
        self.quantize_local: VectorQuantizer2
        self.transformer_wo_ddp: Transformer = transformer_wo_ddp  # after torch.compile
        self.optimizer = optimizer

        del self.transformer_wo_ddp.rng
        self.transformer_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smooth, reduction="none"
        )
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction="mean")
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.reweight_loss = reweight_loss
        if self.reweight_loss:
            loss_weights = self.get_loss_weight(loss_reweight_type, patch_nums)
            self.loss_weight = torch.tensor(loss_weights, device=device).view(1, -1)
        else:
            # this corresponds to the square loss which is the default
            self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for _, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

    def norm_dist_equivalent(self, patch_nums):
        k = patch_nums[-1]
        ratios = np.array(patch_nums) / k

        # Calculate mean and standard deviation of the ratios
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios, ddof=1)  # ddof=1 for sample standard deviation

        patch_nums_reversed = patch_nums[::-1]
        # Calculate the z-score
        z_score = (np.array(patch_nums_reversed) / k - mean_ratio) / std_ratio

        # Calculate the probability density
        result = stats.norm.pdf(z_score, loc=0, scale=1)
        result = result / np.sum(result)
        return list(
            chain.from_iterable(
                [
                    [w / (patch_nums[i] * patch_nums[i])]
                    * (patch_nums[i] * patch_nums[i])
                    for i, w in enumerate(list(result))
                ]
            )
        )

    # how to weight the contribution of different scales to the training loss
    def get_loss_weight(self, loss_reweight_type: str, patch_nums: List[int]):
        if self.reweight_loss:
            loss_weights = []
            if loss_reweight_type == "equal":
                for _, pn in enumerate(patch_nums):
                    loss_weights += [1 / (pn * pn)] * (pn * pn)
            elif loss_reweight_type == "lognorm":
                val_returned = self.norm_dist_equivalent(patch_nums)
                loss_weights += val_returned
            elif loss_reweight_type == "mask_unweighted":
                loss_weights = [1] * self.L
            else:
                raise ValueError(f"Unknown loss_reweight_type: {loss_reweight_type}")
        return loss_weights

    def get_config(self):
        return {
            "patch_nums": self.patch_nums,
            "resos": self.resos,
            "label_smooth": self.label_smooth,
            "reweight_loss": self.reweight_loss
        }

    def state_dict(self):
        state = {"config": self.get_config()}
        for k in ("transformer_wo_ddp", "vae_local", "optimizer"):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, "_orig_mod"):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
