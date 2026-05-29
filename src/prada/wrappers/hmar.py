import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# from typing import Any, Mapping, Optional, Union, Tuple
import torch
import yaml
from tap import Tap
from torch import Tensor
from torch import distributed as tdist
from torch import nn as nn
from torch.nn import functional as F
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from external.HMAR import dist
from external.HMAR.models import HMAR
from external.HMAR.models.helpers import sample_with_top_k_top_p_
from external.HMAR.utils.arg_util import _compile_model, _get_yaml_loader, _seed_everything, _set_tf32
from external.HMAR.utils.data import normalize_01_into_pm1
from external.HMAR.utils.sampling_arg_util import Args  # , get_args

from .base import Wrapper

# copied from original code, only modification to also support Jupyter notebooks

# Copyright (c) 2025, NVIDIA Corporation. All rights reserved. [?]
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE


def get_args(checkpoint, cfg_folder: str = None) -> Args:
    print("Parsing args...")
    # In a Jupyter notebook, sys.argv usually contains 'ipykernel_launcher.py'
    if any("ipykernel_launcher" in a for a in sys.argv):
        args = Args(explicit_bool=True).parse_args([])
    else:
        args = Args(explicit_bool=True).parse_args(known_only=True)
    # ensure checkpoint is set
    args.checkpoint = checkpoint
    loader = _get_yaml_loader()
    # print(args)

    if cfg_folder != None:
        config_path = Path(f"src/external/HMAR/config/{cfg_folder}/{checkpoint}.yaml")
        print(f"Opening {config_path}")
        try:
            with open(config_path, "r") as file:
                config = yaml.load(file, Loader=loader)
                for key, value in config.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
        except FileNotFoundError:
            sys.exit(f"{'*' * 40}  please specify a valid checkpoint {'*' * 40}")

    args.patch_nums = tuple(map(int, args.pn.replace("-", "_").split("_")))

    # set env
    args.set_tf32(args.tf32)
    args.seed_everything(benchmark=True)

    return args


def build_everything(args: Args, device: str = "cuda", checkpoints_root: str = "checkpoints"):
    from external.HMAR.models import VQVAE, build_vae_hmar

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
    # utilize the checkpoint from the VAR repo
    vae_ckpt = os.path.join(f"{checkpoints_root}/var/vae_ch160v4096z32.pth")
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location="cpu", weights_only=True), strict=True)

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    hmar: HMAR = args.compile_model(hmar, args.tfast)

    return hmar


def download_with_progress(repo_id, filename, local_dir):
    """
    Download a file from Hugging Face with a simple progress bar.
    """
    import requests

    os.makedirs(local_dir, exist_ok=True)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        print(f"Already present: {local_path}")
        return local_path

    # Streaming download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB

    print(f"Downloading {filename} from {url}...")
    with (
        open(local_path, "wb") as file,
        tqdm(total=total_size, desc=filename, unit_scale=True, unit_divisor=1024, unit="B") as bar,
    ):
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

    print(f"Downloaded: {local_path}")
    return local_path


def ensure_hmar_checkpoints(checkpoints, path="checkpoints/hmar"):
    """
    Ensure that NVIDIA HMAR .pth checkpoints are downloaded locally.
    Downloads missing ones from https://huggingface.co/nvidia/HMAR
    """
    # from huggingface_hub import hf_hub_download # does not support progress bar

    os.makedirs(path, exist_ok=True)

    if isinstance(checkpoints, str):
        checkpoints = [checkpoints]

    for ckpt in checkpoints:
        download_with_progress("nvidia/HMAR", ckpt, path)

    print("All requested HMAR checkpoints downloaded.")


class HMARWrapper(Wrapper):
    """Kumbong et al. (2025), HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation"""

    range_after_transform = (-1, 1)

    def __init__(self, model_depth: str = "d16", checkpoints_root: str | Path = "checkpoints", **kwargs):
        args: Args = get_args(checkpoint=f"hmar-{model_depth}", cfg_folder="sample")
        # print(args)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using HMAR model with depth: {model_depth}, checkpoint: {args.checkpoint}")
        hmar = build_everything(args, device=self.device, checkpoints_root=checkpoints_root)
        torch.set_default_device(self.device)

        # ensure that checkpoints are present
        checkpoint_path = Path(checkpoints_root) / "hmar"
        ensure_hmar_checkpoints([f"{args.checkpoint}.pth"], path=str(checkpoint_path))

        # load weights
        hmar.eval()
        print(f"Loading HMAR from {checkpoint_path}/{args.checkpoint}.pth...", end=" ")
        hmar.load_state_dict(
            torch.load(f"{checkpoint_path}/{args.checkpoint}.pth", map_location="cpu", weights_only=True)
        )
        print(f"Successfully loaded!")

        self.hmar = hmar
        self.args = args

        # add scale_length
        # self.patch_nums = self.hmar.patch_nums
        self.scale_lengths = [l**2 for l in self.hmar.patch_nums]

    @property
    def transform(self):
        # default transform for 256x256 images
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        )

    def generate_image(self, condition_B, seed: int = 42) -> dict[str, Tensor]:
        """Generate an image from imagenet class."""

        b = len(condition_B)

        # ensure condition_B is a torch tensor of long dtype
        if not isinstance(condition_B, torch.Tensor):
            condition_B = torch.tensor(condition_B)
        if condition_B.dtype != torch.long:
            condition_B = condition_B.long()

        # and put to device...
        condition_B = condition_B.to(self.device)

        with torch.inference_mode():
            imgs = self.hmar.generate(
                b,
                condition_B,
                g_seed=seed,
                num_samples=1,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                cfg=self.args.cfg,
                more_smooth=self.args.more_smooth,
                mask=self.args.mask,
                mask_schedule=self.args.mask_schedule,
            )
        # from torchvision.utils import save_image
        # save_image(imgs, "sample_hmar.png", nrow=4)

        return dict(image_B3HW=imgs)

    def get_gt_idx(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        """Compute the token representation for an image.

        Return gt_idx_BL.
        """
        gt_idx = self.hmar.vae_proxy[0].img_to_idxBl(image_B3HW)

        return dict(gt_idx_BL=torch.cat(gt_idx, dim=1))

    def get_logits(self, gt_idx_BL: Tensor, condition_B: Tensor, return_image: bool = False) -> dict[str, Tensor]:
        """Compute the model's conditional and unconditional logits from tokens and condition (label or prompt).

        Return cond_logits_BLX and uncond_logits_BLX. If `return_image` is true, return decoded image for debugging.

        Derived from src/external/HMAR/models/hmar.py
        """

        gt_idx_SBL = torch.split(gt_idx_BL, split_size_or_sections=self.scale_lengths, dim=-1)

        # variables to store logits
        cond_logits_BLV, uncond_logits_BLV = [], []

        @torch.no_grad()
        def generate(
            self,
            B: int,
            label_B: Optional[Union[int, torch.LongTensor]],
            g_seed: Optional[int] = None,
            cfg=1.5,
            top_k=1100,
            top_p=0.999,
            more_smooth=False,
            num_samples=1,
            mask=True,
            mask_schedule=None,
            kv_cache=False,  # Only used to benchmark and compare performance to VAR
        ) -> torch.Tensor:
            # TODO: Support sampling with gumbel_softmax like in MaskGIT and VAR, when more_smooth is True.

            """
            only used for inference, on autoregressive mode
            :param B: batch size
            :param label_B: imagenet label; if None, randomly sampled
            :param g_seed: random seed
            :param cfg: classifier-free guidance ratio
            :param top_k: top-k sampling
            :param top_p: top-p sampling
            :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
            :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
            """
            if g_seed is None:
                rng = None
            else:
                self.rng.manual_seed(g_seed)
                rng = self.rng

            if label_B is None:
                label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(
                    B
                )
            elif isinstance(label_B, int):
                label_B = torch.full(
                    (B,),
                    fill_value=self.num_classes if label_B < 0 else label_B,
                    device=self.lvl_1L.device,
                )

            sos = cond_BD = self.class_emb(
                torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0)
            )
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
            next_token_map = (
                sos.unsqueeze(1).expand(2 * B, self.first_l, -1)
                + self.pos_start.expand(2 * B, self.first_l, -1)
                + lvl_pos[:, : self.first_l].expand(2 * B, self.first_l, -1)
            )

            cur_L = 0
            f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

            ntokens_per_steps = mask_schedule

            # This is only used for benchmarking to compare the performance to VAR
            if kv_cache:
                for b in self.base_blocks:
                    b.attn.kv_caching(True)
                for b in self.ns_blocks:
                    b.attn.kv_caching(True)

            for si, pn in enumerate(self.patch_nums):  # si: i-th segment
                ratio = si / self.num_stages_minus_1
                x = next_token_map

                for b in self.base_blocks:
                    x = b(
                        x=x,
                        cond_BD=cond_BD_or_gss,
                        using_block_sparse_attn=False,
                        attn_bias=None,
                    )

                for b in self.ns_blocks:
                    x = b(
                        x=x,
                        cond_BD=cond_BD_or_gss,
                        using_block_sparse_attn=False,
                        attn_bias=None,
                    )

                logits_BlV = self.get_ns_logits(x, cond_BD)

                # --------------------------------------------------------- #
                # store logits
                cond_logits, uncond_logits = logits_BlV[:B], logits_BlV[B:]
                cond_logits_BLV.append(cond_logits)
                uncond_logits_BLV.append(uncond_logits)
                # --------------------------------------------------------- #

                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=num_samples
                )
                idx_Bl = idx_Bl[:, :, 0]

                # --------------------------------------------------------- #
                # lace sampled idx with ground truth
                idx_Bl = gt_idx_SBL[si]
                # --------------------------------------------------------- #

                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)

                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

                if mask and pn * pn > 1 and si < len(self.patch_nums):
                    n_steps = len(ntokens_per_steps[si])
                    n_tokens_mask = sum(ntokens_per_steps[si][1:])
                    probs = torch.nn.functional.softmax(logits_BlV, dim=-1)
                    probs_sampled = torch.gather(probs, 2, idx_Bl.unsqueeze(-1)).squeeze(-1)
                    idx_to_mask = torch.argsort(probs_sampled, dim=-1)[:, :n_tokens_mask]

                    for step in range(1, n_steps):
                        ratio_step = 1e-6  # TODO: Remove this from being hardcoded
                        f_hat_mask, next_token_map_mask = self.vae_quant_proxy[0].get_next_mask_input(
                            si, len(self.patch_nums), f_hat, h_BChw
                        )
                        next_token_map_mask = next_token_map_mask.view(B, self.Cvae, -1).transpose(1, 2)
                        f_hat_mask = f_hat_mask.view(B, self.Cvae, -1).transpose(1, 2)
                        f_hat_mask = self.word_embed(f_hat_mask)
                        next_token_map_mask = self.word_embed(next_token_map_mask)
                        next_token_map_mask = torch.scatter(
                            next_token_map_mask,
                            1,
                            idx_to_mask.unsqueeze(-1).expand(-1, -1, self.C),
                            self.mask_embed(torch.tensor(0, device=dist.get_device(), dtype=torch.int)).expand(
                                B, pn * pn, -1
                            ),
                        )
                        next_token_map_mask = (
                            f_hat_mask
                            + next_token_map_mask
                            + self.word_embed_bias
                            + lvl_pos[:, cur_L : cur_L + self.patch_nums[si] ** 2]
                        )
                        next_token_map_mask = next_token_map_mask.repeat(2, 1, 1)

                        x = next_token_map_mask

                        for b in self.base_blocks:
                            x = b(
                                x=x,
                                cond_BD=cond_BD_or_gss,
                                using_block_sparse_attn=False,
                                attn_bias=None,
                            )
                        for b in self.mask_blocks:
                            x = b(
                                x=x,
                                cond_BD=cond_BD_or_gss,
                                using_block_sparse_attn=False,
                                attn_bias=None,
                            )
                        logits_BlV_mask = self.get_mask_logits(x, cond_BD)

                        t = cfg * ratio_step
                        logits_BlV_mask = (1 + t) * logits_BlV_mask[:B] - t * logits_BlV_mask[B:]

                        idx_Bl_mask = sample_with_top_k_top_p_(
                            logits_BlV_mask,
                            rng=rng,
                            top_k=top_k,
                            top_p=top_p,
                            num_samples=num_samples,
                        )
                        idx_Bl_mask = idx_Bl_mask[:, :, 0]

                        idx_Bl[torch.arange(B).unsqueeze(1), idx_to_mask] = idx_Bl_mask[
                            torch.arange(B).unsqueeze(1), idx_to_mask
                        ]
                        h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)

                        h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

                        if step != n_steps - 1:
                            n_tokens_mask = sum(ntokens_per_steps[si][step + 1 :])
                            probs = torch.softmax(logits_BlV_mask, dim=-1)
                            probs_sampled = torch.gather(probs, 2, idx_Bl.unsqueeze(-1)).squeeze(-1)
                            probs_sampled_masked = probs_sampled[torch.arange(B).unsqueeze(1), idx_to_mask]
                            idx_sampled_sorted = torch.argsort(probs_sampled_masked, dim=-1)[:, :n_tokens_mask]
                            idx_to_mask = idx_to_mask[torch.arange(B).unsqueeze(1), idx_sampled_sorted]

                f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, len(self.patch_nums), f_hat, h_BChw
                )

                cur_L += pn * pn

                if si != self.num_stages_minus_1:  # prepare for next stage
                    next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                    next_token_map = (
                        self.word_embed(next_token_map)
                        + self.word_embed_bias
                        + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
                    )
                    next_token_map = next_token_map.repeat(2, 1, 1)  # double the batch sizes due to CFG

            # This is only used for benchmarking to compare the performance to VAR
            if kv_cache:
                for b in self.base_blocks:
                    b.attn.kv_caching(False)
                for b in self.ns_blocks:
                    b.attn.kv_caching(False)

            return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)  # de-normalize, from [-1, 1] to [0, 1]

        condition_B = condition_B.to(self.device)
        b = condition_B.shape[0]

        img = generate(
            self=self.hmar,
            B=b,
            label_B=condition_B,
            cfg=self.args.cfg,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            more_smooth=self.args.more_smooth,
            num_samples=1,
            mask=self.args.mask,
            mask_schedule=self.args.mask_schedule,
        )

        # concatenate along scales
        if return_image:
            return img
        else:
            return dict(
                cond_logits_BLX=torch.cat(cond_logits_BLV, dim=1), uncond_logits_BLX=torch.cat(uncond_logits_BLV, dim=1)
            )

    # def get_ae_rec_and_quant_error(self, image_B3HW: Tensor) -> dict[str, Tensor]:
    #     """Compute the AE reconstruction (D(E(x))) and quantization error (MSE).

    #     Return rec_B3HW and quant_err_BL.
    #     """
    #     # with torch.inference_mode():
    #     #     img_z = self.hmar.vae_proxy[0].img_to_idxBl(image_B3HW)
    #     #     img_rec = self.hmar.vae_proxy[0].idxBl_to_img(img_z, same_shape=True, last_one=True)

    #     # rec_B3HW = img_rec.clamp(0, 1)
    #     # return dict(rec_B3HW=rec_B3HW, quant_err_BL=quant_err_BL)

    def get_ae_rec_and_quant_error(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        # SAME as VARWrapper.get_ae_rec_and_quant_error, but adapted to HMAR
        # so if we change something for VAR, we should also change here!

        def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
            dtype = f_BChw.dtype
            if dtype != torch.float32:
                f_BChw = f_BChw.float()
            B, C, H, W = f_BChw.shape
            f_no_grad = f_BChw.detach()

            f_rest = f_no_grad.clone()
            f_hat = torch.zeros_like(f_rest)

            with torch.cuda.amp.autocast(enabled=False):
                mean_vq_loss: list[torch.Tensor] = []
                vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
                SN = len(self.v_patch_nums)
                for si, pn in enumerate(self.v_patch_nums):  # from small to large
                    # find the nearest embedding
                    if self.using_znorm:
                        rest_NC = (
                            F.interpolate(f_rest, size=(pn, pn), mode="area").permute(0, 2, 3, 1).reshape(-1, C)
                            if (si != SN - 1)
                            else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                        )
                        rest_NC = F.normalize(rest_NC, dim=-1)
                        idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                    else:
                        rest_NC = (
                            F.interpolate(f_rest, size=(pn, pn), mode="area").permute(0, 2, 3, 1).reshape(-1, C)
                            if (si != SN - 1)
                            else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                        )
                        d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(
                            self.embedding.weight.data.square(), dim=1, keepdim=False
                        )
                        d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                        idx_N = torch.argmin(d_no_grad, dim=1)

                    hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                    if self.training:
                        if dist.initialized():
                            handler = tdist.all_reduce(hit_V, async_op=True)

                    # calc loss
                    idx_Bhw = idx_N.view(B, pn, pn)
                    h_BChw = (
                        F.interpolate(
                            self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode="bicubic"
                        ).contiguous()
                        if (si != SN - 1)
                        else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                    )
                    h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                    f_hat = f_hat + h_BChw
                    f_rest -= h_BChw

                    if self.training and dist.initialized():
                        handler.wait()
                        if self.record_hit == 0:
                            self.ema_vocab_hit_SV[si].copy_(hit_V)
                        elif self.record_hit < 100:
                            self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                        else:
                            self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                        self.record_hit += 1
                    vocab_hit_V.add_(hit_V)
                    mean_vq_loss.append(
                        F.mse_loss(f_hat.data, f_BChw, reduction="none").mean(dim=1).flatten(start_dim=1)
                    )  # .mul_(self.beta) + F.mse_loss(f_hat, f_no_grad, reduction="none")

                # mean_vq_loss *= 1.0 / SN
                f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

            # margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
            # # margin = pn*pn / 100
            # if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
            # else: usages = None
            usages = None
            return f_hat, usages, torch.cat(mean_vq_loss, dim=1)

        # new way with forward
        f_BChw = self.hmar.vae_proxy[0].quant_conv(self.hmar.vae_proxy[0].encoder(image_B3HW.to(self.device)))
        f_hat, usages, mean_vq_loss = forward(self=self.hmar.vae_proxy[0].quantize, f_BChw=f_BChw)
        rec_B3HW = self.hmar.vae_proxy[0].decoder(self.hmar.vae_proxy[0].post_quant_conv(f_hat)).clamp_(-1, 1)

        # compute the reconstruction error
        # rec_error_B = torch.mean((rec_B3HW - image_B3HW.to(self.device)) ** 2, dim=(1, 2, 3))

        return dict(
            rec_B3HW=rec_B3HW,
            quant_err_BL=mean_vq_loss,
        )
