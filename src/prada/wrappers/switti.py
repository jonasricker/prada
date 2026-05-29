import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image as PILImage
from torch import Tensor
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm

from external.switti.models.clip import FrozenCLIPEmbedder
from external.switti.models.pipeline import TRAIN_IMAGE_SIZE, SwittiPipeline
from external.switti.models.switti import SwittiHF, get_crop_condition
from external.switti.models.vqvae import VQVAEHF
from external.switti.utils.data import normalize_01_into_pm1

from .base import Wrapper


class SwittiWrapper(Wrapper):
    """Voronov et al. 2025, SWITTI: Designing Scale-Wise Transformers for Text-to-Image Synthesis"""

    range_after_transform = (-1, 1)

    def __init__(self, variant: str = "1024", checkpoints_root: str | Path = "checkpoints") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_dir = Path(checkpoints_root) / "switti"

        self.reso = 1024 if "1024" in variant else 512

        # local_files_only needed for SLURM, only works if already downloaded
        switti_path = snapshot_download(f"yresearch/Switti-{variant}", local_dir=checkpoint_dir / "switti")
        vae_path = snapshot_download(SwittiPipeline.vae_path, local_dir=checkpoint_dir / "vae")
        text_encoder_path = snapshot_download(
            SwittiPipeline.text_encoder_path, local_dir=checkpoint_dir / "text_encoder"
        )
        text_encoder_2_path = snapshot_download(
            SwittiPipeline.text_encoder_2_path, local_dir=checkpoint_dir / "text_encoder_2"
        )

        switti = SwittiHF.from_pretrained(switti_path).to(self.device)
        vae = VQVAEHF.from_pretrained(vae_path).to(self.device)
        text_encoder = FrozenCLIPEmbedder(text_encoder_path, device=self.device)
        text_encoder_2 = FrozenCLIPEmbedder(text_encoder_2_path, device=self.device)

        self.pipe = SwittiPipeline(switti, vae, text_encoder, text_encoder_2, device=self.device, dtype=torch.bfloat16)

        self.scale_lengths = [l**2 for l in self.pipe.switti.patch_nums]

    @property
    def transform(self) -> Callable:

        train_aug = [
            transforms.Resize(
                self.reso,
                interpolation=InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop((self.reso, self.reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ]
        return transforms.Compose(train_aug)

    def generate_image(self, condition_B: torch.Tensor | list[str], seed: int) -> dict[str, Tensor]:
        """Based on inference_example.ipynb."""
        images = self.pipe(
            condition_B,
            cfg=6.0,
            top_k=400,
            top_p=0.95,
            more_smooth=True,
            return_pil=False,
            smooth_start_si=2,
            turn_on_cfg_start_si=2,
            turn_off_cfg_start_si=11 if self.reso == 1024 else 8,
            last_scale_temp=0.1,
            seed=seed,
        )
        return dict(image_B3HW=images.float())

    @torch.inference_mode()
    def get_gt_idx(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        """Get ground-truth token indices from the input image."""
        gt_idx_BL = torch.cat(self.pipe.vae.img_to_idxBl(image_B3HW.to(self.device, dtype=torch.bfloat16)), dim=1)
        return dict(gt_idx_BL=gt_idx_BL)

    def get_logits(
        self, gt_idx_BL: torch.Tensor, condition_B: torch.Tensor, return_image: bool = False
    ) -> dict[str, Tensor]:
        """Derived from src/external/switti/models/pipeline.py"""

        gt_idx_SBL = torch.split(gt_idx_BL, split_size_or_sections=self.scale_lengths, dim=-1)

        # variables to store logits
        cond_logits_BLV, uncond_logits_BLV = [], []

        @torch.inference_mode()
        def __call__(
            self,
            prompt: str | list[str],
            null_prompt: str = "",
            seed: int | None = None,
            cfg=6.0,
            top_k=400,
            top_p=0.95,
            more_smooth=True,
            return_pil=False,
            smooth_start_si=2,
            turn_on_cfg_start_si=2,
            turn_off_cfg_start_si=999,  # never turn off CFG so we can get all data
            last_scale_temp=0.1,
        ) -> torch.Tensor | list[PILImage]:
            """
            only used for inference, on autoregressive mode
            :param prompt: text prompt to generate an image
            :param null_prompt: negative prompt for CFG
            :param seed: random seed
            :param cfg: classifier-free guidance ratio
            :param top_k: top-k sampling
            :param top_p: top-p sampling
            :param more_smooth: sampling using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
            :return: if return_pil: list of PIL Images, else: torch.tensor (B, 3, H, W) in [0, 1]
            """
            assert not self.switti.training
            switti = self.switti
            vae = self.vae
            vae_quant = self.vae.quantize
            if seed is None:
                rng = None
            else:
                switti.rng.manual_seed(seed)
                rng = switti.rng

            context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)

            B = context.shape[0] // 2

            cond_vector = switti.text_pooler(cond_vector)

            if switti.use_crop_cond:
                crop_coords = get_crop_condition(
                    2 * B * [TRAIN_IMAGE_SIZE[0]],
                    2 * B * [TRAIN_IMAGE_SIZE[1]],
                ).to(cond_vector.device)
                crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
                crop_cond = switti.crop_proj(crop_embed)
            else:
                crop_cond = None

            sos = cond_BD = cond_vector

            lvl_pos = switti.lvl_embed(switti.lvl_1L)
            if not switti.rope:
                lvl_pos += switti.pos_1LC
            next_token_map = (
                sos.unsqueeze(1) + switti.pos_start.expand(2 * B, switti.first_l, -1) + lvl_pos[:, : switti.first_l]
            )
            cur_L = 0
            f_hat = sos.new_zeros(B, switti.Cvae, switti.patch_nums[-1], switti.patch_nums[-1])

            for b in switti.blocks:
                b.attn.kv_caching(switti.use_ar)  # Use KV caching if switti is in the AR mode
                b.cross_attn.kv_caching(True)

            for si, pn in enumerate(switti.patch_nums):  # si: i-th segment
                ratio = si / switti.num_stages_minus_1
                x_BLC = next_token_map

                if switti.rope:
                    freqs_cis = switti.freqs_cis[:, cur_L : cur_L + pn * pn]
                else:
                    freqs_cis = switti.freqs_cis

                if si >= turn_off_cfg_start_si:
                    apply_smooth = False
                    x_BLC = x_BLC[:B]
                    context = context[:B]
                    context_attn_bias = context_attn_bias[:B]
                    freqs_cis = freqs_cis[:B]
                    cond_BD = cond_BD[:B]
                    if crop_cond is not None:
                        crop_cond = crop_cond[:B]
                    for b in switti.blocks:
                        if b.attn.caching and b.attn.cached_k is not None:
                            b.attn.cached_k = b.attn.cached_k[:B]
                            b.attn.cached_v = b.attn.cached_v[:B]
                        if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                            b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                            b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
                else:
                    apply_smooth = more_smooth

                for block in switti.blocks:
                    x_BLC = block(
                        x=x_BLC,
                        cond_BD=cond_BD,
                        attn_bias=None,
                        context=context,
                        context_attn_bias=context_attn_bias,
                        freqs_cis=freqs_cis,
                        crop_cond=crop_cond,
                    )
                cur_L += pn * pn

                logits_BlV = switti.get_logits(x_BLC, cond_BD)

                # store logits
                cond_logits, uncond_logits = logits_BlV[:B], logits_BlV[B:]
                cond_logits_BLV.append(cond_logits)
                uncond_logits_BLV.append(uncond_logits)

                # Guidance
                if si < turn_on_cfg_start_si:
                    logits_BlV = logits_BlV[:B]
                elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                    t = cfg * ratio
                    logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
                elif last_scale_temp is not None:
                    logits_BlV = logits_BlV / last_scale_temp

                # if apply_smooth and si >= smooth_start_si:
                #     # not used when evaluating FID/IS/Precision/Recall
                #     gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                #     idx_Bl = gumbel_softmax_with_rng(
                #         logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng,
                #     )
                #     h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
                # else:
                #     # default nucleus sampling
                #     idx_Bl = sample_with_top_k_top_p_(
                #         logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1,
                #     )[:, :, 0]
                #     h_BChw = vae_quant.embedding(idx_Bl)

                # replace sampled idx with ground truth
                idx_Bl = gt_idx_SBL[si]
                h_BChw = vae_quant.embedding(idx_Bl)

                h_BChw = h_BChw.transpose_(1, 2).reshape(B, switti.Cvae, pn, pn)
                f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                    si,
                    len(switti.patch_nums),
                    f_hat,
                    h_BChw,
                )
                if si != switti.num_stages_minus_1:  # prepare for next stage
                    next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
                    next_token_map = (
                        switti.word_embed(next_token_map) + lvl_pos[:, cur_L : cur_L + switti.patch_nums[si + 1] ** 2]
                    )
                    # double the batch sizes due to CFG
                    next_token_map = next_token_map.repeat(2, 1, 1)

            for b in switti.blocks:
                b.attn.kv_caching(False)
                b.cross_attn.kv_caching(False)

            # de-normalize, from [-1, 1] to [0, 1]
            img = vae.fhat_to_img(f_hat).add(1).mul(0.5)
            if return_pil:
                img = self.to_image(img)

            return img

        img = __call__(self=self.pipe, prompt=condition_B)

        if return_image:
            return img
        else:
            return dict(
                cond_logits_BLX=torch.cat(cond_logits_BLV, dim=1), uncond_logits_BLX=torch.cat(uncond_logits_BLV, dim=1)
            )

    def get_ae_rec_and_quant_error(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
            dtype = f_BChw.dtype
            if dtype != torch.float32:
                f_BChw = f_BChw.float()
            B, C, H, W = f_BChw.shape
            f_no_grad = f_BChw.detach()

            f_rest = f_no_grad.clone()
            f_hat = torch.zeros_like(f_rest)

            with torch.cuda.amp.autocast(enabled=False):
                mean_vq_loss: torch.Tensor = 0.0
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
                        idx_N = torch.argmax(
                            rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0),
                            dim=1,
                        )
                    else:
                        rest_NC = (
                            F.interpolate(f_rest, size=(pn, pn), mode="area").permute(0, 2, 3, 1).reshape(-1, C)
                            if (si != SN - 1)
                            else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                        )
                        d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(
                            self.embedding.weight.data.square(), dim=1, keepdim=False
                        )
                        d_no_grad.addmm_(
                            rest_NC, self.embedding.weight.data.T.float(), alpha=-2, beta=1
                        )  # (B*h*w, vocab_size)
                        idx_N = torch.argmin(d_no_grad, dim=1)

                    hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                    if self.training:
                        if dist.initialized():
                            handler = tdist.all_reduce(hit_V, async_op=True)

                    # calc loss
                    idx_Bhw = idx_N.view(B, pn, pn)
                    h_BChw = (
                        F.interpolate(
                            self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                            size=(H, W),
                            mode="bicubic",
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
                    mean_vq_loss += (
                        F.mse_loss(f_hat.data, f_BChw, reduction="none").mean(dim=1).flatten(start_dim=1)
                    )  # .mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

                mean_vq_loss *= 1.0 / SN
                f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

            # margin = (
            #     tdist.get_world_size()
            #     * (f_BChw.numel() / f_BChw.shape[1])
            #     / self.vocab_size
            #     * 0.08
            # )
            # margin = pn*pn / 100
            if ret_usages:
                usages = [
                    (self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100
                    for si, pn in enumerate(self.v_patch_nums)
                ]
            else:
                usages = None
            return f_hat, usages, mean_vq_loss

        # new way with forward
        f_BChw = self.pipe.vae.quant_conv(self.pipe.vae.encoder(image_B3HW.to(self.device, dtype=torch.bfloat16)))
        f_hat, usages, mean_vq_loss = forward(self=self.pipe.vae.quantize, f_BChw=f_BChw)
        rec_B3HW = self.pipe.vae.decoder(self.pipe.vae.post_quant_conv(f_hat.to(torch.bfloat16))).clamp_(-1, 1)

        return dict(
            rec_B3HW=rec_B3HW,
            quant_err_BL=mean_vq_loss,
        )

    def to_PIL_image(image_tensor):
        # [c, h, w] -> [h, w, c]
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.tensor(image_tensor)
        img = (image_tensor.permute(1, 2, 0) * 255).cpu().numpy()
        return PILImage.fromarray(img.astype(np.uint8))

    @torch.inference_mode()
    def prompts_to_image_synthbuster(
        self,
        prompts_csv,
        output_dir,
        n_samples_per_class=1,
        batch_size=4,
        correct_aspect_ratios=False,  # ignored, only relevant for infinity
        seed=0,
    ):
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(prompts_csv)

        prompts = df["Prompt"].tolist()
        img_names = df["image name (matching Raise-1k)"].tolist()

        if n_samples_per_class != 1:
            raise NotImplementedError("n_samples_per_class > 1 not implemented yet.")

        n_total = len(prompts)
        n_batches = math.ceil(n_total / batch_size)

        for i in tqdm(range(n_batches), desc="Generating images"):
            # slice batch
            start = i * batch_size
            end = min(start + batch_size, n_total)
            batch_prompts = prompts[start:end]
            batch_img_names = img_names[start:end]

            print(batch_prompts)

            # generate batch
            result = self.generate_image(
                condition_B=batch_prompts,
                seed=seed,
            )

            # expected: result["image_B3HW"] shape [B, 3, H, W]
            images = result["image_B3HW"].float().cpu()

            # save each image
            for img_tensor, img_name in zip(images, batch_img_names):
                save_path = os.path.join(output_dir, f"{img_name}.png")
                save_image(
                    img_tensor,
                    save_path,
                    normalize=True,
                    value_range=(0, 1),  # generate_image outputs already in [0, 1]
                )
