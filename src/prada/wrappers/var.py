import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch import distributed as tdist
from torch.nn import functional as F
from torchvision.transforms import transforms

from external.VAR.models import build_vae_var
from external.VAR.models.basic_var import AdaLNSelfAttn
from external.VAR.models.helpers import gumbel_softmax_with_rng
from external.VAR.utils.data import normalize_01_into_pm1

from .base import Wrapper


class VARWrapper(Wrapper):
    """Tian et al. 2024, Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"""

    range_after_transform = (-1, 1)

    def __init__(self, model_depth: int = 16, checkpoints_root: str | Path = "checkpoints"):
        """Derived from demo_sample.ipynb."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_dir = Path(checkpoints_root) / "var"

        # download checkpoint
        hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
        vae_ckpt, var_ckpt = "vae_ch160v4096z32.pth", f"var_d{model_depth}.pth"
        if not (checkpoint_dir / vae_ckpt).exists():
            os.system(f"wget {hf_home}/{vae_ckpt} -P {checkpoint_dir}")
        if not (checkpoint_dir / var_ckpt).exists():
            os.system(f"wget {hf_home}/{var_ckpt} -P {checkpoint_dir}")

        # build vae, var
        FOR_512_px = model_depth == 36
        if FOR_512_px:
            patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
        else:
            patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        if "vae" not in globals() or "var" not in globals():
            self.vae, self.var = build_vae_var(
                V=4096,
                Cvae=32,
                ch=160,
                share_quant_resi=4,
                device=self.device,
                patch_nums=patch_nums,
                num_classes=1000,
                depth=model_depth,
                shared_aln=FOR_512_px,
            )

        # define scale lengths
        self.scale_lengths = [l**2 for l in self.var.patch_nums]

        # load checkpoints
        self.vae.load_state_dict(torch.load(checkpoint_dir / vae_ckpt, map_location="cpu"), strict=True)
        self.var.load_state_dict(torch.load(checkpoint_dir / var_ckpt, map_location="cpu"), strict=True)
        self.vae.eval()
        self.var.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.var.parameters():
            p.requires_grad_(False)

        # seed
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # run faster
        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high" if tf32 else "highest")

    @property
    def transform(self) -> Callable:
        reso = 512 if self.var.depth == 36 else 256
        return transforms.Compose(
            [
                transforms.Resize(reso, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop((reso, reso)),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        )

    @torch.inference_mode()
    def generate_image(
        self,
        condition_B: Tensor | None,
        seed: int | None = 0,
        cfg: float = 1.5,
        top_k: int = 900,
        top_p: float = 0.96,
        more_smooth: bool = False,
    ) -> dict[str, Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
            image_B3HW = self.var.autoregressive_infer_cfg(
                B=len(condition_B),
                label_B=condition_B.to(self.device),
                cfg=cfg,
                top_k=top_k,
                top_p=top_p,
                g_seed=seed,
                more_smooth=more_smooth,
            ).cpu()
        return dict(image_B3HW=image_B3HW)

    @torch.inference_mode()
    def get_gt_idx(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        """Get ground-truth token indices from the input image."""
        gt_idx_BL = torch.cat(self.vae.img_to_idxBl(image_B3HW.to(self.device)), dim=1)
        return dict(gt_idx_BL=gt_idx_BL)

    @torch.inference_mode()
    def get_logits(self, gt_idx_BL: Tensor, condition_B: Tensor, return_image: bool = False) -> dict[str, Tensor]:
        """Derived from src/external/VAR/models/var.py."""

        gt_idx_SBL = torch.split(gt_idx_BL, split_size_or_sections=self.scale_lengths, dim=-1)

        # variables to store logits
        cond_logits_BLV, uncond_logits_BLV = [], []

        def autoregressive_infer_cfg(
            self,
            B: int,
            label_B: Optional[Union[int, torch.LongTensor]],
            g_seed: Optional[int] = None,
            cfg=1.5,
            top_k=0,
            top_p=0.0,
            more_smooth=False,
        ) -> Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
            """Patch: store logits."""
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
                    (B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device
                )

            sos = cond_BD = self.class_emb(
                torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0)
            )

            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
            next_token_map = (
                sos.unsqueeze(1).expand(2 * B, self.first_l, -1)
                + self.pos_start.expand(2 * B, self.first_l, -1)
                + lvl_pos[:, : self.first_l]
            )

            cur_L = 0
            f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

            for b in self.blocks:
                b.attn.kv_caching(True)
            for si, pn in enumerate(self.patch_nums):  # si: i-th segment
                ratio = si / self.num_stages_minus_1
                # last_L = cur_L
                cur_L += pn * pn
                # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
                cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                x = next_token_map
                AdaLNSelfAttn.forward
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.get_logits(x, cond_BD)

                # store logits
                cond_logits, uncond_logits = logits_BlV[:B], logits_BlV[B:]
                cond_logits_BLV.append(cond_logits)
                uncond_logits_BLV.append(uncond_logits)

                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

                # replace sampled idx with ground truth
                # idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
                idx_Bl = gt_idx_SBL[si]

                if not more_smooth:  # this is the default case
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
                else:  # not used when evaluating FID/IS/Precision/Recall
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                    h_BChw = gumbel_softmax_with_rng(
                        logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                    ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, len(self.patch_nums), f_hat, h_BChw
                )
                if si != self.num_stages_minus_1:  # prepare for next stage
                    next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                    next_token_map = (
                        self.word_embed(next_token_map) + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
                    )
                    next_token_map = next_token_map.repeat(2, 1, 1)  # double the batch sizes due to CFG

            for b in self.blocks:
                b.attn.kv_caching(False)
            return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)  # de-normalize, from [-1, 1] to [0, 1]

        condition_B = condition_B.to(self.device)
        img = autoregressive_infer_cfg(self=self.var, B=len(condition_B), label_B=condition_B)

        # concatenate along scales
        if return_image:
            return img
        else:
            return dict(
                cond_logits_BLX=torch.cat(cond_logits_BLV, dim=1), uncond_logits_BLX=torch.cat(uncond_logits_BLV, dim=1)
            )

    def get_ae_rec_and_quant_error(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        quant_err_BL = []
        quant_err_B = []
        quant_err_B_loss = []

        def f_to_idxBl_or_fhat(
            self,
            f_BChw: Tensor,
            to_fhat: bool,
            v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        ) -> List[Union[Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
            B, C, H, W = f_BChw.shape
            f_no_grad = f_BChw.detach()
            f_rest = f_no_grad.clone()
            f_hat = torch.zeros_like(f_rest)

            f_hat_or_idx_Bl: List[Tensor] = []

            patch_hws = [
                (pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)
            ]  # from small to large
            assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f"{patch_hws[-1]=} != ({H=}, {W=})"

            SN = len(patch_hws)
            for si, (ph, pw) in enumerate(patch_hws):  # from small to large
                if 0 <= self.prog_si < si:
                    break  # progressive training: not supported yet, prog_si always -1
                # find the nearest embedding
                z_NC = (
                    F.interpolate(f_rest, size=(ph, pw), mode="area").permute(0, 2, 3, 1).reshape(-1, C)
                    if (si != SN - 1)
                    else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                )
                if self.using_znorm:
                    z_NC = F.normalize(z_NC, dim=-1)
                    idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                        self.embedding.weight.data.square(), dim=1, keepdim=False
                    )
                    d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                idx_Bhw = idx_N.view(B, ph, pw)

                # --- store quantization error --- #
                z_q = self.embedding(idx_Bhw).reshape(B, ph, pw, C)
                z = z_NC.reshape(B, ph, pw, C)
                quant_err_BL.append(torch.mean((z_q - z) ** 2, dim=3).reshape(B, ph * pw))
                # quant_err_BL.append(torch.mean(torch.abs(z_q - z), dim=3).reshape(B, ph * pw))
                quant_err_B.append(torch.mean((z_q - z) ** 2, dim=(1, 2, 3)).reshape(B, -1))
                # --- #

                h_BChw = (
                    F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode="bicubic").contiguous()
                    if (si != SN - 1)
                    else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                )
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat.add_(h_BChw)
                f_rest.sub_(h_BChw)
                f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw))

                # --- store vq loss as in the training loss, see src/external/VAR/models/quant.py --- #
                mean_vq_loss = torch.mean((f_hat.data - f_BChw) ** 2, dim=(1, 2, 3)).mul_(self.beta) + torch.mean(
                    (f_hat - f_no_grad) ** 2, dim=(1, 2, 3)
                )
                quant_err_B_loss.append(mean_vq_loss.reshape(B, -1))
                # --- #

            return f_hat_or_idx_Bl

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
        f_BChw = self.vae.quant_conv(self.vae.encoder(image_B3HW.to(self.device)))
        f_hat, usages, mean_vq_loss = forward(self=self.vae.quantize, f_BChw=f_BChw)
        rec_B3HW = self.vae.decoder(self.vae.post_quant_conv(f_hat)).clamp_(-1, 1)

        return dict(
            rec_B3HW=rec_B3HW,
            quant_err_BL=mean_vq_loss,
        )
