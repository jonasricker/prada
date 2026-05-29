# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

from typing import Optional, Union

import torch

from external.HMAR.models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from external.HMAR.models.vqvae import VQVAE
from .transformer import Transformer


class NextScalePrediction(Transformer):
    def __init__(
        self,
        vae_local: VQVAE,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        flash_if_available=True,
        fused_if_available=True,
        using_block_sparse_attn=True,
    ):
        super(NextScalePrediction, self).__init__(
            vae_local,
            num_classes=num_classes,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_eps=norm_eps,
            shared_aln=shared_aln,
            cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
            using_block_sparse_attn=using_block_sparse_attn,
        )

    @torch.no_grad()
    def generate(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        num_samples=1,
    ) -> torch.Tensor:
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
            label_B = torch.multinomial(
                self.uniform_prob, num_samples=B, replacement=True, generator=rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.num_classes if label_B < 0 else label_B,
                device=self.lvl_1L.device,
            )

        sos = cond_BD = self.class_emb(
            torch.cat(
                (label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0
            )
        )
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = (
            sos.unsqueeze(1).expand(2 * B, self.first_l, -1)
            + self.pos_start.expand(2 * B, self.first_l, -1)
            + lvl_pos[:, : self.first_l].expand(2 * B, self.first_l, -1)
        )

        cur_L = 0
        cur = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            cur_L += pn * pn
            cur += pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            for b in self.blocks:
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    using_block_sparse_attn=False,
                    attn_bias=None,
                )
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=num_samples
            )
            idx_Bl = idx_Bl[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    self.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
                )
                next_token_map = next_token_map.repeat(
                    2, 1, 1
                )  # double the batch sizes due to CFG

        return (
            self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        )  # de-normalize, from [-1, 1] to [0, 1]

    def get_word_embed(self, x: torch.Tensor, idx_to_mask) -> torch.Tensor:
        return self.word_embed(x.float())

    def forward(
        self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor
    ) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        B = x_BLCv_wo_first_l.shape[0]
        with torch.amp.autocast("cuda", enabled=True):
            label_B = torch.where(
                torch.rand(B, device=label_B.device) < self.cond_drop_rate,
                self.num_classes,
                label_B,
            )
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(
                B, self.first_l, -1
            )

            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC

        attn_bias = self.attn_bias_for_masking
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        for _, b in enumerate(self.blocks):
            x_BLC = b(
                x=x_BLC,
                cond_BD=cond_BD_or_gss,
                using_block_sparse_attn=self.using_block_sparse_attn,
                attn_bias=attn_bias,
            )
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        return x_BLC  # logits BLV, V is vocab_size
