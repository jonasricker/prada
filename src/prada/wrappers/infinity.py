import argparse
import os
from pathlib import Path
from turtle import pd

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image as PImage
from torch import Tensor
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from external.Infinity.infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_
from external.Infinity.tools.run_infinity import *
from prada.misc import apply_to_dict

from .base import Wrapper


def load_tokenizer_local(t5_path="google/flan-t5-xl", cache_dir=None):
    print(f"[Loading tokenizer and text encoder]")
    # text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    # load directly from the hub
    text_tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-xl", cache_dir=cache_dir)
    text_tokenizer.model_max_length = 512
    # text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
    # load directly from the hub
    text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-xl", cache_dir=cache_dir)
    text_encoder.to("cuda")
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder


class InfinityWrapper(Wrapper):
    range_after_transform = (-1, 1)

    def __init__(self, checkpoints_root: str | Path = "checkpoints", **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_dir = Path(checkpoints_root) / "infinity"

        # add weights to sys path
        print("Loading Infinity model...")
        model_path = hf_hub_download(
            repo_id="FoundationVision/Infinity", filename="infinity_2b_reg.pth", local_dir=checkpoint_dir
        )
        print(" --- Download model if necessary:", model_path)
        vae_path = hf_hub_download(
            repo_id="FoundationVision/Infinity", filename="infinity_vae_d32reg.pth", local_dir=checkpoint_dir
        )
        print(" --- Download VAE checkpoint if necessary:", vae_path)

        args = argparse.Namespace(
            pn="1M",
            model_path=model_path,
            cfg_insertion_layer=0,
            vae_type=32,
            vae_path=vae_path,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            model_type="infinity_2b",
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            text_encoder_ckpt="google/flan-t5-xl",  # checkpoint_dir, #text_encoder_ckpt,
            text_channels=2048,
            apply_spatial_patchify=0,
            h_div_w_template=1.000,
            use_flex_attn=0,
            cache_dir="/dev/shm",
            checkpoint_type="torch",
            seed=0,
            bf16=1,
            save_file="tmp.jpg",
            enable_model_cache=False,
        )
        self.args = args

        # load the components

        # load text encoder
        print(" --- Loading text encoder...")
        self.text_tokenizer, self.text_encoder = load_tokenizer_local(
            t5_path=self.args.text_encoder_ckpt, cache_dir=checkpoint_dir
        )
        # load vae
        print(" --- Loading VAE...")
        self.vae = load_visual_tokenizer(self.args)
        # load infinity
        print(" --- Loading Infinity model (Transformer)...")
        self.infinity = load_transformer(self.vae, self.args)
        print("Infinity model loaded.")

        h_div_w = 1
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn]["scales"]
        self.scale_lengths = [h * w for (_, h, w) in scale_schedule]

    @torch.inference_mode()
    def prompt_to_image(self, prompt, output_file="ipynb_tmp.jpg", h_div_w=1 / 1, seed=42):
        # print(f"Generating image for prompt: {prompt}, saving it to {output_file}")
        # params from 'interactive_infer.pynb' -- can/should be adapted
        cfg = 3
        tau = 0.5

        # h_div_w = h_div_w # aspect ratio, height:width
        # seed = random.randint(0, 10000) # fixed seed
        enable_positive_prompt = 0

        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn]["scales"]
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        generated_image = gen_one_img(
            self.infinity,
            self.vae,
            self.text_tokenizer,
            self.text_encoder,
            prompt,
            g_seed=seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[self.args.cfg_insertion_layer],
            vae_type=self.args.vae_type,
            sampling_per_bits=self.args.sampling_per_bits,
            enable_positive_prompt=enable_positive_prompt,
        )
        dir_name = osp.dirname(output_file)
        if dir_name:  # only create if not empty
            os.makedirs(dir_name, exist_ok=True)
        # os.makedirs(osp.dirname(output_file), exist_ok=True)
        cv2.imwrite(output_file, generated_image.cpu().numpy())
        # print(f'Save to {osp.abspath(output_file)}')

    @torch.inference_mode()
    def prompts_to_image_imagenet(self, output_dir, list_of_classes, n_samples_per_class=10, synthbuster=False):
        print(f"Generating {n_samples_per_class} samples for each class in {output_dir}")

        prefix = "a photo of a"  # simplest prompt - adapt if necessary - might have to little variation? change the prompting?
        for i, class_name in tqdm(enumerate(list_of_classes)):
            prompt = f"{prefix} {class_name}"
            for j in range(n_samples_per_class):
                self.prompt_to_image(prompt, output_file=f"{output_dir}/{i:03d}/{j:04d}.png")

    def prompts_to_image_synthbuster(
        self, output_dir, prompts_csv, n_samples_per_class=1, correct_aspect_ratios=False, batch_size=1, seed=0
    ):
        print(f"Generating {n_samples_per_class} samples for each entry in {prompts_csv}")

        df = pd.read_csv(prompts_csv)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            prompt = row["Prompt"]
            img_name = row["image name (matching Raise-1k)"]
            if correct_aspect_ratios:
                h_div_w = str(row["Midjourney aspect ratio"])  # using this column for aspect ratio...
                h_div_w = h_div_w.replace(":", "/")
                h_div_w = 1 / eval(h_div_w)  # Midjourney seems to be w_div_h?
            else:
                h_div_w = 1 / 1
            if n_samples_per_class == 1:
                self.prompt_to_image(prompt, output_file=f"{output_dir}/{img_name}.png", h_div_w=h_div_w)
            else:
                for j in range(n_samples_per_class):
                    self.prompt_to_image(prompt, output_file=f"{output_dir}/{img_name}_{j:04d}.png", h_div_w=h_div_w)

    @torch.inference_mode()
    def generate_image(self, label_B, seed):
        pass  # not applicable here! # double check, c2i might be possible as well...?

    @torch.inference_mode()
    def get_ae_rec_and_quant_error(self, image_B3HW, return_image=False):
        # image_B3HW: expected shape [B, 3, H, W]
        h_div_w = 1 / 1  # aspect ratio, height:width
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn]["scales"]
        self.scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        def encode(input_image, return_image=False):
            # input_image: shape [B, 3, H, W]
            h, z, all_indices, all_bit_indices, residual_norm_per_scale, var_input = self.vae.encode(
                input_image, scale_schedule=self.scale_schedule, return_residual_norm_per_scale=True
            )
            img_recon, vq_output = self.vae(input_image)
            recon_error = F.mse_loss(input_image, img_recon, reduction="none").mean(dim=[1, 2, 3])  # shape [B]

            # print(residual_norm_per_scale.shape)
            residual_norm_1S = torch.stack(residual_norm_per_scale)  # FOR B=1!
            # print(residual_norm_BS.shape)
            return recon_error, residual_norm_1S, img_recon

        # ensure on cuda
        if not isinstance(image_B3HW, torch.Tensor):
            image_B3HW = image_B3HW.unsqueeze(0).to("cuda")  # shape [1, 3, H, W]
        else:
            image_B3HW = image_B3HW.to("cuda")  # shape [B, 3, H, W]

        recon_error, residual_norm_1S, img_recon = encode(image_B3HW, return_image=True)
        residual_norm_1S = residual_norm_1S.unsqueeze(0)  # FOR B=1!

        rec_B3HW = img_recon.clamp(-1, 1)  # ensure in valid range

        return dict(
            # rec_error_B=recon_error, #cpu(),
            quant_err_BL=residual_norm_1S,  # .cpu(),  # quantization error is now per scale! but only bs=1 supported..
            rec_B3HW=rec_B3HW,  # .cpu(),
        )

    def get_gt_idx(self, image_B3HW: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get the ground truth token index for the given label."""
        # cfg = 3
        # tau = 0.5
        # Infer scale_schedule for encoding!
        h_div_w = 1 / 1  # aspect ratio, height:width
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn]["scales"]
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        # compute ground-truth tokens from input image
        with torch.no_grad():
            # ensure on cuda
            if not isinstance(image_B3HW, torch.Tensor):
                image_B3HW = image_B3HW.unsqueeze(0).to("cuda")  # shape [1, 3, H, W]
            else:
                image_B3HW = image_B3HW.to("cuda")  # shape [B, 3, H, W]
            h, z, all_indices, all_bit_indices, residual_norm_per_scale, var_input = self.vae.encode(
                image_B3HW, scale_schedule=scale_schedule
            )
            gt_bit_indices = all_bit_indices
            # f
        return dict(gt_idx_BL=gt_bit_indices)

    @torch.inference_mode()
    def get_logits(self, gt_idx_BL: Tensor, condition_B: Tensor, return_image: bool = False) -> dict[str, torch.Tensor]:
        """derived from src/external/Infinity/infinity/models/infinity.py, function: autoregressive_infer_cfg"""

        # variables to store logits
        logits_cond_BLV, logits_uncond_BLV = [], []
        idx_Bld_list_sampled = []

        cfg = 3
        tau = 0.5

        # Infer scale_schedule for encoding!
        h_div_w = 1 / 1  # aspect ratio, height:width
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn]["scales"]
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        # BS = 1, ensure that condition_B is just a string (not list of str)
        if isinstance(condition_B, tuple) or isinstance(condition_B, list):
            condition_B = condition_B[0]

        def autoregressive_infer_cfg(
            self,
            vae=None,
            scale_schedule=None,
            label_B_or_BLT=None,
            B=1,
            negative_label_B_or_BLT=None,
            force_gt_Bhw=None,
            g_seed=None,
            cfg_list=[],
            tau_list=[],
            cfg_sc=3,
            top_k=0,
            top_p=0.0,
            returns_vemb=0,
            ratio_Bl1=None,
            gumbel=0,
            norm_cfg=False,
            cfg_exp_k: float = 0.0,
            cfg_insertion_layer=[-5],
            vae_type=0,
            softmax_merge_topk=-1,
            ret_img=False,
            trunk_scale=1000,
            gt_leak=0,
            gt_ls_Bl=None,
            inference_mode=False,
            save_img_path=None,
            sampling_per_bits=1,
        ):  # returns List[idx_Bl]
            if g_seed is None:
                rng = None
            else:
                self.rng.manual_seed(g_seed)
                rng = self.rng
            assert len(cfg_list) >= len(scale_schedule)
            assert len(tau_list) >= len(scale_schedule)

            # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify,
            # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
            if self.apply_spatial_patchify:
                vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = scale_schedule

            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            if any(np.array(cfg_list) != 1):
                bs = 2 * B
                if not negative_label_B_or_BLT:
                    kv_compact_un = kv_compact.clone()
                    total = 0
                    for le in lens:
                        kv_compact_un[total : total + le] = (self.cfg_uncond)[:le]
                        total += le
                    kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                    cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
                else:
                    kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                    kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                    cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:] + cu_seqlens_k[-1]), dim=0)
                    max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
            else:
                bs = B

            kv_compact = self.text_norm(kv_compact)
            sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))  # sos shape: [2, 4096]
            kv_compact = self.text_proj_for_ca(kv_compact)  # kv_compact shape: [304, 4096]
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
            last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

            with torch.amp.autocast("cuda", enabled=False):
                cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
            accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
            idx_Bl_list, idx_Bld_list = [], []

            if inference_mode:
                for b in self.unregistered_blocks:
                    (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
            else:
                assert self.num_block_chunks > 1
                for block_chunk_ in self.block_chunks:
                    for module in block_chunk_.module.module:
                        (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)

            abs_cfg_insertion_layers = []
            add_cfg_on_logits, add_cfg_on_probs = False, False
            leng = len(self.unregistered_blocks)
            for item in cfg_insertion_layer:
                if item == 0:  # add cfg on logits
                    add_cfg_on_logits = True
                elif item == 1:  # add cfg on probs
                    add_cfg_on_probs = True  
                elif item < 0:  # determine to add cfg at item-th layer's output
                    assert leng + item > 0, (
                        f"cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}"
                    )
                    abs_cfg_insertion_layers.append(leng + item)
                else:
                    raise ValueError(f"cfg_insertion_layer: {item} is not valid")

            num_stages_minus_1 = len(scale_schedule) - 1
            summed_codes = 0
            for si, pn in enumerate(scale_schedule):  # si: i-th segment
                cfg = cfg_list[si]
                if si >= trunk_scale:
                    break
                cur_L += np.array(pn).prod()

                need_to_pad = 0
                attn_fn = None
                if self.use_flex_attn:
                    # need_to_pad = (self.pad_to_multiplier - cur_L % self.pad_to_multiplier) % self.pad_to_multiplier
                    # if need_to_pad:
                    #     last_stage = F.pad(last_stage, (0, 0, 0, need_to_pad))
                    attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[: (si + 1)]), None)

                # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
                layer_idx = 0
                for block_idx, b in enumerate(self.block_chunks):
                    # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                    if self.add_lvl_embeding_only_first_block and block_idx == 0:
                        last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                    if not self.add_lvl_embeding_only_first_block:
                        last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)

                    for m in b.module:
                        last_stage = m(
                            x=last_stage,
                            cond_BD=cond_BD_or_gss,
                            ca_kv=ca_kv,
                            attn_bias_or_two_vector=None,
                            attn_fn=attn_fn,
                            scale_schedule=scale_schedule,
                            rope2d_freqs_grid=self.rope2d_freqs_grid,
                            scale_ind=si,
                        )
                        if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                            # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                            last_stage = cfg * last_stage[:B] + (1 - cfg) * last_stage[B:]
                            last_stage = torch.cat((last_stage, last_stage), 0)
                        layer_idx += 1

                if (cfg != 1) and add_cfg_on_logits:
                    # print(f'add cfg on add_cfg_on_logits')
                    logits_BlV = self.get_logits(last_stage, cond_BD).mul(1 / tau_list[si])

                    # ------------------- #
                    # logits_BlV is of shape [2*B, ...], FIRST = conditional, LAST = unconditional. Reasoning from code: kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                    logits_cond_BLV.append(logits_BlV[:B].detach().cpu())
                    logits_uncond_BLV.append(logits_BlV[B:].detach().cpu())
                    # print("cond and uncond logits shape:", logits_BlV[:B].shape, logits_BlV[B:].shape)

                    # ------------------- #
                    logits_BlV = cfg * logits_BlV[:B] + (1 - cfg) * logits_BlV[B:]
                else:
                    logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau_list[si])

                if self.use_bit_label:
                    tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                    logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                    # print("logits shape:", logits_BlV.shape)
                    idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                        logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1
                    )[:, :, 0]
                    # print("idx_Bld shape:", idx_Bld.shape)
                    idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
                else:
                    idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                        logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1
                    )[:, :, 0]
                if vae_type != 0:
                    assert returns_vemb
                    if si < gt_leak:
                        idx_Bld = gt_ls_Bl[si]
                    else:
                        assert pn[0] == 1
                        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)  # shape: [B, h, w, d] or [B, h, w, 4d]
                        if self.apply_spatial_patchify:  # unpatchify operation
                            idx_Bld = idx_Bld.permute(0, 3, 1, 2)  # [B, 4d, h, w]
                            idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2)  # [B, d, 2h, 2w]
                            idx_Bld = idx_Bld.permute(0, 2, 3, 1)  # [B, 2h, 2w, d]
                        idx_Bld = idx_Bld.unsqueeze(1)  # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                    # ---------------------- # Here, the next token(s) are appended... -> Overwrrite with gt_token!
                    # store the sampled index...
                    idx_Bld_list_sampled.append(idx_Bld)  # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                    # overwrite with ground truth!
                    # idx_Bld_list.append(idx_Bld) # original code
                    idx_Bld_list.append(gt_idx_BL[si])  # force ground truth token into context
                    idx_Bld = gt_idx_BL[si]  # force ground truth token into context
                    # ---------------------- #

                    codes = vae.quantizer.lfq.indices_to_codes(
                        idx_Bld, label_type="bit_label"
                    )  # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    if si != num_stages_minus_1:
                        summed_codes += F.interpolate(
                            codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up
                        )
                        last_stage = F.interpolate(
                            summed_codes, size=vae_scale_schedule[si + 1], mode=vae.quantizer.z_interplote_up
                        )  # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                        last_stage = last_stage.squeeze(-3)  # [B, d, h, w] or [B, d, 2h, 2w]
                        if self.apply_spatial_patchify:  # patchify operation
                            last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2)  # [B, 4d, h, w]
                        last_stage = last_stage.reshape(*last_stage.shape[:2], -1)  # [B, d, h*w] or [B, 4d, h*w]
                        last_stage = torch.permute(last_stage, [0, 2, 1])  # [B, h*w, d] or [B, h*w, 4d]
                    else:
                        summed_codes += codes
                else:
                    if si < gt_leak:
                        idx_Bl = gt_ls_Bl[si]
                    h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()  # BlC

                    # h_BChw = h_BChw.float().transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1])
                    h_BChw = h_BChw.transpose_(1, 2).reshape(
                        B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2]
                    )
                    ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                    idx_Bl_list.append(idx_Bl)
                    if si != num_stages_minus_1:
                        accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(
                            si, num_stages_minus_1 + 1, accu_BChw, h_BChw, scale_schedule
                        )

                if si != num_stages_minus_1:
                    last_stage = self.word_embed(self.norm0_ve(last_stage))
                    last_stage = last_stage.repeat(bs // B, 1, 1)

            if inference_mode:
                for b in self.unregistered_blocks:
                    (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
            else:
                assert self.num_block_chunks > 1
                for block_chunk_ in self.block_chunks:
                    for module in block_chunk_.module.module:
                        (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)

            if not ret_img:
                return ret, idx_Bl_list, []

            if vae_type != 0:
                img = vae.decode(summed_codes.squeeze(-3))
            else:
                img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

            img = (img + 1) / 2
            img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
            return ret, idx_Bl_list, img

        def gen_one_img(
            infinity_test,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            cfg_list=[],
            tau_list=[],
            negative_prompt="",
            scale_schedule=None,
            top_k=900,
            top_p=0.97,
            cfg_sc=3,
            cfg_exp_k=0.0,
            cfg_insertion_layer=-5,
            vae_type=0,
            gumbel=0,
            softmax_merge_topk=-1,
            gt_leak=-1,
            gt_ls_Bl=None,
            g_seed=None,
            sampling_per_bits=1,
            enable_positive_prompt=0,
        ):
            sstt = time.time()
            if not isinstance(cfg_list, list):
                cfg_list = [cfg_list] * len(scale_schedule)
            if not isinstance(tau_list, list):
                tau_list = [tau_list] * len(scale_schedule)
            text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
            if negative_prompt:
                negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
            else:
                negative_label_B_or_BLT = None
            # print(f"cfg: {cfg_list}, tau: {tau_list}")
            # with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True):  
                stt = time.time()
                # _, _, img_list = infinity_test.autoregressive_infer_cfg(
                _, _, img_list = autoregressive_infer_cfg(
                    infinity_test,
                    vae=vae,
                    scale_schedule=scale_schedule,
                    label_B_or_BLT=text_cond_tuple,
                    g_seed=g_seed,
                    B=1,
                    negative_label_B_or_BLT=negative_label_B_or_BLT,
                    force_gt_Bhw=None,
                    cfg_sc=cfg_sc,
                    cfg_list=cfg_list,
                    tau_list=tau_list,
                    top_k=top_k,
                    top_p=top_p,
                    returns_vemb=1,
                    ratio_Bl1=None,
                    gumbel=gumbel,
                    norm_cfg=False,
                    cfg_exp_k=cfg_exp_k,
                    cfg_insertion_layer=cfg_insertion_layer,
                    vae_type=vae_type,
                    softmax_merge_topk=softmax_merge_topk,
                    ret_img=True,
                    trunk_scale=1000,
                    gt_leak=gt_leak,
                    gt_ls_Bl=gt_ls_Bl,
                    inference_mode=True,
                    sampling_per_bits=sampling_per_bits,
                )
            # print(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")
            img = img_list[0]
            return img

        # generate the sample!
        # prompt = "The image features a smiling woman holding a pizza slice and a glass of wine."

        generated_image = gen_one_img(
            self.infinity,
            self.vae,
            self.text_tokenizer,
            self.text_encoder,
            condition_B,  # using the provided prompt (label_B = prompt, for a single image for now...)
            g_seed=42,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],  # [args.cfg_insertion_layer],
            vae_type=32,  # args.vae_type,
            sampling_per_bits=1,  # args.sampling_per_bits,
            enable_positive_prompt=0,  # enable_positive_prompt,
        )

        # add generated image if requested
        if return_image:
            return generated_image.cpu()
        else:
            # concatenate likelihoods
            out_dict = dict(
                cond_logits_BLX=logits_cond_BLV,
                uncond_logits_BLX=logits_uncond_BLV,
                # idx_Bld_list_sampled=idx_Bld_list_sampled, # sampled indices (do not return for now to comply with extract_features.py -> get_gt_llh)
            )
            return out_dict

    # Source src/external/Infinity/infinity/dataset/dataset_t2i_iterable.py
    def transform_(pil_img, tgt_h=1024, tgt_w=1024):
        width, height = pil_img.size
        if width / height <= tgt_w / tgt_h:
            resized_width = tgt_w
            resized_height = int(tgt_w / (width / height))
        else:
            resized_height = tgt_h
            resized_width = int((width / height) * tgt_h)
        pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
        # crop the center out
        arr = np.array(pil_img)
        crop_y = (arr.shape[0] - tgt_h) // 2
        crop_x = (arr.shape[1] - tgt_w) // 2
        im = to_tensor(arr[crop_y : crop_y + tgt_h, crop_x : crop_x + tgt_w])
        # print(f'im size {im.shape}')
        return im.add(im).add_(-1)

    @staticmethod
    def normalize(t):
        return t.add(t).add_(-1)

    @property
    def transform(self):

        return transforms.Compose(
            [
                transforms.Resize(1024),
                transforms.CenterCrop(1024),
                transforms.ToTensor(),
                # transforms.Lambda(lambda t: t.add(t).add_(-1)),  # Normalize to [-1, 1]
                transforms.Lambda(self.normalize),  # rewrote to allow for multiprocessing
            ]
        )

    def get_gt_llh(self, logits_BLX: Tensor | dict[str, Tensor], gt_idx_BL: Tensor) -> dict[str, Tensor]:
        """Compute log-likelihood for each ground-truth token from logits for Infinity.
        Here, we have multiple binary classifiers per token (e.g. 32 for 32-bit quantization).
        By independence, we sum the log-likelihoods over all binary classifiers per token.

        Return gt_llh_BL.
        """
        if isinstance(logits_BLX, dict):
            return apply_to_dict(func=self.get_gt_llh, tensor_dict=logits_BLX, gt_idx_BL=gt_idx_BL)
        else:
            lls_per_token = []  # sum over all binary classifications per token
            lls_all = []  # individual log-likelihoods from all binary classifiers
            # lls_per_stage = []

            for logits_k, gt_idx_k in zip(logits_BLX, gt_idx_BL):
                B = gt_idx_k.shape[0]  # batch size (should be 1 for now)
                gt_orig_shape = gt_idx_k.shape[1:]  # original shape of the gt tokens

                logits_k = logits_k.view(B, -1, 2)  # (B, l, 2) # binary logits for infinity
                gt_idx_k = gt_idx_k.cpu().long().view(B, -1, 1)  # (B, l, 1) # binary gt idx for infinity

                # print("Number of token/features in this stage:", gt_orig_shape[-3], "x", gt_orig_shape[-2], "=", gt_orig_shape[-3]*gt_orig_shape[-2])
                # print(" --- Shapes logits:", logits_k.shape, "+ Shapes gt_idx:", gt_idx_k.shape)

                llh_k = logits_k.log_softmax(dim=-1)
                gt_llh_k = torch.gather(llh_k, dim=-1, index=gt_idx_k).squeeze(-1)  # (B, l)

                # append log-likelihoods across all binary classifications
                lls_all.append(gt_llh_k)

                # reshape back to the gt token shape
                gt_llh_k = gt_llh_k.view(B, *gt_orig_shape)  # (B, original_shape...)
                # print(" --- Reshaped:", gt_llh_k.shape)

                # loglikelihood per entry = sum log likelihoods over all 32 binary classifications (independence)
                gt_llh_k = gt_llh_k.sum(dim=-1)
                # print(" --- Summed over binary classifications:", gt_llh_k.shape)

                # flatten to (B, num_entries)
                gt_llh_k = gt_llh_k.view(B, -1)
                # print(" --- Flattened:", gt_llh_k.shape)

                lls_per_token.append(gt_llh_k)

                # print("Avg. likelihood for this stage:", gt_llh_k.exp().mean().item())
                # lls_per_stage.append(gt_llh_k.mean().item())

            lls_all = torch.cat(lls_all, dim=1)  # (B, total_length)
            lls_per_token = torch.cat(lls_per_token, dim=1)  # (B, total_length)
            return dict(gt_llh_BL=lls_per_token)

    def get_llh_mu(self, logits_BLX: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute expectation of log-likelihood.

        Return llh_mu_BL.
        """
        if isinstance(logits_BLX, dict):
            return apply_to_dict(func=self.get_llh_mu, tensor_dict=logits_BLX)
        else:
            mu_per_token = []  # sum over all binary classifications per token
            mu_all = []  # individual expectations from all binary classifiers

            # std_per_token = []  # sum over all binary classifications per token
            # std_all = []        # individual stds from all binary classifiers

            for logits_k in logits_BLX:
                B, T2, V2 = logits_k.shape
                logits_k = logits_k.view(B, -1, 2)  # (B, l, 2) # binary logits for infinity

                # print("Stage with logits shape:", logits_k.shape)

                lh_k = logits_k.softmax(dim=-1)  # probabilities
                llh_k = logits_k.log_softmax(dim=-1)  # log-probabilities
                # print(" --- shapes llh_k:", llh_k.shape)

                # per bit expectation and std of log-likelihood
                mu_k = (lh_k * llh_k).sum(dim=-1)  # (B, l)
                second_moment_k = (lh_k * llh_k**2).sum(dim=-1)  # (B, l)
                var_k = second_moment_k - mu_k**2

                # print(" --- shapes mu_k:", mu_k.shape)
                # mu_all.append(mu_k)
                # std_all.append(var_k.sqrt())

                # per 32-bit token -> by indipendence, sum over all 32 binary classifications
                mu_k = mu_k.view(B, -1, 32).sum(dim=-1)  # (B, num_tokens)
                var_k = var_k.view(B, -1, 32).sum(dim=-1)  # (B, num_tokens)
                # print(" --- shapes mu_k (per token):", mu_k.shape, "var_k:", var_k.shape)
                mu_per_token.append(mu_k)
                # std_per_token.append(var_k.sqrt())

            return dict(
                # per token (sum over all binary classifications)
                llh_mu_BL=torch.cat(mu_per_token, dim=1),  # (B, L)
                # llh_std_BL=torch.cat(std_per_token, dim=1),  # (B, L)
                # all individual classifiers
                # llh_mu_all=torch.cat(mu_all, dim=1),  # (B, total_length)
                # llh_std_all=torch.cat(std_all, dim=1),  # (B, total_length)
            )

    def get_llh_sigma(self, logits_BLX: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute standard deviation of log-likelihood.

        Return llh_std_BL.
        """
        if isinstance(logits_BLX, dict):
            return apply_to_dict(func=self.get_llh_sigma, tensor_dict=logits_BLX)
        else:
            # mu_per_token = []  # sum over all binary classifications per token
            # mu_all = []        # individual expectations from all binary classifiers

            std_per_token = []  # sum over all binary classifications per token
            std_all = []  # individual stds from all binary classifiers

            for logits_k in logits_BLX:
                B, T2, V2 = logits_k.shape
                logits_k = logits_k.view(B, -1, 2)  # (B, l, 2) # binary logits for infinity

                # print("Stage with logits shape:", logits_k.shape)

                lh_k = logits_k.softmax(dim=-1)  # probabilities
                llh_k = logits_k.log_softmax(dim=-1)  # log-probabilities
                # print(" --- shapes llh_k:", llh_k.shape)

                # per bit expectation and std of log-likelihood
                mu_k = (lh_k * llh_k).sum(dim=-1)  # (B, l)
                second_moment_k = (lh_k * llh_k**2).sum(dim=-1)  # (B, l)
                var_k = second_moment_k - mu_k**2

                # print(" --- shapes mu_k:", mu_k.shape)
                # mu_all.append(mu_k)
                # std_all.append(var_k.sqrt())

                # per 32-bit token -> by independence, sum over all 32 binary classifications
                mu_k = mu_k.view(B, -1, 32).sum(dim=-1)  # (B, num_tokens)
                var_k = var_k.view(B, -1, 32).sum(dim=-1)  # (B, num_tokens)
                # print(" --- shapes mu_k (per token):", mu_k.shape, "var_k:", var_k.shape)
                # mu_per_token.append(mu_k)
                std_per_token.append(var_k.sqrt())

            return dict(
                # per token (sum over all binary classifications)
                # llh_mu_BL=torch.cat(mu_per_token, dim=1),  # (B, L)
                llh_sigma_BL=torch.cat(std_per_token, dim=1),  # (B, L)
                # all individual classifiers
                # llh_mu_all=torch.cat(mu_all, dim=1),  # (B, total_length)
                # llh_std_all=torch.cat(std_all, dim=1),  # (B, total_length)
            )

    def get_entropy(self, logits_BLX: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute entropy from logits.
        For Infinity, we have multiple binary classifiers per token (e.g. 32 for 32-bit quantization).
        By independence, we sum the entropies over all binary classifiers per token.

        Return entropy_BL.
        """
        if isinstance(logits_BLX, dict):
            return apply_to_dict(func=self.get_entropy, tensor_dict=logits_BLX)
        else:
            entropy_per_token = []  # sum over all binary classifications per token
            entropy_all = []  # individual entropies from all binary classifiers

            for logits_k in logits_BLX:
                B, T2, V2 = logits_k.shape

                # gt_orig shape is sqrt(T2) x sqrt(T2)
                gt_orig_shape = (int(T2**0.5), int(T2**0.5), 32)  # (h_k, w_k)
                # print("B, T2, V2:", B, T2, V2, "gt shape:",gt_orig_shape)

                logits_k = logits_k.view(B, -1, 2)  # (B, l, 2) # binary logits for infinity

                # print("Number of token/features in this stage:", gt_orig_shape[-3], "x", gt_orig_shape[-2], "=", gt_orig_shape[-3]*gt_orig_shape[-2])
                # print(" --- Shapes logits:", logits_k.shape, "+ Shapes gt_idx:", gt_idx_k.shape)

                lh_k = logits_k.softmax(dim=-1)

                # compute entropies: https://en.wikipedia.org/wiki/Binary_entropy_function (Bernoulli)
                p = lh_k[..., 1]  # probability of "bit=1"
                bit_entropy = -(p * torch.log(p + 1e-10) + (1 - p) * torch.log(1 - p + 1e-10))  # (B, l*d_k)
                entropy_all.append(bit_entropy)

                # reshape back to the gt token shape
                bit_entropy = bit_entropy.view(B, *gt_orig_shape)  # (B, original_shape...)
                # print(" --- Reshaped:", bit_entropy.shape)

                # sum over entropy of binary classifications (additivity of entropy, when RVs are independent: https://en.wikipedia.org/wiki/Entropy_(information_theory))
                bit_entropy = bit_entropy.sum(dim=-1)
                # print(" --- Summed over binary classifications:", bit_entropy.shape)

                # flatten and append
                bit_entropy = bit_entropy.view(B, -1)
                entropy_per_token.append(bit_entropy)

            entropy_all = torch.cat(entropy_all, dim=1)  # (B, total_length)
            entropy_per_token = torch.cat(entropy_per_token, dim=1)  # (B, total_length)
            return dict(entropy_BL=entropy_per_token)  # , entropy_all=entropy_all)
