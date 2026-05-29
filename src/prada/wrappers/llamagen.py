import collections
import math
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from external.LlamaGen.autoregressive.models.generate import generate, top_k_top_p_filtering
from external.LlamaGen.autoregressive.models.gpt import GPT_models
from external.LlamaGen.language.t5 import T5Embedder
from external.LlamaGen.tokenizer.tokenizer_image.vq_model import VQ_models, compute_entropy_loss

from .base import Wrapper


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,  # instead of deprecated Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def normalize_gpt_model_name(name: str) -> str:
    name = name.strip().upper()
    if name.startswith("GPT-"):
        return name
    elif len(name) <= 3 and name.isalpha():
        return f"GPT-{name.upper()}"
    else:
        raise ValueError(f"Invalid model name format: {name}")


# fixed FixedT5Embedder, now directly downloads T5 if not available locally
class FixedT5Embedder(T5Embedder):
    available_models = ["t5-v1_1-xxl", "t5-v1_1-xl", "flan-t5-xl"]
    bad_punct_regex = re.compile(
        r"["
        + r"#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + r"\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa
    print("Info: SyntaxErrors above from the bad_punct_regex in original code...")

    def __init__(
        self,
        device,
        dir_or_name="t5-v1_1-xxl",
        *,
        local_cache=False,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=120,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        if t5_model_kwargs is None:
            t5_model_kwargs = {"low_cpu_mem_usage": True, "torch_dtype": self.torch_dtype}
            t5_model_kwargs["device_map"] = {"shared": self.device, "encoder": self.device}

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/IF_")
        self.dir_or_name = dir_or_name
        tokenizer_path, path = dir_or_name, dir_or_name

        if local_cache:
            local_path = os.path.abspath(dir_or_name)
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Local model path '{local_path}' not found. Make sure you've downloaded it first."
                )
            model_path = local_path
        else:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=dir_or_name,
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False,
                token=self.hf_token,
            )

        print(f"Loading model from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir)
        self.model = T5EncoderModel.from_pretrained(model_path, cache_dir=cache_dir, **t5_model_kwargs).eval()
        self.model_max_length = model_max_length


class LlamaGenWrapper(Wrapper):
    range_after_transform = (-1, 1)

    def __init__(
        self,
        checkpoints_root: str | Path = "checkpoints",
        gpt_model_name: str = "GPT-B",  
        gpt_type: str = "c2i",
        gpt_stage_t2i: str = "stage1",  # only for t2i, stage1 or stage2
        vq_model_name: str = "VQ-16",
        image_size: int = 256,
    ) -> None:
        """Derived from autoregressive/sample/sample_c2i.py"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_dir = Path(checkpoints_root) / "llamagen"

        # Setup PyTorch:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_grad_enabled(False)

        if gpt_type == "c2i":
            hf_home = "https://huggingface.co/FoundationVision/LlamaGen/resolve/main"
        else:
            hf_home = "https://huggingface.co/peizesun/llamagen_t2i/resolve/main"

        self.codebook_size = 16384
        self.codebook_embed_dim = 8
        self.num_classes = 1000
        self.cls_token_num = 120 if gpt_type == "t2i" else 1
        self.precision = "bf16"
        self.downsample_size = 16
        self.image_size = image_size
        self.gpt_type = gpt_type

        if self.gpt_type == "t2i":
            assert gpt_stage_t2i in ["stage1", "stage2"], "gpt_stage_t2i must be 'stage1' or 'stage2'"
            if gpt_stage_t2i == "stage1":
                assert self.image_size == 256, "For t2i stage1, image_size must be 256"
            else:
                assert self.image_size == 512, "For t2i stage2, image_size must be 512"

        # create and load model
        self.vq_model = VQ_models[vq_model_name](
            codebook_size=self.codebook_size, codebook_embed_dim=self.codebook_embed_dim
        )
        self.vq_model.to(self.device)
        self.vq_model.eval()

        vq_ckpt = f"vq_ds{vq_model_name.split('-')[1]}_{gpt_type}.pt"
        if not (checkpoint_dir / vq_ckpt).exists():
            os.system(f"wget {hf_home}/{vq_ckpt} -P {checkpoint_dir}")
        checkpoint = torch.load(checkpoint_dir / vq_ckpt, map_location="cpu", weights_only=True)
        self.vq_model.load_state_dict(checkpoint["model"])
        del checkpoint
        print("image tokenizer is loaded")

        # create and load gpt model
        precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[self.precision]
        self.latent_size = self.image_size // self.downsample_size
        gpt_model_name = normalize_gpt_model_name(gpt_model_name)
        self.gpt_model = GPT_models[gpt_model_name](
            vocab_size=self.codebook_size,
            block_size=self.latent_size**2,
            num_classes=self.num_classes,
            cls_token_num=self.cls_token_num,
            model_type=gpt_type,
        ).to(device=self.device, dtype=precision)

        if gpt_type == "t2i":
            gpt_ckpt = f"{gpt_type}_{gpt_model_name.split('-')[1]}_{gpt_stage_t2i}_{self.image_size}.pt"
        else:
            gpt_ckpt = f"{gpt_type}_{gpt_model_name.split('-')[1]}_{self.image_size}.pt"
        print(f"Loading GPT checkpoint from {checkpoint_dir / gpt_ckpt}...")
        if not (checkpoint_dir / gpt_ckpt).exists():
            os.system(f"wget {hf_home}/{gpt_ckpt} -P {checkpoint_dir}")
        checkpoint = torch.load(checkpoint_dir / gpt_ckpt, map_location="cpu", weights_only=True)
        if gpt_type in ["GPT-XXL", "GPT-3B"]:  # fspd
            model_weight = checkpoint
        elif "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "module" in checkpoint:  # deepspeed
            model_weight = checkpoint["module"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight, maybe add --from-fsdp to run command")
        self.gpt_model.load_state_dict(model_weight, strict=False)
        self.gpt_model.eval()
        del checkpoint
        print("gpt model is loaded")

        if gpt_type == "t2i":
            t5_model = FixedT5Embedder(
                device=self.device,
                dir_or_name="google/flan-t5-xl",
                local_cache=False,  # auto-download using snapshot_download
                cache_dir=checkpoint_dir / "flan-t5-xl",
            )
            self.t5_model = t5_model
            print("t5 model is loaded")

    def get_gt_idx(self, image_B3HW):
        _, _, [_, _, indices] = self.vq_model.encode(image_B3HW.to(self.device))
        gt_idx_BL = indices.reshape(len(image_B3HW), -1)
        return dict(gt_idx_BL=gt_idx_BL)

    def apply_center_crop(self, pil_image):
        return center_crop_arr(pil_image, self.image_size)

    @property
    def transform(self) -> Callable:
        return transforms.Compose(
            [
                transforms.Lambda(self.apply_center_crop),  # changed to allow for multiprocessing
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @torch.inference_mode()
    def generate_image(
        self,
        condition_B,  # Tensor for c2i, list of text prompts for t2i
        seed: int,
        # cfg: float = 4.0,         # overwritten by default below
        cfg_interval: int = -1,
        temperature: float = 1.0,
        # top_k: int = 2000,        # overwritten by default below
        top_p: float = 1.0,
    ) -> dict[str, Tensor]:
        torch.manual_seed(seed)

        if self.gpt_type == "c2i":
            cfg = 4.0
            top_k = 2000
            c_emb_masks = None

        elif self.gpt_type == "t2i":
            cfg = 7.5
            top_k = 1000
            no_left_padding = False

            # ensure that condition is a list of strings
            assert isinstance(condition_B, list), "For t2i, condition_B should be a list of text prompts."
            caption_embs, emb_masks = self.t5_model.get_text_embeddings(condition_B)

            if not no_left_padding:
                # print(f"processing left-padding...")
                # a naive way to implement left-padding
                new_emb_masks = torch.flip(emb_masks, dims=[-1])
                new_caption_embs = []
                for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                    valid_num = int(emb_mask.sum().item())
                    # print(f"  prompt {idx} token len: {valid_num}")
                    new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                    new_caption_embs.append(new_caption_emb)
                new_caption_embs = torch.stack(new_caption_embs)
            else:
                new_caption_embs, new_emb_masks = caption_embs, emb_masks
            c_indices = new_caption_embs * new_emb_masks[:, :, None]
            c_emb_masks = new_emb_masks

            # c_indices are now the condition_B for t2i
            condition_B = c_indices

        condition_B = condition_B.to(self.device)
        qzshape = [len(condition_B), self.codebook_embed_dim, self.latent_size, self.latent_size]
        index_sample = generate(
            self.gpt_model,
            condition_B,
            self.latent_size**2,
            emb_masks=c_emb_masks,  # None for c2i
            cfg_scale=cfg,
            cfg_interval=cfg_interval,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=True,
        )
        samples = self.vq_model.decode_code(index_sample, qzshape)  # output value is between [-1, 1]
        samples = samples.add(1).mul(0.5).clamp(0, 1).cpu()
        return dict(image_B3HW=samples)

    @torch.inference_mode()
    def get_logits(self, condition_B: Tensor, gt_idx_BL: Tensor, return_image: bool = False) -> dict[str, Tensor]:
        """Derived from src/external/LlamaGen/autoregressive/models/generate.py."""
        # variables to store logits
        cond_logits_BLV, uncond_logits_BLV = [], []

        def sample(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, sample_logits=True):
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0 or top_p < 1.0:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            if sample_logits:
                # replace sampled idx with ground truth
                # idx = torch.multinomial(probs, num_samples=1)
                idx = gt_idx_BL[:, len(cond_logits_BLV) - 1].unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
            return idx, probs

        def prefill(model, cond_idx: Tensor, input_pos: Tensor, cfg_scale: float, **sampling_kwargs):
            if cfg_scale > 1.0:
                logits, _ = model(None, cond_idx, input_pos)
                logits_combined = logits
                cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)

                # store logits
                cond_logits_BLV.append(cond_logits)
                uncond_logits_BLV.append(uncond_logits)

                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits, _ = model(None, cond_idx, input_pos)

            return sample(logits, **sampling_kwargs)[0]

        def decode_one_token(model, x: Tensor, input_pos: Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
            assert input_pos.shape[-1] == 1
            if cfg_scale > 1.0:
                x_combined = torch.cat([x, x])
                logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
                logits_combined = logits
                cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)

                # store logits
                cond_logits_BLV.append(cond_logits)
                uncond_logits_BLV.append(uncond_logits)

                if cfg_flag:
                    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                else:
                    logits = cond_logits
            else:
                logits, _ = model(x, cond_idx=None, input_pos=input_pos)
            return sample(logits, **sampling_kwargs)

        def decode_n_tokens(
            model,
            cur_token: Tensor,
            input_pos: Tensor,
            num_new_tokens: int,
            cfg_scale: float,
            cfg_interval: int,
            **sampling_kwargs,
        ):
            new_tokens, new_probs = [], []
            cfg_flag = True
            for i in range(num_new_tokens):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True
                ):  # Actually better for Inductor to codegen attention here
                    if cfg_interval > -1 and i > cfg_interval:
                        cfg_flag = False
                    next_token, next_prob = decode_one_token(
                        model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
                    )
                    input_pos += 1
                    new_tokens.append(next_token.clone())
                    new_probs.append(next_prob.clone())
                    cur_token = next_token.view(-1, 1)

            return new_tokens, new_probs

        def generate(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.1, cfg_interval=-1, **sampling_kwargs):
            if model.model_type == "c2i":
                if cfg_scale > 1.0:
                    cond_null = torch.ones_like(cond) * model.num_classes
                    cond_combined = torch.cat([cond, cond_null])
                else:
                    cond_combined = cond
                T = 1
            elif model.model_type == "t2i":
                if cfg_scale > 1.0:
                    cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
                    cond_combined = torch.cat([cond, cond_null])
                else:
                    cond_combined = cond
                T = cond.shape[1]
            else:
                raise Exception("please check model type")

            T_new = T + max_new_tokens
            max_seq_length = T_new
            max_batch_size = cond.shape[0]

            device = cond.device
            with torch.device(device):
                max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
                model.setup_caches(
                    max_batch_size=max_batch_size_cfg,
                    max_seq_length=max_seq_length,
                    dtype=model.tok_embeddings.weight.dtype,
                )

            if emb_masks is not None:
                assert emb_masks.shape[0] == max_batch_size
                assert emb_masks.shape[-1] == T
                if cfg_scale > 1.0:
                    model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat(
                        [emb_masks, emb_masks]
                    ).unsqueeze(1)
                else:
                    model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

                eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
                model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix

            # create an empty tensor of the expected final shape and fill in the current tokens
            seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

            input_pos = torch.arange(0, T, device=device)
            next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
            seq[:, T : T + 1] = next_token

            input_pos = torch.tensor([T], device=device, dtype=torch.int)
            generated_tokens, _ = decode_n_tokens(
                model, next_token, input_pos, max_new_tokens - 1, cfg_scale, cfg_interval, **sampling_kwargs
            )
            seq[:, T + 1 :] = torch.cat(generated_tokens, dim=1)

            return seq[:, T:]

        if self.gpt_type == "c2i":
            cfg = 4.0
            top_k = 2000
        elif self.gpt_type == "t2i":
            cfg = 7.5
            top_k = 1000
            no_left_padding = False

            # ensure that condition is a list of strings
            assert isinstance(condition_B, collections.abc.Iterable), (
                "For t2i, condition_B should be a list of text prompts."
            )
            caption_embs, emb_masks = self.t5_model.get_text_embeddings(condition_B)

            if not no_left_padding:
                # print(f"processing left-padding...")
                # a naive way to implement left-padding
                new_emb_masks = torch.flip(emb_masks, dims=[-1])
                new_caption_embs = []
                for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                    valid_num = int(emb_mask.sum().item())
                    # print(f"  prompt {idx} token len: {valid_num}")
                    new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                    new_caption_embs.append(new_caption_emb)
                new_caption_embs = torch.stack(new_caption_embs)
            else:
                new_caption_embs, new_emb_masks = caption_embs, emb_masks
            c_indices = new_caption_embs * new_emb_masks[:, :, None]
            c_emb_masks = new_emb_masks

            # c_indices are now the condition_B for t2i
            condition_B = c_indices

        condition_B = condition_B.to(self.device)
        output = generate(
            self.gpt_model,
            condition_B,
            self.latent_size**2,
            emb_masks=(c_emb_masks if self.gpt_model == "t2i" else None),
            cfg_scale=cfg,
            top_k=top_k,
            sample_logits=True,
        )

        if return_image:
            img = self.vq_model.decode_code(
                output, [len(condition_B), self.codebook_embed_dim, self.latent_size, self.latent_size]
            )
            print(img.min(), img.max())
            img = img.add(1).mul(0.5).clamp(0, 1).cpu()
            return img
        else:
            return dict(
                cond_logits_BLX=torch.cat(cond_logits_BLV, dim=1)[:, -output.shape[1] :],
                uncond_logits_BLX=torch.cat(uncond_logits_BLV, dim=1)[:, -output.shape[1] :],
            )

    @torch.inference_mode()
    def get_ae_rec_and_quant_error(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        """Derived from src/external/LlamaGen/tokenizer/tokenizer_image/vq_model.py."""

        def vq_forward(self, z):
            # reshape z -> (batch, height, width, channel) and flatten
            z = torch.einsum("b c h w -> b h w c", z).contiguous()
            z_flattened = z.view(-1, self.e_dim)
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

            if self.l2_norm:
                z = F.normalize(z, p=2, dim=-1)
                z_flattened = F.normalize(z_flattened, p=2, dim=-1)
                embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
            else:
                embedding = self.embedding.weight

            d = (
                torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(embedding**2, dim=1)
                - 2 * torch.einsum("bd,dn->bn", z_flattened, torch.einsum("n d -> d n", embedding))
            )

            min_encoding_indices = torch.argmin(d, dim=1)
            z_q = embedding[min_encoding_indices].view(z.shape)
            perplexity = None
            min_encodings = None
            vq_loss = None
            commit_loss = None
            entropy_loss = None
            codebook_usage = 0

            if self.show_usage and self.training:
                cur_len = min_encoding_indices.shape[0]
                self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
                self.codebook_used[-cur_len:] = min_encoding_indices
                codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

            # compute loss for embedding
            # if self.training:
            # vq_loss = torch.mean((z_q - z.detach()) ** 2)
            vq_loss = (z_q - z.detach()) ** 2
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

            # preserve gradients
            z_q = z + (z_q - z).detach()

            # reshape back to match original input shape
            z_q = torch.einsum("b h w c -> b c h w", z_q)

            return (
                z_q,
                (vq_loss, commit_loss, entropy_loss, codebook_usage),
                (perplexity, min_encodings, min_encoding_indices),
            )

        def encode(self, x):
            h = self.encoder(x)
            h = self.quant_conv(h)
            quant, emb_loss, info = vq_forward(self=self.quantize, z=h)
            return (
                quant,
                emb_loss,
                info,
                # torch.mean((quant - h) ** 2, dim=1).flatten(start_dim=1), # <- this gives super large values? normalization not applied?
            )

        def forward(self, input):
            quant, emb_loss, info = encode(self=self, x=input)
            # use vq_loss as quantization error, see src/external/LlamaGen/tokenizer/tokenizer_image/vq_model.py
            quantization_error = emb_loss[0]
            # commitment_loss = emb_loss[1]
            # entropy_loss = emb_loss[2]

            # quantization_error = quantization_error.unsqueeze(0) -> fixed in the code base! src/external/LlamaGen/tokenizer/tokenizer_image/vq_model.py

            dec = self.decode(quant)
            return dec, quantization_error  # return quantization error

        rec_B3HW, quant_err_BL = forward(self=self.vq_model, input=image_B3HW.to(self.device))

        # compute the reconstruction error
        # rec_error_B = torch.mean((rec_B3HW - image_B3HW.to(self.device)) ** 2, dim=(1, 2, 3))

        rec_B3HW = rec_B3HW.clamp(-1, 1)

        return dict(
            rec_B3HW=rec_B3HW,
            # rec_error_B=rec_error_B,
            quant_err_BL=quant_err_BL.mean(dim=-1).flatten(start_dim=1),
        )

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
            images = result["image_B3HW"]
            if isinstance(images, torch.Tensor):
                images = images.detach().cpu()

            # save each image
            for img_tensor, img_name in zip(images, batch_img_names):
                save_path = os.path.join(output_dir, f"{img_name}.png")
                save_image(
                    img_tensor,
                    save_path,
                    normalize=True,
                    value_range=(0, 1),  # generate_image outputs already in [0, 1]
                )
