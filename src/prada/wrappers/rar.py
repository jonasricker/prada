import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor
from torchvision import transforms

from .base import Wrapper

import external.rar.demo_util as demo_util
from external.rar.utils.train_utils import create_pretrained_tokenizer


class RARWrapper(Wrapper):
    """Yu et al. 2024. Randomized Autoregressive Visual Generation"""

    def __init__(self, rar_model_size: str = "b", checkpoints_root: str | Path = "checkpoints") -> None:
        """Derived from README_RAR.md."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_dir = Path(checkpoints_root) / "rar"

        # download the maskgit-vq tokenizer
        hf_hub_download(
            repo_id="fun-research/TiTok", filename="maskgit-vqgan-imagenet-f16-256.bin", local_dir=checkpoint_dir
        )

        # download the rar generator weight
        hf_hub_download(repo_id="yucornetto/RAR", filename=f"rar_{rar_model_size}.bin", local_dir=checkpoint_dir)

        # load config
        self.config = demo_util.get_config(
            "src/external/rar/configs/training/generator/rar.yaml"
        )
        self.config.experiment.generator_checkpoint = checkpoint_dir / f"rar_{rar_model_size}.bin"
        self.config.model.vq_model.pretrained_tokenizer_weight = checkpoint_dir / "maskgit-vqgan-imagenet-f16-256.bin"
        self.config.model.generator.hidden_size = {"b": 768, "l": 1024, "xl": 1280, "xxl": 1408}[rar_model_size]
        self.config.model.generator.num_hidden_layers = {"b": 24, "l": 24, "xl": 32, "xxl": 40}[rar_model_size]
        self.config.model.generator.num_attention_heads = 16
        self.config.model.generator.intermediate_size = {"b": 3072, "l": 4096, "xl": 5120, "xxl": 6144}[rar_model_size]

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_grad_enabled(False)

        # maskgit-vq as tokenizer
        self.tokenizer = create_pretrained_tokenizer(self.config)
        self.generator = demo_util.get_rar_generator(self.config)
        self.generator.eval()
        self.tokenizer.eval()
        self.tokenizer.to(self.device)
        self.generator.to(self.device)

    @property
    def transform(self) -> Callable:
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ]
        )

    @torch.inference_mode()
    def generate_image(self, condition_B: Tensor, seed: int) -> dict[str, Tensor]:
        torch.manual_seed(seed)
        condition_B = condition_B.to(self.device)

        image_B3HW = demo_util.sample_fn(
            generator=self.generator,
            tokenizer=self.tokenizer,
            labels=condition_B.long(),
            randomize_temperature=self.config.model.generator.randomize_temperature,
            guidance_scale=self.config.model.generator.guidance_scale,
            guidance_scale_pow=self.config.model.generator.guidance_scale_pow,
            device=self.device,
        )
        return dict(image_B3HW=image_B3HW)

    @torch.inference_mode()
    def get_gt_idx(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        """Get ground-truth token indices for the given image."""
        gt_idx_BL = self.tokenizer.encode(image_B3HW.to(self.device))
        return dict(gt_idx_BL=gt_idx_BL)

    @torch.inference_mode()
    def get_logits(self, gt_idx_BL: Tensor, condition_B: Tensor, return_image: bool = False) -> dict[str, Tensor]:
        """Derived from src/external/rar/modeling/rar.py."""

        # variables to store logits
        cond_logits_BLV, uncond_logits_BLV = [], []

        def generate(
            self, condition, guidance_scale, randomize_temperature, guidance_scale_pow, kv_cache=True, **kwargs
        ):
            condition = self.preprocess_condition(condition, cond_drop_prob=0.0)
            device = condition.device
            num_samples = condition.shape[0]
            ids = torch.full((num_samples, 0), -1, device=device)
            cfg_scale = 0.0

            if kv_cache:
                self.enable_kv_cache()

            orders = None
            cfg_orders = None

            for step in range(self.image_seq_len):
                # ref: https://github.com/sail-sg/MDT/blob/441d6a1d49781dbca22b708bbd9ed81e9e3bdee4/masked_diffusion/models.py#L513C13-L513C23
                scale_pow = torch.ones((1), device=device) * guidance_scale_pow
                scale_step = (1 - torch.cos(((step / self.image_seq_len) ** scale_pow) * torch.pi)) * 1 / 2
                cfg_scale = (guidance_scale - 1) * scale_step + 1

                if guidance_scale != 0:
                    logits = self.forward_fn(
                        torch.cat([ids, ids], dim=0),
                        torch.cat([condition, self.get_none_condition(condition)], dim=0),
                        orders=cfg_orders,
                        is_sampling=True,
                    )
                    cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]

                    # store logits, first step returns two, keep last like below
                    cond_logits_BLV.append(cond_logits[:, -1].unsqueeze(1))
                    uncond_logits_BLV.append(uncond_logits[:, -1].unsqueeze(1))

                    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                else:
                    logits = self.forward_fn(ids, condition, orders=orders, is_sampling=True)

                # keep the logit of last token
                logits = logits[:, -1]
                logits = logits / randomize_temperature
                probs = F.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)

                # use ground-truth ids instead of sampled ones
                # ids = torch.cat((ids, sampled), dim=-1)
                ids = torch.cat((ids, gt_idx_BL[:, step].unsqueeze(1)), dim=-1)

            self.disable_kv_cache()
            return ids

        generated_tokens = generate(
            self=self.generator,
            condition=condition_B.to(self.device).long(),
            randomize_temperature=self.config.model.generator.randomize_temperature,
            guidance_scale=self.config.model.generator.guidance_scale,
            guidance_scale_pow=self.config.model.generator.guidance_scale_pow,
            guidance_decay="constant",
            softmax_temperature_annealing=False,
            num_sample_steps=8,
        )

        if return_image:
            return self.tokenizer.decode_tokens(generated_tokens.view(generated_tokens.shape[0], -1)).clamp(0, 1)
        else:
            # concatenate likelihoods
            return dict(
                cond_logits_BLX=torch.cat(cond_logits_BLV, dim=1),
                uncond_logits_BLX=torch.cat(uncond_logits_BLV, dim=1),
            )

    @torch.inference_mode()
    def get_ae_rec_and_quant_error(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        """Derived from src/external/rar/modeling/titok.py."""

        def encode(self, x):
            hidden_states = self.encoder(x)
            quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
            return codebook_indices.detach(), torch.mean((quantized_states - hidden_states) ** 2, dim=1).flatten(
                start_dim=1
            )  # compute and return quantization error

        def decode(self, codes):
            quantized_states = self.quantize.get_codebook_entry(codes)
            rec_images = self.decoder(quantized_states)
            rec_images = torch.clamp(rec_images, 0.0, 1.0)
            return rec_images.detach()

        codes, quant_err_BL = encode(self=self.tokenizer, x=image_B3HW.to(self.device))

        rec_B3HW = decode(self=self.tokenizer, codes=codes)
        # rec_error_B = torch.mean((rec_B3HW - image_B3HW.to(self.device)) ** 2, dim=[1, 2, 3])
        rec_B3HW = torch.clamp(rec_B3HW, 0.0, 1.0)

        return dict(
            rec_B3HW=rec_B3HW,
            quant_err_BL=quant_err_BL,  # quant_err_B=quant_err_BL.mean(dim=1), rec_error_B=rec_error_B
        )
