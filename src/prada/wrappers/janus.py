import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoModelForCausalLM

from external.Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from external.Janus.janus.models.vq_model import compute_entropy_loss
from prada.misc import apply_to_dict

from .base import Wrapper

# CAUTION:
# - janus requirements differ from current uv env (torch==2.0.1...)
# - test whether that makes a difference...


class JanusWrapper(Wrapper):
    range_after_transform = (-1, 1)  # just as Llamagen (Janus uses the same VQ-VAE as Llamagen)

    def __init__(
        self, variant: str = "1b", checkpoints_root: str | Path = "checkpoints", device: str = "cuda", **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.local_dir = str(Path(checkpoints_root) / "janus")
        self.variant = f"Janus-Pro-{variant.upper()}"
        model_path = os.path.join(self.local_dir, self.variant)

        # check if model_path exists
        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist! Downloading model from HuggingFace...")
            print(f"Requested Model name: {self.variant}, Local dir: {self.local_dir}")

            from huggingface_hub import snapshot_download

            if self.variant == "Janus-Pro-1B":
                repo_id = "deepseek-ai/Janus-Pro-1B"
            elif self.variant == "Janus-Pro-7B":
                repo_id = "deepseek-ai/Janus-Pro-7B"
            else:
                raise ValueError(
                    f"Unknown model name {self.variant}. Currently supporting Janus-Pro-1B and Janus-Pro-7B."
                )

            snapshot_download(repo_id=repo_id, local_dir=model_path)
            print(f"Model downloaded to {model_path}")

        print(f"Loading model from {model_path}...")

        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

        # add to self
        self.vl_chat_processor = vl_chat_processor
        self.tokenizer = tokenizer
        self.vl_gpt = vl_gpt

        # image size
        self.image_size = 384  # Janus default

    @property
    def transform(self):
        return transforms.Compose(
            [
                # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.image_size)),
                transforms.ToTensor(),
                transforms.Resize(self.image_size),  # Resize smaller side to img_size, preserve aspect ratio
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                # transform to torch.bfloat16 (Janus requires this)
                # transforms.Lambda(lambda x: x.to(torch.bfloat16)),
                transforms.ConvertImageDtype(torch.bfloat16),  # changed for multiprocessing
            ]
        )

    @torch.inference_mode()
    def get_ae_rec_and_quant_error(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        """Compute the AE reconstruction (D(E(x))) and quantization error.
        Janus also uses VQ-VAE from Llamagen.
        Returns rec_B3HW and quant_err_BL.
        """

        def vq_forward(self, z):
            # from original VQ-VAE code, also in .../janus/models/vq_model.py

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

            # compute loss for embedding
            # if self.training: # always compute vq_loss! (no aggregation)
            vq_loss = (z_q - z.detach()) ** 2
            # commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            # entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

            # preserve gradients
            z_q = z + (z_q - z).detach()

            # reshape back to match original input shape
            z_q = torch.einsum("b h w c -> b c h w", z_q)

            return (
                z_q,
                (vq_loss, commit_loss, entropy_loss),
                (perplexity, min_encodings, min_encoding_indices),
            )

        def encode(self, x):
            h = self.encoder(x)
            h = self.quant_conv(h)
            quant, emb_loss, info = vq_forward(self.quantize, h)
            return quant, emb_loss, info

        def forward(self, input):
            quant, emb_loss, info = encode(self=self, x=input)
            # use vq_loss as quantization error, see src/external/LlamaGen/tokenizer/tokenizer_image/vq_model.py
            quantization_error = emb_loss[0]
            dec = self.decode(quant)
            return dec, quantization_error  # return quantization error

        rec_B3HW, quant_err_BL = forward(
            self=self.vl_gpt.gen_vision_model,  # the VQ-VAE
            input=image_B3HW.to(self.device),
        )
        rec_B3HW = rec_B3HW.clamp(-1, 1)

        return dict(
            rec_B3HW=rec_B3HW,
            quant_err_BL=quant_err_BL.mean(dim=-1).flatten(start_dim=1),  # average over channels, then flatten tokens
        )

    def generate_image(self, condition_B: Tensor | list[str], seed: int) -> dict[str, Tensor]:
        """Generate an image from class label or prompt.
        Based on the sampling script provided in official Janus repo: https://github.com/deepseek-ai/Janus/blob/main/generation_inference.py
        Returns image_B3HW.
        """

        # set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        @torch.inference_mode()
        def generate(
            mmgpt: MultiModalityCausalLM,
            vl_chat_processor: VLChatProcessor,
            prompt: str,
            temperature: float = 1,
            parallel_size: int = 16,
            cfg_weight: float = 5,
            image_token_num_per_image: int = 576,
            img_size: int = 384,
            patch_size: int = 16,
            # out_dir: str = "generated_images",
        ):
            input_ids = vl_chat_processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)

            tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
            for i in range(parallel_size * 2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = vl_chat_processor.pad_id

            inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

            for i in range(image_token_num_per_image):
                outputs = mmgpt.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=outputs.past_key_values if i != 0 else None,
                )
                hidden_states = outputs.last_hidden_state

                logits = mmgpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]

                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

            dec = mmgpt.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
            )
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

            dec = np.clip((dec + 1) / 2 * 255, 0, 255)

            visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec

            return visual_img

        # ensure that condition is list of strings?
        if isinstance(condition_B, str):
            condition_B = [condition_B]

        conversation = [
            {
                "role": "<|User|>",
                "content": condition_B[0],  # only support batch size 1 for now
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag

        print(f"Generating image for prompt: {prompt}")

        samples = generate(
            mmgpt=self.vl_gpt,
            vl_chat_processor=self.vl_chat_processor,
            prompt=prompt,  # assumes batch size 1 for now
            parallel_size=1,  # just one image for now
        )

        return dict(image_B3HW=samples)

    def get_gt_idx(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        """Compute the token representation for an image.
        Return gt_idx_BL.
        """
        _, _, [_, _, indices] = self.vl_gpt.gen_vision_model.encode(image_B3HW)  # encoding with VQ-VAE
        gt_idx_BL = indices.reshape(len(image_B3HW), -1)
        return dict(gt_idx_BL=gt_idx_BL)

    @torch.inference_mode()
    def get_logits(self, gt_idx_BL: Tensor, condition_B: Tensor, return_image: bool = False) -> dict[str, Tensor]:
        """Compute the model's conditional and unconditional logits from tokens and condition (label or prompt).

        Return cond_logits_BLX and uncond_logits_BLX. If `return_image` is true, return decoded image for debugging.
        """
        cond_logits_BLV, uncond_logits_BLV = [], []

        def generate(
            mmgpt: MultiModalityCausalLM,
            vl_chat_processor: VLChatProcessor,
            prompt: str,
            temperature: float = 1,
            parallel_size: int = 1,
            cfg_weight: float = 5,
            image_token_num_per_image: int = 576,
            img_size: int = 384,
            patch_size: int = 16,
            return_image: bool = False,  # whether to return the image for debugging
        ):
            input_ids = vl_chat_processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)

            tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
            for i in range(parallel_size * 2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = vl_chat_processor.pad_id

            inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

            for i in range(image_token_num_per_image):
                outputs = mmgpt.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=outputs.past_key_values if i != 0 else None,
                )
                hidden_states = outputs.last_hidden_state

                logits = mmgpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]

                # --------------------------------------------- #
                # extract logits
                cond_logits_BLV.append(logit_cond.detach())
                uncond_logits_BLV.append(logit_uncond.detach())
                # --------------------------------------------- #

                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                # --------------------------------------------- #
                # overwrite next token with gt token
                # print(next_token.shape, gt_idx_BL[:, i].unsqueeze(-1).shape)
                next_token = gt_idx_BL[:, i].unsqueeze(-1)
                # --------------------------------------------- #
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

            if return_image:
                dec = mmgpt.gen_vision_model.decode_code(
                    generated_tokens.to(dtype=torch.int),
                    shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
                )
                dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

                dec = np.clip((dec + 1) / 2 * 255, 0, 255)

                visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
                visual_img[:, :, :] = dec
                return visual_img
            else:
                return generated_tokens

        # prepare conversation/prompt
        conversation = [
            {
                "role": "<|User|>",
                "content": condition_B[0],  # only support batch size 1 for now
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag

        output = generate(self.vl_gpt, self.vl_chat_processor, prompt=prompt, return_image=return_image)
        if return_image:
            return output
        else:
            return dict(
                cond_logits_BLX=torch.cat(cond_logits_BLV, dim=0).unsqueeze(0),  # shape (B=1, L, V)
                uncond_logits_BLX=torch.cat(uncond_logits_BLV, dim=0).unsqueeze(0),
            )

    def prompts_to_image_synthbuster(
        self, output_dir, prompts_csv, n_samples_per_class=1, correct_aspect_ratios=False, batch_size=1, seed=0
    ):
        import pandas as pd
        from tqdm import tqdm

        print(f"Generating {n_samples_per_class} samples for each entry in {prompts_csv}...")

        df = pd.read_csv(prompts_csv)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            prompt = row["Prompt"]
            img_name = row["image name (matching Raise-1k)"]
            if n_samples_per_class == 1:
                # print(f"Generating image for prompt: {prompt}") # for debugging

                out = self.generate_image([prompt], seed=seed)
                output_file = f"{output_dir}/{img_name}.png"

                # save to file
                pil_image = PIL.Image.fromarray(out["image_B3HW"][0])
                pil_image.save(output_file)
            else:
                for j in range(n_samples_per_class):
                    # print(f"Generating image for prompt: {prompt} with seed: {j}") # for debugging
                    out = self.generate_image([prompt], seed=j)
                    output_file = f"{output_dir}/{img_name}_{j:02d}.png"

                    # save to file
                    pil_image = PIL.Image.fromarray(out["image_B3HW"][0])
                    pil_image.save(output_file)
        print(f"All images generated and saved to {output_dir}.")
