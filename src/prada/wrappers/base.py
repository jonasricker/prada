import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from prada.misc import apply_to_dict


class Wrapper(ABC):
    """Abstract base class for wrapping VARs.

    Allows for easy sampling, extraction of logits/likelihoods, and computation of reconstruction/quantization errors.
    Legend:
      - B: batch size
      - H: height
      - W: width
      - L: token sequence length
      - V: vocabulary size
      - S: scale (for multiscale models)
      - X: one or more dimensions depending on model specifics
    """

    range_after_transform = (0, 1)

    @property
    @abstractmethod
    def transform(self) -> Callable:
        """Model's transform to be used in DataLoader."""
        pass

    @abstractmethod
    def get_gt_idx(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        """Compute the token representation for an image.

        Return gt_idx_BL.
        """
        pass

    @abstractmethod
    def get_logits(self, gt_idx_BL: Tensor, condition_B: Tensor, return_image: bool = False) -> dict[str, Tensor]:
        """Compute the model's conditional and unconditional logits from tokens and condition (label or prompt).

        Return cond_logits_BLX and uncond_logits_BLX. If `return_image` is true, return decoded image for debugging.
        """
        pass

    def get_gt_llh(self, logits_BLX: Tensor | dict[str, Tensor], gt_idx_BL: Tensor) -> dict[str, Tensor]:
        """Compute log-likelihood for each ground-truth token from logits.
        Default implementation assumes logits_BLV.

        Return gt_llh_BL.
        """
        if isinstance(logits_BLX, dict):
            return apply_to_dict(func=self.get_gt_llh, tensor_dict=logits_BLX, gt_idx_BL=gt_idx_BL)
        else:
            llh_BLV = logits_BLX.log_softmax(dim=-1)
            gt_llh_BL = torch.gather(llh_BLV, dim=-1, index=gt_idx_BL.unsqueeze(-1)).squeeze(-1)
            return dict(gt_llh_BL=gt_llh_BL)

    def get_llh_mu(self, logits_BLX: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute expectation of log-likelihood.
        Default implementation assumes logits_BLV.

        # Update: This is just the negative entropy...

        Return llh_mu_BL.
        """
        if isinstance(logits_BLX, dict):
            return apply_to_dict(func=self.get_llh_mu, tensor_dict=logits_BLX)
        else:
            lh = logits_BLX.softmax(dim=-1)
            llh = logits_BLX.log_softmax(dim=-1)
            mu_BL = (lh * llh).sum(dim=-1)
            return dict(llh_mu_BL=mu_BL)

    def get_llh_sigma(self, logits_BLX: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute standard deviation of log-likelihood.
        Default implementation assumes logits_BLV.

        Return llh_sigma_BL.
        """
        if isinstance(logits_BLX, dict):
            return apply_to_dict(func=self.get_llh_sigma, tensor_dict=logits_BLX)
        else:
            lh = logits_BLX.softmax(dim=-1)
            llh = logits_BLX.log_softmax(dim=-1)
            mu_BL = (lh * llh).sum(dim=-1)
            sigma_BL = (lh * torch.square(llh)).sum(dim=-1) - torch.square(mu_BL)
            return dict(llh_sigma_BL=sigma_BL)

    def get_entropy(self, logits_BLX: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute entropy from logits.
        Default implementation assumes logits_BLV.

        Return entropy_BL.
        """
        if isinstance(logits_BLX, dict):
            return apply_to_dict(func=self.get_entropy, tensor_dict=logits_BLX)
        else:
            lh_BLV = logits_BLX.softmax(dim=-1)
            entropy_BL = -torch.sum(lh_BLV * torch.log(lh_BLV + 1e-10), dim=-1)
            return dict(entropy_BL=entropy_BL)

    @abstractmethod
    def get_ae_rec_and_quant_error(self, image_B3HW: Tensor) -> dict[str, Tensor]:
        """Compute the AE reconstruction (D(E(x))) and quantization error (MSE).

        Return rec_B3HW and quant_err_BL.
        """
        pass

    @abstractmethod
    def generate_image(self, condition_B: Tensor | list[str], seed: int) -> dict[str, Tensor]:
        """Generate an image from class label or prompt."""
        pass

    def sample_imagenet(
        self,
        output_dir: str | Path,
        class_labels: Optional[Tensor] = None,
        samples_per_class: int = 10,
        batch_size: int = 16,
        seed: int = 0,
        **kwargs,
    ) -> None:
        """Generate images from ImageNet class labels and save them."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        if class_labels is None:
            class_labels = torch.arange(1000)

        for i in tqdm(range(samples_per_class), desc="samples_per_class"):
            for label_B in tqdm(
                itertools.batched(class_labels, n=batch_size),
                desc="batches",
                total=math.ceil(len(class_labels) / batch_size),
            ):
                label_B = torch.tensor(label_B, dtype=torch.int)
                image_B3HW = self.generate_image(condition_B=label_B, seed=seed, **kwargs)["image_B3HW"]
                for sample_idx, sample in enumerate(image_B3HW):
                    image = to_pil_image(sample)
                    (output_dir / f"{int(label_B[sample_idx]):03d}").mkdir(exist_ok=True, parents=True)
                    image.save(output_dir / f"{int(label_B[sample_idx]):03d}" / f"{i:04d}.png")
                seed += 1

    def sample_from_prompts(
        self,
        output_dir: str | Path,
        prompts_csv_path: str,
        batch_size: int = 1,
        seed: int = 0,
        **kwargs
    ):
        """Generate images from prompts and save them."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        df = pd.read_csv(prompts_csv_path)
        prompts, filenames = df["Prompt"], df["image name (matching Raise-1k)"]

        for indices in tqdm(
            itertools.batched(range(len(prompts)), n=batch_size),
            desc="batches",
            total=math.ceil(len(prompts) / batch_size),
        ):
            condition_B = [prompts[i] for i in indices]
            save_paths = [filenames[i] for i in indices]
            image_B3HW = self.generate_image(condition_B=condition_B, seed=seed, **kwargs)["image_B3HW"]
            for sample_idx, sample in enumerate(image_B3HW):
                image = to_pil_image(sample)
                image.save(output_dir / f"{save_paths[sample_idx]}.png")
            seed += 1