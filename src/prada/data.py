from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS


def get_dataloader(
    image_dir: str | Path,
    transform: Callable,
    mode: Literal["c2i", "t2i"],
    batch_size: int,
    num_workers: int,
    prompt_file: str | Path | None = None,
) -> DataLoader:
    if mode == "c2i":
        ds = ImageFolder(image_dir, transform=transform)
    elif mode == "t2i":
        ds = ImagePromptDataset(ds=FlatImageFolder(image_dir, transform=transform), prompt_file=prompt_file)

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


class FlatImageFolder(Dataset):
    def __init__(self, root: str | Path, transform: Callable | None = None):
        root = Path(root)
        self.paths = sorted([path for path in root.iterdir() if path.suffix in IMG_EXTENSIONS])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class ImagePromptDataset(Dataset):
    def __init__(self, ds: FlatImageFolder, prompt_file: str | Path) -> None:
        self.ds = ds
        self.prompt_df = pd.read_csv(prompt_file)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx) -> tuple[Tensor, str]:
        image_name = Path(self.ds.paths[idx]).stem
        prompt = self.prompt_df.loc[self.prompt_df["image"] == image_name]["prompt"].item()

        return self.ds[idx], prompt


def cat_tensor_dicts(
    tensor_dict_list: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    out = {}
    for key in tensor_dict_list[0].keys():
        if isinstance(tensor_dict_list[0][key], Tensor):
            out[key] = torch.cat([d[key] if d[key].ndim > 0 else d[key].unsqueeze(dim=0) for d in tensor_dict_list])
        else:
            print(f"Could not concatenate {key}.")
    return out


def tensor_dict_to_cpu(tensor_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: tensor.cpu() for key, tensor in tensor_dict.items()}
