import shutil
from pathlib import Path

import fire
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def main(path_to_imagenet_val: str | Path) -> None:
    output_root = Path("data/darg/c2i/real")
    output_root.mkdir()

    ds = ImageFolder(root=path_to_imagenet_val)
    indices, _ = train_test_split(
        range(len(ds)),
        train_size=10_000,
        stratify=ds.targets,
        random_state=0,
        shuffle=True,
    )

    for index in tqdm(indices, desc="Recreating ImageNet subset"):
        output_dir = output_root / f"{ds.targets[index]:03d}"
        output_dir.mkdir(exist_ok=True)
        shutil.copy(ds.imgs[index][0], output_dir / Path(ds.imgs[index][0]).name)

    print(f"Done. ImageNet subset saved to {output_root}.")


if __name__ == "__main__":
    fire.Fire(main)
