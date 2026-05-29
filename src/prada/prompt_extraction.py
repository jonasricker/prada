import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS
from tqdm import tqdm
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)


def download_and_save(model_cls, processor_cls, model_name, local_dir):
    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        # Check if model weights & config already exist
        if any(fname.endswith(".bin") or fname.endswith(".safetensors") for fname in os.listdir(local_dir)):
            print(f"Already downloaded... ({model_name} is in {local_dir}). Skipping...")
            return
    print(f"Downloading {model_name}...")
    model = model_cls.from_pretrained(model_name)
    processor = processor_cls.from_pretrained(model_name)
    model.save_pretrained(local_dir)
    processor.save_pretrained(local_dir)
    print(f"Done, saved {model_name} to {local_dir}!")


def extract_prompts(
    image_folder: str | Path,
    prompt_file: str | Path,
    checkpoint_dir: str | Path,
):
    """
    Extract captions for all images in a folder using BLIP2.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_folder = Path(image_folder)
    prompt_file = Path(prompt_file)
    checkpoint_dir = Path(checkpoint_dir)

    download_and_save(
        Blip2ForConditionalGeneration,
        Blip2Processor,
        "Salesforce/blip2-flan-t5-xl",
        f"{checkpoint_dir}/blip2-flan-t5-xl",
    )

    processor = Blip2Processor.from_pretrained(checkpoint_dir / "blip2-flan-t5-xl")
    model_obj = Blip2ForConditionalGeneration.from_pretrained(f"{checkpoint_dir}/blip2-flan-t5-xl").to(device)

    # iterate over images
    results = []
    paths = sorted([path for path in image_folder.iterdir() if path.suffix in IMG_EXTENSIONS])
    for img_path in tqdm(paths, desc="Extracting prompts"):
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model_obj.generate(**inputs, max_new_tokens=50)
        pred_text = processor.decode(out[0], skip_special_tokens=True)

        results.append({"image": img_path.stem, "prompt": pred_text})

    # save results to CSV
    df = pd.DataFrame(results)
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prompt_file, index=False)
    print(f"Saved results to {prompt_file}.")
