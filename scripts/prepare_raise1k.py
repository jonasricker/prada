import io
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def download_and_convert(url: str, save_path: Path) -> None:
    buffer = tempfile.SpooledTemporaryFile(max_size=100_000)
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        for chunk in r.iter_content(chunk_size=1024):
            buffer.write(chunk)
        buffer.seek(0)
        Image.open(io.BytesIO(buffer.read())).save(save_path)


def main(path_to_raise1k_csv: str | Path) -> None:
    output_dir = Path("data/darg/t2i/real")
    output_dir.mkdir(parents=True, exist_ok=True)

    executor = ThreadPoolExecutor(max_workers=8)
    futures = []
    rows = list(pd.read_csv(path_to_raise1k_csv).iterrows())
    for index, row in rows:
        output_path = output_dir / (row["File"] + ".png")
        if not output_path.exists():
            futures.append(executor.submit(download_and_convert, row["TIFF"], output_path))

    # track progress of successful downloads
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading RAISE-1k"):
        pass


if __name__ == "__main__":
    fire.Fire(main)
