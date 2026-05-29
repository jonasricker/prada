# [CVPR Findings 2026] PRADA: Probability-Ratio-Based Attribution and Detection of Autoregressive-Generated Images

This repository contains the official implementation of *PRADA* and instructions to reproduce our experiments.

## Setup
We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/).
Once installed, you can recreate our environment by running
```bash
uv sync
```
from within the root directory.
Depending on your environment, you might have to make changes to `pyproject.toml` to [select the right PyTorch version](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch).


## The DARG Dataset (**D**ataset of **A**uto**R**egressive-**G**enerated Images)

### Generated Images
Our generated images can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.20327169).
It contains 155,000 images from 20 autoregressive image generators.
Note that the evaluation in the paper uses a reduced subset.

### Real Images
- Download the validation set from the [official ImageNet website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and run `uv run scripts/prepare_imagenet.py path/to/imagenet/val` to recreate out subset of 10000 images.
- Download `RAISE_1k.csv` from the [RAISE-1k website](https://loki.disi.unitn.it/RAISE/confirm.php?package=1k) and run `uv run scripts/prepare_raise1k.py path/to/RAISE_1k.csv` to download and convert the images.

## Reproducing Our Results
To reproduce our results for class-to-image models, run
```bash
uv run predict.py --real-image-dir data/darg/c2i/real --fake-image-dirs "['data/darg/c2i/hmar_d20','data/darg/c2i/hmar_d30','data/darg/c2i/llamagen_b256','data/darg/c2i/llamagen_l256','data/darg/c2i/rar_l','data/darg/c2i/rar_xxl','data/darg/c2i/var_d20','data/darg/c2i/var_d30']" --mode c2i --train-size 250 --output-root results/c2i
```

To reproduce our results for text-to-image models, run
```bash
uv run predict.py --real-image-dir data/darg/t2i/real --fake-image-dirs "['data/darg/t2i/infinity_2b','data/darg/t2i/janus_1b','data/darg/t2i/llamagen_xlstage2','data/darg/t2i/switti_1024']" --mode t2i --train-size 250 --output-root results/t2i
```

These commands will automatically use our pre-trained score functions.


## Acknowledgments
All code in `src/external` is copied (and slightly modified) from the original repositories:
- [HMAR](https://github.com/NVlabs/HMAR)
- [LlamaGen](https://github.com/foundationvision/llamagen)
- [VAR](https://github.com/FoundationVision/VAR)
- [RAR](https://github.com/bytedance/1d-tokenizer/blob/main/README_RAR.md)
- [Infinity](https://github.com/FoundationVision/Infinity)
- [Janus](github.com/deepseek-ai/Janus)
- [Switti](https://github.com/yandex-research/switti)

## Citation
If you find this repository or the DARG dataset helpful, please cite our work as follows:
```
@InProceedings{Damm_2026_CVPR,
    author    = {Damm, Simon and Ricker, Jonas and Petzka, Henning and Fischer, Asja},
    title     = {PRADA: Probability-Ratio-Based Attribution and Detection of Autoregressive-Generated Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {6506-6516}
}
```