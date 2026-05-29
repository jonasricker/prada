from datetime import datetime
from pathlib import Path
from typing import Literal

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from prada.data import cat_tensor_dicts, get_dataloader, tensor_dict_to_cpu
from prada.metrics import aeroblade, icas
from prada.prada import PRADA
from prada.prompt_extraction import extract_prompts
from prada.wrappers import C2I_MODELS, T2I_MODELS, get_wrapper


def predict(
    real_image_dir: str | Path,
    fake_image_dirs: str | Path | list[str | Path],
    output_root: str | Path,
    mode: Literal["c2i", "t2i"] = "t2i",
    prada_weight_dir: str | Path = "prada_weights",
    model_weight_dir: str | Path = "model_weights",
    train_size: int = 250,
    epochs: int = 3000,
    seed: int = 42,
    n_hidden: int = 16,
    input_dim: int = 1,
    learnable_difference: bool = True,
    learnable_stage_weights: bool = True,
    training_noise: float = 0.1,
    batch_size: int = 1,
    num_workers: int = 1,
):
    """
    Perform detection and attribution with PRADA and baselines.
    By default, use our pre-trained weights for the score function.
    Optionally, a subset of the images is used to fit a custom score function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # prepare directories
    real_image_dir = Path(real_image_dir)
    if not isinstance(fake_image_dirs, list):
        fake_image_dirs = [fake_image_dirs]
    fake_image_dirs = [Path(fake_image_dir) for fake_image_dir in fake_image_dirs]
    all_image_dirs = {"real": real_image_dir} | {d.name: d for d in fake_image_dirs}
    prada_weight_dir = Path(prada_weight_dir)
    model_weight_dir = Path(model_weight_dir)
    output_root = Path(output_root)
    feature_dir = output_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir.mkdir(parents=True)

    # define models
    if mode == "c2i":
        models = C2I_MODELS
    elif mode == "t2i":
        models = T2I_MODELS
    else:
        raise ValueError(f"Unsupported mode '{mode}'.")

    data = {}

    for dataset, image_dir in all_image_dirs.items():
        for model in models:
            # step 1: extract (or load) likelihood- and reconstruction-based features from the AR models
            feature_file = feature_dir / f"dataset={dataset}_model={model}.pt"

            if feature_file.exists():
                features = torch.load(feature_file)
            else:
                wrapper = get_wrapper(model, checkpoints_root=model_weight_dir)

                # for T2I, load prompts or extract if not available
                if mode == "t2i":
                    prompt_file = output_root / "extracted_prompts" / f"{dataset}.csv"
                    if not prompt_file.exists():
                        extract_prompts(
                            image_folder=image_dir,
                            prompt_file=prompt_file,
                            checkpoint_dir=model_weight_dir,
                        )
                else:
                    prompt_file = None

                # create dataloader
                dl = get_dataloader(
                    image_dir=image_dir,
                    transform=wrapper.transform,
                    mode=mode,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prompt_file=prompt_file,
                )

                features = []
                for image_B3HW, condition_B in tqdm(dl, desc="Extracting likelihoods"):
                    # move data to device
                    image_B3HW = image_B3HW.to(wrapper.device)
                    if isinstance(condition_B, Tensor):
                        condition_B = condition_B.to(wrapper.device)

                    feature_dict = {}

                    # compute ground-truth tokens for image
                    gt_idx_dict = wrapper.get_gt_idx(image_B3HW=image_B3HW)

                    # compute conditional/unconditional likelihood for ground-truth tokens
                    logits_dict = wrapper.get_logits(gt_idx_BL=gt_idx_dict["gt_idx_BL"], condition_B=condition_B)
                    gt_llh_dict = wrapper.get_gt_llh(logits_BLX=logits_dict, gt_idx_BL=gt_idx_dict["gt_idx_BL"])
                    feature_dict.update(gt_llh_dict)

                    # compute quantization error
                    rec_dict = wrapper.get_ae_rec_and_quant_error(image_B3HW=image_B3HW)
                    feature_dict["quant_err"] = -rec_dict["quant_err_BL"]

                    # compute AEROBLADE, negate s.t. larger values correspond to fake class
                    feature_dict["aeroblade"] = -aeroblade(
                        img1=image_B3HW,
                        img2=rec_dict["rec_B3HW"],
                        net_type="vgg",
                        reduction="none",
                        normalize=wrapper.range_after_transform == (0, 1),
                    )

                    # append batch to result_list
                    features.append(tensor_dict_to_cpu(feature_dict))

                # combine batches and save results
                features = cat_tensor_dicts(features)
                if hasattr(wrapper, "scale_lengths"):
                    features["scale_lengths"] = wrapper.scale_lengths  # type: ignore
                torch.save(features, feature_file)

            # step 2: compute features composed from conditional/unconditional likelihoods
            features["gt_llh_diff"] = features["cond_gt_llh_BL"] - features["uncond_gt_llh_BL"]
            features["icas"] = icas(diff_gt_llh_BL=features["gt_llh_diff"])

            # step 3: prepare train/test data for PRADA
            stacked_likelihoods = torch.stack(
                [features["cond_gt_llh_BL"], features["uncond_gt_llh_BL"]], dim=-1
            ).float()

            if train_size == 0:
                features["prada_test"] = stacked_likelihoods
            else:
                features["prada_train"], features["prada_test"] = train_test_split(
                    stacked_likelihoods, train_size=train_size, random_state=seed, shuffle=True
                )

            data[dataset, model] = features

    # load or train score function (load if available in weight dir, if not train and save there) and apply to test data
    for model in models:
        scale_lengths = data["real", model].get("scale_lengths")  # use real here simply because we know the name
        prada = PRADA(
            n_hidden=n_hidden,
            scale_lengths=scale_lengths,
            input_dim=input_dim,
            learnable_difference=learnable_difference,
            learnable_stage_weights=learnable_stage_weights,
            training_noise=training_noise,
        )
        weight_path = prada_weight_dir / model / "final.pt"
        if weight_path.exists():
            prada.load_state_dict(torch.load(weight_path, weights_only=True))
            prada.eval()
            prada.to(device)
        else:
            if train_size == 0.0:
                raise ValueError(f"Cannot train PRADA. Found no weights at {weight_path} and train_frac is set to 0.0.")

            # create train features and labels
            real_x_train = data["real", model]["prada_train"]
            fake_x_train = data[model, model]["prada_train"]
            x_train = torch.cat([real_x_train, fake_x_train])
            y_train = torch.cat([torch.zeros(len(real_x_train)), torch.ones(len(fake_x_train))], dim=0).float()
            y_train = y_train * (1 - 0.05) + 0.05 / 2  # label smoothing (0.05)

            # train
            train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=len(x_train), shuffle=True)
            prada.fit(
                train_loader=train_loader,
                test_loader=None,
                model_name=model,
                epochs=epochs,
                val_every=50,
                weight_path=weight_path,
                device=device,
            )

        # apply score function to test data
        prada.eval()
        with torch.inference_mode():
            for dataset in all_image_dirs.keys():
                data[dataset, model]["prada_test"] = prada(data[dataset, model]["prada_test"].to(device)).cpu()

    # evaluate detection/attribution using different features
    for feature in ["quant_err", "aeroblade", "gt_llh_diff", "icas", "prada_test"]:
        max_score_dict = {}
        max_scores = []
        max_preds = []
        true_preds = []

        # prepare scores
        for dataset in all_image_dirs.keys():
            # compute mean if feature is not scalar
            mean_features = [
                data[dataset, model][feature].mean(dim=1, keepdim=True)
                if data[dataset, model][feature].ndim == 2
                else data[dataset, model][feature].unsqueeze(1)
                for model in models
            ]
            mean_features = torch.cat(mean_features, dim=1)

            # stack features for all models, add pseudo score for real class
            mean_features_with_real = torch.cat([torch.zeros(mean_features.shape[0], 1), mean_features], dim=1)

            # maximum score of all models is our "fake" score
            max_values = mean_features.amax(dim=1)

            # model with maximum score (or real if all below zero) is our predicted model
            max_indices = mean_features_with_real.argmax(dim=1)

            # save maximum scores, model belonging to maximum scores, and ground-truth labels
            max_score_dict[dataset] = max_values
            max_scores.extend(max_values)
            max_preds.extend((["real"] + models)[i] for i in max_indices)
            true_preds.extend([dataset] * len(mean_features))

        # compute detection metrics
        auroc = {}
        for dataset in all_image_dirs.keys():
            if dataset == "real":
                continue

            y_score_real = max_score_dict["real"]
            y_score_fake = max_score_dict[dataset]
            y_score = torch.cat([y_score_real, y_score_fake])
            y_true = [0] * len(y_score_real) + [1] * len(y_score_fake)

            auroc[dataset] = roc_auc_score(y_true=y_true, y_score=y_score)

        auroc = pd.Series(auroc, name="AUROC")
        auroc.to_csv(result_dir / f"auroc_{feature}.csv")

        # create confusion matrix plot for PRADA attribution
        if feature == "prada_test":
            crosstab = pd.crosstab(
                index=pd.Series(true_preds), columns=pd.Series(max_preds), normalize="index"
            ).reindex(index=all_image_dirs.keys(), columns=["real"] + models, fill_value=0)
            sns.heatmap(crosstab, annot=True, fmt=".2f", cmap="Blues")
            plt.xlabel("Predicted Class")
            plt.ylabel("True Class")
            plt.tight_layout()
            plt.savefig(result_dir / f"cfm_{feature}.pdf")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    fire.Fire(predict)
