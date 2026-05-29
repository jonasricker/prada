import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import trange


class PRADA(nn.Module):
    def __init__(
        self,
        n_hidden: int = 16,
        scale_lengths: list[int] | None = None,
        input_dim: int = 2,
        learnable_difference: bool = True,
        learnable_stage_weights: bool = True,
        training_noise: float = 0.3,
    ):
        super().__init__()
        if input_dim == 1:  # 1D -> 1D, optionall with learnable alpha, otherwise cond - uncond
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=learnable_difference)
        elif input_dim == 2:  # 2D -> 1D
            if learnable_difference:
                raise ValueError("input_dim == 2 and learnable_difference == True is incompatible.")

        self.learnable_difference = learnable_difference
        self.learnable_stage_weights = learnable_stage_weights
        self.input_dim = input_dim
        self.scale_lengths = scale_lengths
        self.training_noise = training_noise
        self.n_stages = len(scale_lengths) if scale_lengths is not None else 1
        if self.n_stages > 1:
            # optional learning the stage weights
            self.w = nn.Parameter(torch.ones(self.n_stages) / self.n_stages, requires_grad=learnable_stage_weights)
        else:
            # just a single stage, no need to learn it
            self.w = nn.Parameter(torch.ones(1), requires_grad=False)

        self.non_linear = nn.ELU()
        self.net = nn.Sequential(
            # layer 1
            nn.Linear(input_dim, n_hidden),
            self.non_linear,
            # layer 2
            nn.Linear(n_hidden, n_hidden),
            self.non_linear,
            # output layer
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        # expected shape: (batch_size, total_tokens, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch_size, total_tokens, 1)
        batch_size = x.size(0)
        stage_means = []
        start = 0

        for si, tokens_in_stage in enumerate(self.scale_lengths if self.scale_lengths is not None else [x.size(1)]):
            end = start + tokens_in_stage
            x_stage = x[:, start:end]  # (batch_size, tokens_in_stage, input_dim)
            x_flat = x_stage.flatten(0, 1)  # (batch_size * tokens_in_stage, input_dim)
            token_scores = self.forward_single_token(x_flat).view(batch_size, tokens_in_stage)
            stage_mean = token_scores.mean(dim=1)  # (batch_size,)
            stage_means.append(self.w[si] * stage_mean)
            start = end

        weighted_sum = torch.stack(stage_means, dim=0).sum(dim=0)  # (batch_size,)
        return weighted_sum

    def forward_single_token(self, x):
        # expected shape: (batch_size, input_dim)
        if self.input_dim == 1:
            x = (2 - self.alpha) * x[:, 0] - self.alpha * x[:, 1]  # (batch_size,) (2 - alpha) * cond - alpha * uncond
            x = x.unsqueeze(-1)  # (batch_size, 1)
        if self.training:
            # stage level noise regularization
            noise_std = torch.std(x) * self.training_noise
            x = x + noise_std * torch.randn_like(x)
        token_scores = self.net(x)  # (batch_size, 1)
        return token_scores.flatten()  # (batch_size,)

    def forward_delta_alpha(self, x):
        # expected shape: (batch_size, 1) = already the difference (2 - alpha) * cond - alpha * uncond as second input
        return self.net(x).flatten()  # (batch_size,)

    def fit(
        self,
        train_loader: DataLoader,
        device: torch.device,
        test_loader: DataLoader | None = None,
        model_name: str = "PRADA",
        lr: float = 1e-3,
        epochs: int = 1000,
        val_every: int = 50,
        weight_path: Path | None = None,
    ):
        # default AdamW, with higher learning rate for alpha
        all_params = [{"params": self.net.parameters()}]
        if self.learnable_difference:
            all_params.append({"params": [self.alpha]})
        if self.learnable_stage_weights:
            all_params.append({"params": [self.w]})

        optimizer = optim.AdamW(
            all_params,
            lr=lr,
        )
        criterion = nn.BCEWithLogitsLoss()

        train_losses = []
        train_aurocs = []

        val_losses = []
        val_aurocs = []

        best_val_auroc = 0.0
        best_epoch = -1

        alphas = []

        self.to(device)

        # full batch training as this is faster and our datasets are very small
        all_train_data = []
        all_train_labels = []
        for data, labels in train_loader:
            all_train_data.append(data)
            all_train_labels.append(labels)
        all_train_data = torch.cat(all_train_data).to(device)
        all_train_labels = torch.cat(all_train_labels).to(device).float()

        for epoch in trange(epochs, desc=f"Calibrating {model_name}"):
            self.train()

            optimizer.zero_grad()

            outputs = self.forward(all_train_data)
            loss = criterion(outputs, all_train_labels)

            # regularize with L1 norm to encourage sparsity
            L1_norm = (torch.norm(self.w, p=1) - 1.0) ** 2
            loss = loss + 0.1 * L1_norm

            loss.backward()
            optimizer.step()

            # compute metrics
            with torch.no_grad():
                preds = torch.sigmoid(outputs).detach().cpu().numpy()
                labels = all_train_labels.cpu().numpy()
                labels = [0 if x < 0.5 else 1 for x in labels]
                auroc = roc_auc_score(labels, preds)

            train_losses.append(loss.item())
            train_aurocs.append(auroc)
            alphas.append(self.alpha.data.cpu().numpy().item() if self.input_dim == 1 else None)

            # Validation
            if (epoch + 1) % val_every == 0 and test_loader is not None:
                self.eval()
                val_epoch_losses = []
                val_all_labels = []
                val_all_preds = []

                with torch.no_grad():
                    for val_batch_data, val_batch_labels in test_loader:
                        val_batch_data = val_batch_data.to(device)
                        val_batch_labels = val_batch_labels.to(device).float()

                        val_outputs = self.forward(val_batch_data)
                        val_loss = criterion(val_outputs, val_batch_labels)

                        val_epoch_losses.append(val_loss.item())
                        val_all_labels.extend(val_batch_labels.cpu().numpy())
                        val_all_preds.extend(torch.sigmoid(val_outputs).cpu().numpy())

                val_epoch_loss = sum(val_epoch_losses) / len(val_epoch_losses)
                val_all_labels = [0 if x < 0.5 else 1 for x in val_all_labels]
                val_epoch_auroc = roc_auc_score(val_all_labels, val_all_preds)

                val_losses.append(val_epoch_loss)
                val_aurocs.append(val_epoch_auroc)

                # Save best model
                if val_epoch_auroc > best_val_auroc:
                    best_epoch = epoch + 1
                    best_val_auroc = val_epoch_auroc

                    if weight_path is not None:
                        weight_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(self.state_dict(), weight_path)

        # save final model
        if weight_path is not None:
            final_model_path = weight_path / "final.pt"
            final_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), final_model_path)

            # also save config
            training_info = {
                "n_hidden": self.net[0].in_features,
                "n_stages": self.n_stages,
                "token_per_stage": self.scale_lengths,
                "input_dim": self.input_dim,
                "learnable_difference": self.learnable_difference,
                "training_noise": self.training_noise,
                "best_val_auroc": best_val_auroc,
                "best_epoch": best_epoch,
                "model_name": model_name,
                "train_losses": train_losses,
                "train_aurocs": train_aurocs,
                "val_losses": val_losses,
                "val_aurocs": val_aurocs,
                "alpha": self.alpha.data.cpu().numpy().tolist() if self.input_dim == 1 else None,
                "w": self.w.data.cpu().numpy().tolist(),
                "alphas_over_time": alphas,
            }

            # save training info as json
            config_name = f"{weight_path}_training_stats.json"
            with open(config_name, "w") as f:
                json.dump(training_info, f)

        return {
            "train_losses": train_losses,
            "train_aurocs": train_aurocs,
            "val_losses": val_losses,
            "val_aurocs": val_aurocs,
            "best_val_auroc": best_val_auroc,
            "best_epoch": best_epoch,
            "alphas_over_time": alphas,
        }
