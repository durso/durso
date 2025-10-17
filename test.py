"""Tabular Joint Embedding Predictive Architecture (T-JEPA) implementation.

This module condenses the core ideas of the open-source `t-jepa` project into a
single-file utility that can operate directly on :class:`pandas.DataFrame`
objects.  The implementation keeps the neural, self-supervised learning
character of JEPA models while providing a pragmatic API tailored to tabular
workflows.  The main capabilities are:

* self-supervised representation learning via a JEPA/BYOL style objective;
* Optuna-powered hyper-parameter tuning for both the neural encoder and the
  downstream clustering stage;
* UMAP-based clustering of latent embeddings to generate pseudo-labels; and
* synthetic sample generation by decoding perturbed latent representations.

Typical usage::

    >>> import pandas as pd
    >>> from t_jepa import TJEPA
    >>> df = pd.DataFrame({"x": [0.0, 0.1, 0.2, 1.8, 2.0, 2.1],
    ...                    "y": [0.1, 0.0, 0.2, 1.9, 2.2, 2.0]})
    >>> jepa = TJEPA(random_state=0)
    >>> augmented = jepa.fit_transform(df)
    >>> augmented[["x", "y", "pseudo_label", "is_synthetic"]].head()

The resulting dataframe combines the original and synthetic samples, annotated
with pseudo labels derived from UMAP clustering.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:  # Optional dependency guards for friendlier error messages.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - informative error for users
    raise ImportError(
        "t_jepa requires PyTorch to be installed. Install it with `pip install torch`."
    ) from exc

try:
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - informative error for users
    raise ImportError(
        "t_jepa requires scikit-learn to be installed. Install it with `pip install scikit-learn`."
    ) from exc

try:
    import optuna
except ImportError as exc:  # pragma: no cover - informative error for users
    raise ImportError(
        "t_jepa requires Optuna to be installed. Install it with `pip install optuna`."
    ) from exc

try:
    import umap
    from umap.umap_ import find_clusters
except ImportError as exc:  # pragma: no cover - informative error for users
    raise ImportError(
        "t_jepa requires umap-learn to be installed. Install it with `pip install umap-learn`."
    ) from exc


ArrayLike = np.ndarray


def _set_global_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the frame contains purely numeric data suitable for JEPA training."""

    if df.empty:
        raise ValueError("Input dataframe is empty.")
    numeric_df = df.apply(pd.to_numeric, errors="raise")
    return numeric_df.astype(np.float32)


class _TabularDataset(Dataset):
    """Simple :class:`torch.utils.data.Dataset` wrapper over a NumPy array."""

    def __init__(self, array: ArrayLike):
        self.array = torch.as_tensor(array, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.array[index]


class _MLP(nn.Module):
    """Feed-forward network with configurable depth and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        modules: list[nn.Module] = []
        in_dim = input_dim
        if depth == 1:
            modules.append(nn.Linear(in_dim, output_dim))
        else:
            for _ in range(depth - 1):
                modules.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                    ]
                )
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            modules.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x)


@dataclass
class _TrainingResult:
    encoder: nn.Module
    predictor: nn.Module
    decoder: nn.Module
    history: list[Dict[str, float]]
    val_loss: Optional[float]


class TJEPA:
    """Neural self-supervised augmenter inspired by the original T-JEPA project."""

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        encoder_depth: int = 3,
        predictor_depth: int = 2,
        decoder_depth: int = 3,
        dropout: float = 0.1,
        noise_std: float = 0.1,
        feature_dropout: float = 0.15,
        recon_weight: float = 1.0,
        latent_noise: float = 0.25,
        optuna_trials: int = 10,
        optuna_timeout: Optional[int] = None,
        tune_val_split: float = 0.2,
        tune_epochs: int = 25,
        silhouette_weight: float = 0.2,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_n_components: int = 2,
        min_cluster_size: int = 10,
        cluster_selection_method: str = "eom",
        synthetic_multiplier: float = 1.0,
        synthetic_temperature: float = 1.0,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if synthetic_multiplier < 0:
            raise ValueError("synthetic_multiplier must be non-negative")
        if synthetic_temperature <= 0:
            raise ValueError("synthetic_temperature must be strictly positive")

        self.epochs = epochs
        self.base_batch_size = batch_size
        self.optuna_trials = optuna_trials
        self.optuna_timeout = optuna_timeout
        self.tune_val_split = tune_val_split
        self.tune_epochs = tune_epochs
        self.silhouette_weight = silhouette_weight
        self.min_cluster_size = max(2, min_cluster_size)
        self.cluster_selection_method = cluster_selection_method
        self.synthetic_multiplier = synthetic_multiplier
        self.synthetic_temperature = synthetic_temperature
        self.random_state = random_state

        self.hparams: Dict[str, float | int] = {
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "encoder_depth": encoder_depth,
            "predictor_depth": predictor_depth,
            "decoder_depth": decoder_depth,
            "dropout": dropout,
            "noise_std": noise_std,
            "feature_dropout": feature_dropout,
            "recon_weight": recon_weight,
            "latent_noise": latent_noise,
            "umap_n_neighbors": umap_n_neighbors,
            "umap_min_dist": umap_min_dist,
            "umap_n_components": umap_n_components,
        }

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.scaler: Optional[StandardScaler] = None
        self.encoder: Optional[nn.Module] = None
        self.predictor: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.training_history: list[Dict[str, float]] = []
        self.training_embeddings_: Optional[ArrayLike] = None
        self.training_umap_: Optional[ArrayLike] = None
        self.training_labels_: Optional[ArrayLike] = None
        self.training_cluster_probabilities_: Optional[ArrayLike] = None
        self.umap_params: Dict[str, float | int] = {
            "n_neighbors": umap_n_neighbors,
            "min_dist": umap_min_dist,
            "n_components": umap_n_components,
            "metric": "euclidean",
        }
        self._scaled_data: Optional[ArrayLike] = None
        self._original_columns: Optional[Iterable[str]] = None
        self._umap_model: Optional[umap.UMAP] = None
        self._rng = np.random.default_rng(random_state)
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "TJEPA":
        """Train the JEPA encoder and downstream clustering pipeline."""

        _set_global_seed(self.random_state)
        self._rng = np.random.default_rng(self.random_state)
        numeric = _validate_dataframe(df)
        if numeric.shape[0] < 2:
            raise ValueError("At least two samples are required to train TJEPA.")

        self._original_columns = list(numeric.columns)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(numeric.values.astype(np.float32))
        self._scaled_data = scaled.astype(np.float32)

        tuned_params = self._maybe_tune(self._scaled_data)
        self.hparams.update(tuned_params)
        self.umap_params.update(
            {
                "n_neighbors": int(self.hparams["umap_n_neighbors"]),
                "min_dist": float(self.hparams["umap_min_dist"]),
                "n_components": int(self.hparams["umap_n_components"]),
            }
        )

        result = self._run_training(
            self._scaled_data,
            params=self.hparams,
            epochs=self.epochs,
            val_array=None,
        )
        self.encoder = result.encoder
        self.predictor = result.predictor
        self.decoder = result.decoder
        self.training_history = result.history

        self.training_embeddings_ = self._encode(self._scaled_data)
        (
            self._umap_model,
            self.training_umap_,
            self.training_labels_,
            self.training_cluster_probabilities_,
        ) = self._cluster_embeddings(self.training_embeddings_)

        self._fitted = True
        return self

    def transform(
        self,
        df: Optional[pd.DataFrame] = None,
        synthetic_multiplier: Optional[float] = None,
    ) -> pd.DataFrame:
        """Generate synthetic samples and pseudo labels for the provided data."""

        if not self._fitted:
            raise RuntimeError("fit must be called before transform.")
        if self.scaler is None or self.encoder is None or self.decoder is None:
            raise RuntimeError("Model components are not initialised correctly.")

        if df is None:
            if self._scaled_data is None or self._original_columns is None:
                raise RuntimeError("Training data was not cached.")
            numeric = pd.DataFrame(
                self.scaler.inverse_transform(self._scaled_data),
                columns=self._original_columns,
            )
            scaled = self._scaled_data
        else:
            numeric = _validate_dataframe(df)
            if self._original_columns is not None and list(numeric.columns) != list(
                self._original_columns
            ):
                warnings.warn(
                    "Columns differ from training data. Scaling may be inconsistent.",
                    RuntimeWarning,
                )
            scaled = self.scaler.transform(numeric.values.astype(np.float32))

        if synthetic_multiplier is None:
            synthetic_multiplier = self.synthetic_multiplier
        if synthetic_multiplier < 0:
            raise ValueError("synthetic_multiplier must be non-negative")

        n_original = scaled.shape[0]
        n_synth = int(round(n_original * synthetic_multiplier))
        synthetic_scaled, base_indices = self._generate_synthetic_scaled(n_synth)

        combined_scaled = (
            np.vstack([scaled, synthetic_scaled]) if n_synth > 0 else scaled
        )
        combined_embeddings = self._encode(combined_scaled)
        (
            self._umap_model,
            low_dim,
            labels,
            probabilities,
        ) = self._cluster_embeddings(combined_embeddings)

        combined_df = pd.DataFrame(
            self.scaler.inverse_transform(combined_scaled),
            columns=self._original_columns,
        )
        combined_df["is_synthetic"] = [False] * n_original + [True] * n_synth
        combined_df["pseudo_label"] = labels
        if probabilities is not None:
            padded = np.full(len(combined_df), np.nan, dtype=np.float32)
            padded[: probabilities.shape[0]] = probabilities
            combined_df["cluster_probability"] = padded
        if n_synth > 0:
            synthetic_sources = np.full(len(combined_df), np.nan, dtype=np.float32)
            synthetic_sources[n_original:] = base_indices
            combined_df["source_index"] = synthetic_sources

        for dim in range(low_dim.shape[1]):
            combined_df[f"umap_{dim}"] = low_dim[:, dim]

        return combined_df

    def fit_transform(
        self,
        df: pd.DataFrame,
        synthetic_multiplier: Optional[float] = None,
    ) -> pd.DataFrame:
        """Convenience wrapper around :meth:`fit` followed by :meth:`transform`."""

        return self.fit(df).transform(None, synthetic_multiplier=synthetic_multiplier)

    def _build_components(
        self, input_dim: int, params: Dict[str, float | int]
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        embedding_dim = int(params["embedding_dim"])
        encoder = _MLP(
            input_dim=input_dim,
            hidden_dim=int(params["hidden_dim"]),
            output_dim=embedding_dim,
            depth=int(params["encoder_depth"]),
            dropout=float(params["dropout"]),
        ).to(self.device)
        predictor = _MLP(
            input_dim=embedding_dim,
            hidden_dim=int(params["hidden_dim"]),
            output_dim=embedding_dim,
            depth=max(1, int(params["predictor_depth"])),
            dropout=float(params["dropout"]),
        ).to(self.device)
        decoder = _MLP(
            input_dim=embedding_dim,
            hidden_dim=int(params["hidden_dim"]),
            output_dim=input_dim,
            depth=int(params["decoder_depth"]),
            dropout=float(params["dropout"]),
        ).to(self.device)
        return encoder, predictor, decoder

    @staticmethod
    def _byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        p = F.normalize(p, dim=-1)
        z = F.normalize(z.detach(), dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    def _augment_batch(
        self, batch: torch.Tensor, params: Dict[str, float | int]
    ) -> torch.Tensor:
        noise_std = float(params["noise_std"])
        feature_dropout = float(params["feature_dropout"])
        noise = torch.randn_like(batch) * noise_std
        if feature_dropout > 0:
            mask = (torch.rand_like(batch) > feature_dropout).float()
        else:
            mask = torch.ones_like(batch)
        return (batch * mask) + noise

    def _compute_loss(
        self,
        batch: torch.Tensor,
        encoder: nn.Module,
        predictor: nn.Module,
        decoder: nn.Module,
        params: Dict[str, float | int],
        augment: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if augment:
            view_one = self._augment_batch(batch, params)
            view_two = self._augment_batch(batch, params)
        else:
            view_one = batch
            view_two = batch

        z_one = encoder(view_one)
        z_two = encoder(view_two)
        pred_one = predictor(z_one)
        pred_two = predictor(z_two)
        loss_jepa = self._byol_loss(pred_one, z_two) + self._byol_loss(pred_two, z_one)

        recon_weight = float(params["recon_weight"])
        if recon_weight > 0:
            recon_one = decoder(z_one)
            recon_two = decoder(z_two)
            loss_recon = F.mse_loss(recon_one, batch) + F.mse_loss(recon_two, batch)
        else:
            loss_recon = torch.tensor(0.0, device=batch.device)

        total_loss = loss_jepa + recon_weight * loss_recon
        return total_loss, loss_jepa.detach(), loss_recon.detach()

    def _run_training(
        self,
        array: ArrayLike,
        params: Dict[str, float | int],
        epochs: int,
        val_array: Optional[ArrayLike] = None,
    ) -> _TrainingResult:
        batch_size = max(1, min(int(params["batch_size"]), array.shape[0]))
        dataset = _TabularDataset(array)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        encoder, predictor, decoder = self._build_components(array.shape[1], params)
        optim_params = list(encoder.parameters())
        optim_params += list(predictor.parameters())
        optim_params += list(decoder.parameters())
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )

        history: list[Dict[str, float]] = []
        for epoch in range(max(1, int(epochs))):
            encoder.train()
            predictor.train()
            decoder.train()
            total_loss = 0.0
            total_jepa = 0.0
            total_recon = 0.0
            total_samples = 0
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss, loss_jepa, loss_recon = self._compute_loss(
                    batch,
                    encoder,
                    predictor,
                    decoder,
                    params,
                    augment=True,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(optim_params, max_norm=5.0)
                optimizer.step()

                batch_size_now = batch.shape[0]
                total_loss += float(loss.item()) * batch_size_now
                total_jepa += float(loss_jepa.item()) * batch_size_now
                total_recon += float(loss_recon.item()) * batch_size_now
                total_samples += batch_size_now

            if total_samples > 0:
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        "loss": total_loss / total_samples,
                        "jepa_loss": total_jepa / total_samples,
                        "recon_loss": total_recon / total_samples,
                    }
                )

        val_loss = None
        if val_array is not None and len(val_array) > 0:
            val_loss = self._evaluate_loss(val_array, encoder, predictor, decoder, params)

        encoder.eval()
        predictor.eval()
        decoder.eval()
        return _TrainingResult(encoder, predictor, decoder, history, val_loss)

    def _evaluate_loss(
        self,
        array: ArrayLike,
        encoder: nn.Module,
        predictor: nn.Module,
        decoder: nn.Module,
        params: Dict[str, float | int],
    ) -> float:
        dataset = _TabularDataset(array)
        batch_size = max(1, min(int(params["batch_size"]), len(dataset)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        total_loss = 0.0
        total_samples = 0
        encoder.eval()
        predictor.eval()
        decoder.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                loss, _, _ = self._compute_loss(
                    batch,
                    encoder,
                    predictor,
                    decoder,
                    params,
                    augment=False,
                )
                batch_size_now = batch.shape[0]
                total_loss += float(loss.item()) * batch_size_now
                total_samples += batch_size_now
        return total_loss / max(1, total_samples)

    def _train_and_score(
        self,
        train_array: ArrayLike,
        val_array: ArrayLike,
        params: Dict[str, float | int],
    ) -> Tuple[float, float]:
        if val_array.size == 0:
            raise ValueError("Validation data must be non-empty for tuning.")

        result = self._run_training(
            train_array,
            params=params,
            epochs=min(self.tune_epochs, self.epochs),
            val_array=val_array,
        )
        encoder = result.encoder
        predictor = result.predictor
        decoder = result.decoder
        val_loss = result.val_loss
        if val_loss is None:
            val_loss = self._evaluate_loss(val_array, encoder, predictor, decoder, params)

        with torch.no_grad():
            encoder.eval()
            tensor = torch.as_tensor(val_array, dtype=torch.float32, device=self.device)
            embeddings = encoder(tensor).cpu().numpy()

        reducer = umap.UMAP(
            n_neighbors=max(2, int(params["umap_n_neighbors"])),
            min_dist=float(params["umap_min_dist"]),
            n_components=int(params["umap_n_components"]),
            metric="euclidean",
            random_state=self.random_state,
        )
        low_dim = reducer.fit_transform(embeddings)
        labels = getattr(reducer, "labels_", None)
        if labels is None:
            clusters = find_clusters(
                reducer.graph_,
                min_cluster_size=self.min_cluster_size,
                cluster_selection_method=self.cluster_selection_method,
            )
            if isinstance(clusters, tuple):
                labels = clusters[0]
            else:
                labels = clusters
        labels_array = np.asarray(labels, dtype=int)
        mask = labels_array >= 0
        silhouette = -1.0
        if mask.sum() >= 2 and np.unique(labels_array[mask]).size >= 2:
            try:
                silhouette = float(silhouette_score(low_dim[mask], labels_array[mask]))
            except Exception:  # pragma: no cover - robustness for edge cases
                silhouette = -1.0

        for module in (encoder, predictor, decoder):
            module.cpu()

        return float(val_loss), silhouette

    def _maybe_tune(self, scaled: ArrayLike) -> Dict[str, float | int]:
        params = dict(self.hparams)
        if self.optuna_trials <= 0 or scaled.shape[0] < 4:
            return params

        val_size = max(1, int(round(scaled.shape[0] * self.tune_val_split)))
        if val_size >= scaled.shape[0]:
            val_size = scaled.shape[0] - 1
        if val_size <= 0:
            return params

        train_array, val_array = train_test_split(
            scaled,
            test_size=val_size,
            random_state=self.random_state,
            shuffle=True,
        )

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: optuna.trial.Trial) -> float:
            trial_params = dict(params)
            trial_params["embedding_dim"] = trial.suggest_int(
                "embedding_dim",
                32,
                max(32, min(512, scaled.shape[1] * 4)),
                step=32,
            )
            trial_params["hidden_dim"] = trial.suggest_int("hidden_dim", 64, 512, step=64)
            trial_params["encoder_depth"] = trial.suggest_int("encoder_depth", 2, 4)
            trial_params["predictor_depth"] = trial.suggest_int("predictor_depth", 1, 3)
            trial_params["decoder_depth"] = trial.suggest_int("decoder_depth", 2, 4)
            trial_params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
            trial_params["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            trial_params["weight_decay"] = trial.suggest_float(
                "weight_decay", 1e-6, 1e-2, log=True
            )
            trial_params["noise_std"] = trial.suggest_float("noise_std", 0.01, 0.3)
            trial_params["feature_dropout"] = trial.suggest_float(
                "feature_dropout", 0.0, 0.6
            )
            trial_params["recon_weight"] = trial.suggest_float(
                "recon_weight", 0.1, 5.0, log=True
            )
            trial_params["latent_noise"] = trial.suggest_float("latent_noise", 0.05, 0.8)
            trial_params["batch_size"] = trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256]
            )
            max_neighbors = max(5, min(60, train_array.shape[0] - 1))
            trial_params["umap_n_neighbors"] = trial.suggest_int(
                "umap_n_neighbors", 5, max_neighbors
            )
            trial_params["umap_min_dist"] = trial.suggest_float(
                "umap_min_dist", 0.0, 0.8
            )
            trial_params["umap_n_components"] = trial.suggest_int(
                "umap_n_components", 2, min(10, max(2, trial_params["embedding_dim"]))
            )

            val_loss, silhouette = self._train_and_score(train_array, val_array, trial_params)
            if math.isnan(val_loss) or math.isinf(val_loss):
                return float("inf")
            score = float(val_loss) - self.silhouette_weight * float(silhouette)
            return score

        study.optimize(
            objective,
            n_trials=self.optuna_trials,
            timeout=self.optuna_timeout,
            show_progress_bar=False,
        )

        if study.best_trial is not None:
            best_params = dict(params)
            best_params.update(study.best_trial.params)
            for key in (
                "batch_size",
                "embedding_dim",
                "hidden_dim",
                "encoder_depth",
                "predictor_depth",
                "decoder_depth",
                "umap_n_neighbors",
                "umap_n_components",
            ):
                if key in best_params:
                    best_params[key] = int(best_params[key])
            return best_params

        return params

    def _encode(self, array: ArrayLike) -> ArrayLike:
        if self.encoder is None:
            raise RuntimeError("Encoder has not been trained.")
        self.encoder.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
            embeddings = self.encoder(tensor).cpu().numpy()
        return embeddings

    def _generate_synthetic_scaled(
        self, n_samples: int
    ) -> Tuple[ArrayLike, ArrayLike]:
        if self._scaled_data is None:
            raise RuntimeError("Training data is not available for synthesis.")
        if n_samples <= 0:
            return (
                np.empty((0, self._scaled_data.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        indices = self._rng.integers(
            low=0,
            high=self._scaled_data.shape[0],
            size=n_samples,
            endpoint=False,
        )
        batch = torch.as_tensor(
            self._scaled_data[indices],
            dtype=torch.float32,
            device=self.device,
        )
        latent_noise_scale = float(self.hparams["latent_noise"]) * self.synthetic_temperature
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            latent = self.encoder(batch)
            noise = torch.randn_like(latent) * latent_noise_scale
            synthetic = self.decoder(latent + noise).cpu().numpy()
        return synthetic.astype(np.float32), indices.astype(np.int64)

    def _cluster_embeddings(
        self, embeddings: ArrayLike
    ) -> Tuple[umap.UMAP, ArrayLike, ArrayLike, Optional[ArrayLike]]:
        reducer = umap.UMAP(
            n_neighbors=max(2, int(self.umap_params.get("n_neighbors", 15))),
            min_dist=float(self.umap_params.get("min_dist", 0.1)),
            n_components=int(self.umap_params.get("n_components", 2)),
            metric=str(self.umap_params.get("metric", "euclidean")),
            random_state=self.random_state,
        )
        low_dim = reducer.fit_transform(embeddings)
        labels = getattr(reducer, "labels_", None)
        probabilities = getattr(reducer, "cluster_probabilities_", None)
        if labels is None:
            clusters = find_clusters(
                reducer.graph_,
                min_cluster_size=self.min_cluster_size,
                cluster_selection_method=self.cluster_selection_method,
            )
            if isinstance(clusters, tuple):
                labels = clusters[0]
                if probabilities is None and len(clusters) > 1:
                    probabilities = clusters[1]
            else:
                labels = clusters
        labels_array = np.asarray(labels, dtype=int)
        probabilities_array = (
            None
            if probabilities is None
            else np.asarray(probabilities, dtype=np.float32)
        )
        return reducer, low_dim, labels_array, probabilities_array


__all__ = ['TJEPA']
