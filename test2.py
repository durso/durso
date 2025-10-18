"""Gaussian Mixture Model (GMM) Tabular Variational Auto-Encoder.

This module provides a single-file implementation of a VAE tailored for
mixed-type tabular datasets.  It borrows several ideas from the
`t-jepa`-inspired augmenter shipped with this repository, but replaces the
self-supervised objective with a probabilistic decoder trained under a Gaussian
mixture prior.  The class exposed here aims to satisfy the following
requirements:

* operate directly on :class:`pandas.DataFrame` inputs;
* accept a schema that declares the distribution type of each feature and how
  categorical values are encoded (ordinal vs. one-hot);
* compute distribution-aware reconstruction losses that gracefully skip missing
  values on a per-feature basis;
* use Optuna for hyper-parameter optimisation and UMAP for downstream
  visualisation/pseudo-labelling; and
* synthesise new samples with pseudo labels obtained from the latent GMM.

Supported distributions include ``gaussian``, ``bernoulli``, ``categorical``,
``student_t`` (labelled ``t`` in the schema), ``poisson``, ``beta``,
``negative_binomial``, and ``pareto``.  When no schema is provided the class
assumes that all columns follow a Gaussian distribution and performs standard
scaling accordingly.

Example schema declaration::

    schema = [
        {"name": "age", "distribution": "gaussian", "min": 0, "max": 100},
        {"name": "income", "distribution": "lognormal"},  # will fall back to gaussian
        {
            "name": "color",
            "distribution": "categorical",
            "categories": ["red", "green", "blue"],
            "encoding": "ordinal",  # automatically expanded into one-hot internally
        },
        {
            "name": ["cat_A", "cat_B", "cat_C"],
            "distribution": "categorical",
            "encoding": "one_hot",
        },
        {"name": "clicked", "distribution": "bernoulli"},
    ]

The model converts ordinal categorical features into one-hot columns while
preserving the ability to reconstruct the original representation when
returning augmented dataframes.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - informative error
    raise ImportError(
        "gmm_tabular_vae requires PyTorch to be installed. Install it with `pip install torch`."
    ) from exc

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - informative error
    raise ImportError(
        "gmm_tabular_vae requires scikit-learn to be installed. Install it with `pip install scikit-learn`."
    ) from exc

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gmm_tabular_vae requires Optuna to be installed. Install it with `pip install optuna`."
    ) from exc

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gmm_tabular_vae requires umap-learn to be installed. Install it with `pip install umap-learn`."
    ) from exc

try:
    from scipy import special
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gmm_tabular_vae requires SciPy to be installed. Install it with `pip install scipy`."
    ) from exc

ArrayLike = np.ndarray
JSONLike = Union[str, Sequence[Dict[str, Any]]]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)


def _ensure_schema(schema: Optional[JSONLike], columns: Sequence[str]) -> List[Dict[str, Any]]:
    """Normalise the schema declaration.

    The schema can be provided as a Python object or a JSON string; if omitted,
    every column defaults to a Gaussian feature.
    """

    if schema is None:
        return [{"name": col, "distribution": "gaussian"} for col in columns]
    if isinstance(schema, str):
        parsed = json.loads(schema)
    else:
        parsed = list(schema)
    normalised: List[Dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            raise TypeError("Each schema entry must be a mapping.")
        if "name" not in entry:
            raise KeyError("Schema entries must define a 'name' attribute.")
        entry_copy = dict(entry)
        normalised.append(entry_copy)
    return normalised


@dataclass
class FeatureSpec:
    name: Union[str, Sequence[str]]
    distribution: str
    columns: List[str]
    decoder_params: int
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    categories: Optional[List[Any]] = None
    encoding: str = "numeric"
    scaler: Optional[StandardScaler] = None
    ordinal_map: Optional[Dict[Any, int]] = None
    inverse_ordinal: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        self.distribution = self.distribution.lower()


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class _MaskedDataset(Dataset):
    def __init__(self, data: ArrayLike, mask: ArrayLike):
        if data.shape != mask.shape:
            raise ValueError("Data and mask must share the same shape.")
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.mask[index]


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, hidden_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.network(x)


# ---------------------------------------------------------------------------
# Distribution-specific log-likelihoods
# ---------------------------------------------------------------------------

def _gaussian_nll(x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    var = torch.exp(log_var)
    log_prob = -0.5 * (math.log(2 * math.pi) + log_var + (x - mean) ** 2 / var)
    return -torch.sum(log_prob * mask) / torch.clamp(mask.sum(), min=1.0)


def _student_t_nll(
    x: torch.Tensor,
    loc: torch.Tensor,
    log_scale: torch.Tensor,
    log_df: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    scale = torch.exp(log_scale)
    df = torch.exp(log_df) + 2.0  # ensure > 2 for finite variance
    z = (x - loc) / scale
    log_prob = (
        torch.lgamma((df + 1) / 2)
        - torch.lgamma(df / 2)
        - 0.5 * torch.log(df * math.pi)
        - log_scale
        - ((df + 1) / 2) * torch.log1p(z ** 2 / df)
    )
    return -torch.sum(log_prob * mask) / torch.clamp(mask.sum(), min=1.0)


def _bernoulli_nll(target: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return torch.sum(loss * mask) / torch.clamp(mask.sum(), min=1.0)


def _categorical_nll(
    target: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    loss = -torch.sum(target * log_probs, dim=-1)
    weights = mask[:, 0]
    return torch.sum(loss * weights) / torch.clamp(weights.sum(), min=1.0)


def _poisson_nll(target: torch.Tensor, log_rate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rate = torch.exp(log_rate)
    loss = rate - target * log_rate + torch.lgamma(target + 1)
    return torch.sum(loss * mask) / torch.clamp(mask.sum(), min=1.0)


def _beta_nll(target: torch.Tensor, log_alpha: torch.Tensor, log_beta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    alpha = torch.exp(log_alpha) + 1e-4
    beta = torch.exp(log_beta) + 1e-4
    log_prob = (
        (alpha - 1) * torch.log(torch.clamp(target, min=1e-6, max=1 - 1e-6))
        + (beta - 1) * torch.log(torch.clamp(1 - target, min=1e-6, max=1 - 1e-6))
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
    )
    return -torch.sum(log_prob * mask) / torch.clamp(mask.sum(), min=1.0)


def _neg_binomial_nll(
    target: torch.Tensor,
    log_r: torch.Tensor,
    logit_p: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    r = torch.exp(log_r) + 1e-4
    p = torch.sigmoid(logit_p)
    log_prob = (
        torch.lgamma(target + r)
        - torch.lgamma(r)
        - torch.lgamma(target + 1)
        + r * torch.log1p(-p)
        + target * torch.log(torch.clamp(p, min=1e-6))
    )
    return -torch.sum(log_prob * mask) / torch.clamp(mask.sum(), min=1.0)


def _pareto_nll(target: torch.Tensor, log_scale: torch.Tensor, log_alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scale = torch.exp(log_scale) + 1e-6
    alpha = torch.exp(log_alpha) + 1e-6
    adjusted = torch.clamp(target, min=scale + 1e-6)
    log_prob = torch.log(alpha) + alpha * torch.log(scale) - (alpha + 1) * torch.log(adjusted)
    return -torch.sum(log_prob * mask) / torch.clamp(mask.sum(), min=1.0)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def _prepare_features(
    df: pd.DataFrame,
    schema: Optional[JSONLike],
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[FeatureSpec], ArrayLike, List[str]]:
    schema_entries = _ensure_schema(schema, df.columns)
    prepared = df.copy()
    feature_specs: List[FeatureSpec] = []
    masks: List[np.ndarray] = []
    ordered_columns: List[str] = []

    for entry in schema_entries:
        raw_name = entry["name"]
        distribution = str(entry.get("distribution", "gaussian")).lower()
        encoding = str(entry.get("encoding", "numeric")).lower()
        bounds = None
        if "min" in entry or "max" in entry:
            bounds = (entry.get("min"), entry.get("max"))
        if isinstance(raw_name, (list, tuple)):
            columns = [str(col) for col in raw_name]
        else:
            columns = [str(raw_name)]
        categories = entry.get("categories")
        if distribution == "categorical" and encoding == "ordinal" and categories is None:
            # derive categories from data
            cat_values = pd.Categorical(prepared[columns[0]])
            categories = [v for v in cat_values.categories]
        scaler: Optional[StandardScaler] = None
        ordinal_map: Optional[Dict[Any, int]] = None
        inverse_ordinal: Optional[List[Any]] = None
        effective_columns: List[str] = []
        column_masks: List[np.ndarray] = []

        if distribution == "categorical":
            if encoding == "one_hot":
                effective_columns = columns
                for col in columns:
                    if col not in prepared.columns:
                        raise KeyError(f"Unknown column '{col}' declared in schema.")
                    column_masks.append((~prepared[col].isna()).to_numpy().astype(np.float32))
            else:
                base_col = columns[0]
                series = prepared[base_col]
                if categories is None:
                    categories = sorted(series.dropna().unique().tolist())
                ordinal_map = {cat: idx for idx, cat in enumerate(categories)}
                inverse_ordinal = list(categories)
                encoded = pd.get_dummies(pd.Categorical(series, categories=categories))
                encoded = encoded.astype(np.float32)
                missing_mask = series.isna().to_numpy()
                one_hot_cols = []
                for cat in categories:
                    new_col = f"{base_col}__{cat}"
                    prepared[new_col] = encoded[cat]
                    one_hot_cols.append(new_col)
                    mask = (~missing_mask).astype(np.float32)
                    column_masks.append(mask)
                prepared.drop(columns=[base_col], inplace=True)
                effective_columns = one_hot_cols
        else:
            for col in columns:
                if col not in prepared.columns:
                    raise KeyError(f"Unknown column '{col}' declared in schema.")
            effective_columns = columns
            scaler = StandardScaler()
            values = prepared[columns].astype(np.float32)
            mask = ~values.isna()
            scaler.fit(values.fillna(0.0))
            prepared[columns] = scaler.transform(values.fillna(0.0))
            column_masks = [mask[col].to_numpy().astype(np.float32) for col in columns]
            for col in columns:
                prepared[col] = prepared[col].fillna(0.0)

        if not column_masks:
            for col in effective_columns:
                column_masks.append((~prepared[col].isna()).to_numpy().astype(np.float32))
                prepared[col] = prepared[col].fillna(0.0)

        spec = FeatureSpec(
            name=raw_name,
            distribution=distribution,
            columns=effective_columns,
            decoder_params=_decoder_param_count(distribution, len(effective_columns)),
            bounds=bounds,
            categories=categories,
            encoding=encoding if distribution == "categorical" else "numeric",
            scaler=scaler,
            ordinal_map=ordinal_map,
            inverse_ordinal=inverse_ordinal,
        )
        masks.extend(column_masks)
        feature_specs.append(spec)
        ordered_columns.extend(effective_columns)

    prepared = prepared[ordered_columns]
    mask_array = np.stack(masks, axis=1).astype(np.float32)
    return prepared, feature_specs, mask_array, ordered_columns


def _decoder_param_count(distribution: str, columns: int) -> int:
    distribution = distribution.lower()
    if distribution == "gaussian":
        return columns * 2
    if distribution in {"bernoulli", "poisson"}:
        return columns
    if distribution == "categorical":
        return columns
    if distribution in {"t", "student_t"}:
        return columns * 3
    if distribution == "beta":
        return columns * 2
    if distribution == "negative_binomial":
        return columns * 2
    if distribution == "pareto":
        return columns * 2
    # default back to Gaussian
    return columns * 2


# ---------------------------------------------------------------------------
# Latent GMM prior utilities
# ---------------------------------------------------------------------------

class _GMM(nn.Module):
    def __init__(self, latent_dim: int, components: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.components = components
        self.logits = nn.Parameter(torch.zeros(components))
        self.means = nn.Parameter(torch.randn(components, latent_dim) * 0.1)
        self.log_vars = nn.Parameter(torch.zeros(components, latent_dim))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(1)  # (batch, 1, dim)
        log_weights = F.log_softmax(self.logits, dim=0)
        log_probs = -0.5 * (
            (z - self.means) ** 2 / torch.exp(self.log_vars) + self.log_vars + math.log(2 * math.pi)
        ).sum(dim=-1)
        return torch.logsumexp(log_weights + log_probs, dim=1)

    def sample(self, n: int) -> torch.Tensor:
        categorical = torch.distributions.Categorical(logits=self.logits)
        idx = categorical.sample((n,))
        means = self.means[idx]
        std = torch.exp(0.5 * self.log_vars[idx])
        eps = torch.randn_like(means)
        return means + eps * std

    def responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(1)
        log_weights = F.log_softmax(self.logits, dim=0)
        log_probs = -0.5 * (
            (z - self.means) ** 2 / torch.exp(self.log_vars) + self.log_vars + math.log(2 * math.pi)
        ).sum(dim=-1)
        return F.softmax(log_weights + log_probs, dim=1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GMMTabularVAE:
    """Variational auto-encoder with a Gaussian mixture prior for tabular data."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        encoder_depth: int = 3,
        decoder_depth: int = 3,
        dropout: float = 0.1,
        latent_components: int = 5,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 128,
        epochs: int = 150,
        tune_trials: int = 20,
        tune_timeout: Optional[int] = None,
        tune_val_split: float = 0.2,
        tune_epochs: int = 30,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_n_components: int = 2,
        synthetic_multiplier: float = 1.0,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if latent_components < 1:
            raise ValueError("latent_components must be positive")
        if synthetic_multiplier < 0:
            raise ValueError("synthetic_multiplier must be non-negative")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.dropout = dropout
        self.latent_components = latent_components
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.tune_trials = tune_trials
        self.tune_timeout = tune_timeout
        self.tune_val_split = tune_val_split
        self.tune_epochs = tune_epochs
        self.umap_params = {
            "n_neighbors": umap_n_neighbors,
            "min_dist": umap_min_dist,
            "n_components": umap_n_components,
        }
        self.synthetic_multiplier = synthetic_multiplier
        self.random_state = random_state
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.feature_specs: List[FeatureSpec] = []
        self.mask_: Optional[ArrayLike] = None
        self.train_array_: Optional[ArrayLike] = None
        self.encoder: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.q_mu_layer: Optional[nn.Linear] = None
        self.q_logvar_layer: Optional[nn.Linear] = None
        self.gmm: Optional[_GMM] = None
        self.decoder_head: Optional[nn.Linear] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.schema_: Optional[List[Dict[str, Any]]] = None
        self.latent_history_: List[Dict[str, float]] = []
        self._fitted = False
        self._rng = np.random.default_rng(random_state)
        self.training_latent_: Optional[ArrayLike] = None
        self.training_umap_: Optional[ArrayLike] = None
        self.training_labels_: Optional[ArrayLike] = None
        self.training_responsibilities_: Optional[ArrayLike] = None
        self._column_order: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, schema: Optional[JSONLike] = None) -> "GMMTabularVAE":
        _set_global_seed(self.random_state)
        self._rng = np.random.default_rng(self.random_state)
        prepared, feature_specs, mask, column_order = _prepare_features(df, schema, self.random_state)
        array = prepared.to_numpy(dtype=np.float32)
        if array.shape[0] < 2:
            raise ValueError("At least two samples are required to fit the VAE.")
        self.feature_specs = feature_specs
        self.mask_ = mask.astype(np.float32)
        self.train_array_ = array
        self.schema_ = _ensure_schema(schema, df.columns)
        self._column_order = column_order

        tuned_params = self._maybe_tune(array, mask)
        self._apply_params(tuned_params)
        self._build_model(array.shape[1])
        self._train_model(array, mask, epochs=self.epochs)
        self.training_latent_ = self._encode(array, mask)
        (
            self.training_umap_,
            self.training_labels_,
            self.training_responsibilities_,
        ) = self._cluster_latent(self.training_latent_)
        self._fitted = True
        return self

    def transform(
        self,
        df: Optional[pd.DataFrame] = None,
        schema: Optional[JSONLike] = None,
        synthetic_multiplier: Optional[float] = None,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("fit must be called before transform.")
        if synthetic_multiplier is None:
            synthetic_multiplier = self.synthetic_multiplier
        if synthetic_multiplier < 0:
            raise ValueError("synthetic_multiplier must be non-negative")

        if df is None:
            if self.train_array_ is None or self.mask_ is None:
                raise RuntimeError("No cached training data available.")
            array = self.train_array_
            mask = self.mask_
            original_df = self._reconstruct_dataframe(array, mask, is_synthetic=False)
        else:
            array, mask = self._transform_with_existing_specs(df)
            original_df = self._reconstruct_dataframe(array, mask, is_synthetic=False)

        n_original = array.shape[0]
        n_synth = int(round(n_original * synthetic_multiplier))
        latent = self._encode(array, mask)
        combined_latent = latent
        if n_synth > 0:
            synthetic_df, synthetic_latent = self._generate_synthetic(n_synth)
            combined = pd.concat([original_df, synthetic_df], ignore_index=True)
            combined_latent = np.vstack([latent, synthetic_latent])
        else:
            synthetic_df = pd.DataFrame(columns=original_df.columns)
            combined = original_df.copy()

        umap_embeddings, combined_labels, responsibilities = self._cluster_latent(combined_latent)

        combined["pseudo_label"] = combined_labels
        combined["is_synthetic"] = [False] * n_original + [True] * n_synth
        if responsibilities is not None:
            combined["cluster_probability"] = responsibilities.max(axis=1)
        for dim in range(umap_embeddings.shape[1]):
            column = f"umap_{dim}"
            combined[column] = umap_embeddings[:, dim]
        return combined

    def fit_transform(
        self,
        df: pd.DataFrame,
        schema: Optional[JSONLike] = None,
        synthetic_multiplier: Optional[float] = None,
    ) -> pd.DataFrame:
        return self.fit(df, schema=schema).transform(None, synthetic_multiplier=synthetic_multiplier)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_params(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            setattr(self, key, value)

    def _build_model(self, input_dim: int) -> None:
        encoder_layers = _MLP(input_dim, self.hidden_dim, self.encoder_depth, self.dropout)
        decoder_layers = _MLP(self.latent_dim, self.hidden_dim, self.decoder_depth, self.dropout)
        self.encoder = encoder_layers.to(self.device)
        self.decoder = decoder_layers.to(self.device)
        self.q_mu_layer = nn.Linear(self.hidden_dim, self.latent_dim).to(self.device)
        self.q_logvar_layer = nn.Linear(self.hidden_dim, self.latent_dim).to(self.device)
        total_params = sum(spec.decoder_params for spec in self.feature_specs)
        self.decoder_head = nn.Linear(self.hidden_dim, total_params).to(self.device)
        self.gmm = _GMM(self.latent_dim, self.latent_components).to(self.device)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        params += list(self.q_mu_layer.parameters()) + list(self.q_logvar_layer.parameters())
        params += list(self.decoder_head.parameters()) + list(self.gmm.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _train_model(self, array: ArrayLike, mask: ArrayLike, epochs: int) -> None:
        dataset = _MaskedDataset(array, mask)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        self.latent_history_ = []
        for epoch in range(max(1, epochs)):
            epoch_loss = 0.0
            recon_loss_acc = 0.0
            kl_loss_acc = 0.0
            count = 0
            for batch, batch_mask in loader:
                batch = batch.to(self.device)
                batch_mask = batch_mask.to(self.device)
                recon_loss, kl_loss = self._loss(batch, batch_mask)
                loss = recon_loss + kl_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                params = [p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, 5.0)
                self.optimizer.step()

                batch_size_now = batch.shape[0]
                epoch_loss += float(loss.item()) * batch_size_now
                recon_loss_acc += float(recon_loss.item()) * batch_size_now
                kl_loss_acc += float(kl_loss.item()) * batch_size_now
                count += batch_size_now
            self.latent_history_.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": epoch_loss / max(count, 1),
                    "recon_loss": recon_loss_acc / max(count, 1),
                    "kl_loss": kl_loss_acc / max(count, 1),
                }
            )

    def _loss(self, batch: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(batch)
        mu = self.q_mu_layer(hidden)
        log_var = self.q_logvar_layer(hidden)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        decoded_hidden = self.decoder(z)
        params = self.decoder_head(decoded_hidden)
        recon_loss = self._decode_loss(batch, mask, params)

        if self.gmm is None:
            raise RuntimeError("GMM prior is not initialised.")
        log_q = -0.5 * (
            torch.sum(log_var, dim=1) + self.latent_dim * math.log(2 * math.pi) + torch.sum(((z - mu) ** 2) / torch.exp(log_var), dim=1)
        )
        log_p = self.gmm.log_prob(z)
        kl = torch.mean(log_q - log_p)
        return recon_loss, kl

    def _decode_loss(self, batch: torch.Tensor, mask: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        offset = 0
        total_loss = 0.0
        pointer = 0
        for spec in self.feature_specs:
            cols = len(spec.columns)
            mask_slice = mask[:, pointer : pointer + cols]
            target_slice = batch[:, pointer : pointer + cols]
            pointer += cols
            param_slice = params[:, offset : offset + spec.decoder_params]
            offset += spec.decoder_params
            if spec.distribution == "gaussian":
                mean = param_slice[:, :cols]
                log_var = param_slice[:, cols: 2 * cols]
                total_loss = total_loss + _gaussian_nll(target_slice, mean, log_var, mask_slice)
            elif spec.distribution in {"bernoulli"}:
                total_loss = total_loss + _bernoulli_nll(target_slice, param_slice[:, :cols], mask_slice)
            elif spec.distribution == "categorical":
                logits = param_slice.view(param_slice.shape[0], cols)
                total_loss = total_loss + _categorical_nll(target_slice, logits, mask_slice)
            elif spec.distribution in {"t", "student_t"}:
                loc = param_slice[:, :cols]
                log_scale = param_slice[:, cols : 2 * cols]
                log_df = param_slice[:, 2 * cols : 3 * cols]
                total_loss = total_loss + _student_t_nll(target_slice, loc, log_scale, log_df, mask_slice)
            elif spec.distribution == "poisson":
                total_loss = total_loss + _poisson_nll(target_slice, param_slice[:, :cols], mask_slice)
            elif spec.distribution == "beta":
                log_alpha = param_slice[:, :cols]
                log_beta = param_slice[:, cols : 2 * cols]
                total_loss = total_loss + _beta_nll(target_slice, log_alpha, log_beta, mask_slice)
            elif spec.distribution == "negative_binomial":
                log_r = param_slice[:, :cols]
                logit_p = param_slice[:, cols : 2 * cols]
                total_loss = total_loss + _neg_binomial_nll(target_slice, log_r, logit_p, mask_slice)
            elif spec.distribution == "pareto":
                log_scale = param_slice[:, :cols]
                log_alpha = param_slice[:, cols : 2 * cols]
                total_loss = total_loss + _pareto_nll(target_slice, log_scale, log_alpha, mask_slice)
            else:
                mean = param_slice[:, :cols]
                log_var = param_slice[:, cols: 2 * cols]
                total_loss = total_loss + _gaussian_nll(target_slice, mean, log_var, mask_slice)
        return total_loss

    def _encode(self, array: ArrayLike, mask: ArrayLike) -> ArrayLike:
        dataset = _MaskedDataset(array, mask)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=False)
        latents: List[np.ndarray] = []
        self.encoder.eval()
        with torch.no_grad():
            for batch, _ in loader:
                batch = batch.to(self.device)
                hidden = self.encoder(batch)
                mu = self.q_mu_layer(hidden)
                latents.append(mu.cpu().numpy())
        self.encoder.train()
        return np.concatenate(latents, axis=0)

    def _cluster_latent(self, latent: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        reducer = umap.UMAP(
            n_neighbors=max(2, int(self.umap_params.get("n_neighbors", 15))),
            min_dist=float(self.umap_params.get("min_dist", 0.1)),
            n_components=int(self.umap_params.get("n_components", 2)),
            random_state=self.random_state,
        )
        embedding = reducer.fit_transform(latent)
        if self.gmm is None:
            raise RuntimeError("GMM prior is missing.")
        responsibilities = self.gmm.responsibilities(torch.as_tensor(latent, dtype=torch.float32, device=self.device)).cpu().numpy()
        labels = responsibilities.argmax(axis=1)
        return embedding, labels, responsibilities

    def _maybe_tune(self, array: ArrayLike, mask: ArrayLike) -> Dict[str, Any]:
        if self.tune_trials <= 0 or array.shape[0] < 4:
            return {}
        val_size = max(1, int(round(array.shape[0] * self.tune_val_split)))
        if val_size >= array.shape[0]:
            val_size = array.shape[0] - 1
        if val_size <= 0:
            return {}
        data = np.hstack([array, mask])
        train, val = train_test_split(data, test_size=val_size, random_state=self.random_state, shuffle=True)
        train_array, train_mask = train[:, : array.shape[1]], train[:, array.shape[1]:]
        val_array, val_mask = val[:, : array.shape[1]], val[:, array.shape[1]:]

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: optuna.trial.Trial) -> float:
            params = {
                "latent_dim": trial.suggest_int("latent_dim", 16, 64, step=8),
                "hidden_dim": trial.suggest_int("hidden_dim", 128, 512, step=64),
                "encoder_depth": trial.suggest_int("encoder_depth", 2, 4),
                "decoder_depth": trial.suggest_int("decoder_depth", 2, 4),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                "latent_components": trial.suggest_int("latent_components", 2, 12),
                "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            }
            self._apply_params(params)
            self._build_model(train_array.shape[1])
            self._train_model(train_array, train_mask, epochs=min(self.tune_epochs, self.epochs))
            latent = self._encode(val_array, val_mask)
            recon, kl = self._evaluate(val_array, val_mask)
            score = recon + kl
            return float(score)

        study.optimize(objective, n_trials=self.tune_trials, timeout=self.tune_timeout, show_progress_bar=False)
        best = study.best_trial.params if study.best_trial is not None else {}
        return best

    def _evaluate(self, array: ArrayLike, mask: ArrayLike) -> Tuple[float, float]:
        dataset = _MaskedDataset(array, mask)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=False)
        recon_total = 0.0
        kl_total = 0.0
        count = 0
        self.encoder.eval()
        with torch.no_grad():
            for batch, batch_mask in loader:
                batch = batch.to(self.device)
                batch_mask = batch_mask.to(self.device)
                hidden = self.encoder(batch)
                mu = self.q_mu_layer(hidden)
                log_var = self.q_logvar_layer(hidden)
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = mu + eps * std
                decoded = self.decoder(z)
                params = self.decoder_head(decoded)
                recon = self._decode_loss(batch, batch_mask, params)
                log_q = -0.5 * (
                    torch.sum(log_var, dim=1) + self.latent_dim * math.log(2 * math.pi) + torch.sum(((z - mu) ** 2) / torch.exp(log_var), dim=1)
                )
                log_p = self.gmm.log_prob(z)
                kl = torch.mean(log_q - log_p)
                bsize = batch.shape[0]
                recon_total += float(recon.item()) * bsize
                kl_total += float(kl.item()) * bsize
                count += bsize
        self.encoder.train()
        return recon_total / max(count, 1), kl_total / max(count, 1)

    def _transform_with_existing_specs(self, df: pd.DataFrame) -> Tuple[ArrayLike, ArrayLike]:
        if not self.feature_specs:
            raise RuntimeError("Model has not been fitted.")
        if len(df) == 0:
            return np.empty((0, len(self._column_order)), dtype=np.float32), np.empty((0, len(self._column_order)), dtype=np.float32)

        column_values: List[np.ndarray] = []
        column_masks: List[np.ndarray] = []
        column_names: List[str] = []

        for spec in self.feature_specs:
            if spec.distribution == "categorical" and spec.encoding != "one_hot":
                base_col = str(spec.columns[0]).split("__", 1)[0]
                if base_col not in df.columns:
                    raise KeyError(f"Expected categorical column '{base_col}' not present in input data.")
                series = df[base_col]
                categories = spec.inverse_ordinal or []
                cat_series = pd.Categorical(series, categories=categories)
                encoded = pd.get_dummies(cat_series)
                encoded = encoded.reindex(columns=categories, fill_value=0.0).astype(np.float32)
                missing_mask = series.isna().to_numpy()
                for col_name, cat_name in zip(spec.columns, categories):
                    values = encoded[cat_name].to_numpy(dtype=np.float32)
                    mask = (~missing_mask).astype(np.float32)
                    column_values.append(values)
                    column_masks.append(mask)
                    column_names.append(col_name)
            elif spec.distribution == "categorical":
                for col in spec.columns:
                    if col not in df.columns:
                        raise KeyError(f"Expected one-hot column '{col}' not present in input data.")
                    series = df[col]
                    mask = (~series.isna()).to_numpy().astype(np.float32)
                    values = series.fillna(0.0).to_numpy(dtype=np.float32)
                    column_values.append(values)
                    column_masks.append(mask)
                    column_names.append(col)
            else:
                missing_cols = [col for col in spec.columns if col not in df.columns]
                if missing_cols:
                    raise KeyError(f"Missing columns {missing_cols} for feature '{spec.name}'.")
                values = df[spec.columns].astype(np.float32)
                mask_df = (~values.isna()).astype(np.float32)
                filled = values.fillna(0.0)
                if spec.scaler is not None:
                    transformed = spec.scaler.transform(filled)
                else:
                    transformed = filled.to_numpy(dtype=np.float32)
                for idx, col in enumerate(spec.columns):
                    column_values.append(transformed[:, idx])
                    column_masks.append(mask_df.iloc[:, idx].to_numpy(dtype=np.float32))
                    column_names.append(col)

        data_matrix = np.stack(column_values, axis=1).astype(np.float32)
        mask_matrix = np.stack(column_masks, axis=1).astype(np.float32)
        if self._column_order:
            name_to_idx = {name: idx for idx, name in enumerate(column_names)}
            try:
                index = [name_to_idx[name] for name in self._column_order]
            except KeyError as exc:  # pragma: no cover - defensive
                missing = [name for name in self._column_order if name not in name_to_idx]
                raise KeyError(f"Input data is missing expected columns: {missing}") from exc
            data_matrix = data_matrix[:, index]
            mask_matrix = mask_matrix[:, index]
        return data_matrix, mask_matrix

    # ------------------------------------------------------------------
    # Data reconstruction utilities
    # ------------------------------------------------------------------

    def _reconstruct_dataframe(
        self,
        array: ArrayLike,
        mask: ArrayLike,
        is_synthetic: bool,
    ) -> pd.DataFrame:
        columns: List[str] = []
        pointer = 0
        reconstructed: Dict[str, np.ndarray] = {}
        for spec in self.feature_specs:
            cols = len(spec.columns)
            data_slice = array[:, pointer : pointer + cols]
            mask_slice = mask[:, pointer : pointer + cols]
            pointer += cols
            if spec.distribution == "categorical" and spec.encoding != "one_hot":
                probs = data_slice
                indices = probs.argmax(axis=1)
                base_name = spec.columns[0].split("__", 1)[0]
                values = np.array(
                    [spec.inverse_ordinal[idx] if idx < len(spec.inverse_ordinal) else np.nan for idx in indices],
                    dtype=object,
                )
                observed = mask_slice.sum(axis=1) > 0.5
                result = np.empty(len(values), dtype=object)
                result[:] = np.nan
                result[observed] = values[observed]
                reconstructed[base_name] = result
                columns.append(base_name)
            else:
                for idx, col in enumerate(spec.columns):
                    values = data_slice[:, idx]
                    mask_vals = mask_slice[:, idx]
                    if spec.scaler is not None:
                        reshaped = values.reshape(-1, 1)
                        restored = spec.scaler.inverse_transform(reshaped)[:, 0]
                    else:
                        restored = values
                    result = np.where(mask_vals > 0.5, restored, np.nan)
                    original_name = col if spec.distribution != "categorical" else col
                    reconstructed[original_name] = result
                    columns.append(original_name)
        df = pd.DataFrame(reconstructed, columns=columns)
        df["is_synthetic"] = is_synthetic
        return df

    def _generate_synthetic(self, n_samples: int) -> Tuple[pd.DataFrame, ArrayLike]:
        if self.gmm is None:
            raise RuntimeError("Model has not been fitted.")
        with torch.no_grad():
            latent = self.gmm.sample(n_samples).to(self.device)
            hidden = self.decoder(latent)
            params = self.decoder_head(hidden)
        array, mask = self._sample_from_decoder(params.cpu(), n_samples)
        df = self._reconstruct_dataframe(array, mask, is_synthetic=True)
        latent_np = latent.cpu().numpy()
        return df, latent_np

    def _sample_from_decoder(self, params: torch.Tensor, n_samples: int) -> Tuple[ArrayLike, ArrayLike]:
        pointer = 0
        columns: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        rng = self._rng or np.random.default_rng()
        for spec in self.feature_specs:
            cols = len(spec.columns)
            param_slice = params[:, pointer : pointer + spec.decoder_params]
            pointer += spec.decoder_params
            if spec.distribution == "gaussian":
                mean = param_slice[:, :cols].numpy()
                log_var = param_slice[:, cols : 2 * cols].numpy()
                std = np.exp(0.5 * log_var)
                samples = mean + std * rng.standard_normal(size=mean.shape)
                if spec.bounds is not None:
                    low, high = spec.bounds
                    if low is not None:
                        samples = np.maximum(samples, low)
                    if high is not None:
                        samples = np.minimum(samples, high)
                columns.append(samples)
                masks.append(np.ones_like(samples))
            elif spec.distribution == "bernoulli":
                logits = param_slice[:, :cols].numpy()
                probs = special.expit(logits)
                samples = (rng.random(size=probs.shape) < probs).astype(np.float32)
                columns.append(samples)
                masks.append(np.ones_like(samples))
            elif spec.distribution == "categorical":
                logits = param_slice.numpy()
                probs = special.softmax(logits, axis=1)
                cumulative = np.cumsum(probs, axis=1)
                random_vals = rng.random(size=(n_samples, 1))
                indices = (cumulative > random_vals).argmax(axis=1)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(n_samples), indices] = 1.0
                columns.append(one_hot)
                masks.append(np.ones_like(one_hot))
            elif spec.distribution in {"t", "student_t"}:
                loc = param_slice[:, :cols].numpy()
                scale = np.exp(param_slice[:, cols : 2 * cols].numpy())
                df = np.exp(param_slice[:, 2 * cols : 3 * cols].numpy()) + 2.0
                standard = rng.standard_normal(size=loc.shape)
                gamma = rng.gamma(shape=df / 2.0, scale=2.0 / df)
                samples = loc + scale * standard / np.sqrt(gamma)
                columns.append(samples)
                masks.append(np.ones_like(samples))
            elif spec.distribution == "poisson":
                rate = np.exp(param_slice[:, :cols].numpy())
                samples = rng.poisson(rate)
                columns.append(samples.astype(np.float32))
                masks.append(np.ones_like(samples))
            elif spec.distribution == "beta":
                alpha = np.exp(param_slice[:, :cols].numpy()) + 1e-4
                beta = np.exp(param_slice[:, cols : 2 * cols].numpy()) + 1e-4
                samples = rng.beta(alpha, beta)
                columns.append(samples.astype(np.float32))
                masks.append(np.ones_like(samples))
            elif spec.distribution == "negative_binomial":
                r = np.exp(param_slice[:, :cols].numpy()) + 1e-4
                p = special.expit(param_slice[:, cols : 2 * cols].numpy())
                samples = rng.negative_binomial(r, 1 - p)
                columns.append(samples.astype(np.float32))
                masks.append(np.ones_like(samples))
            elif spec.distribution == "pareto":
                scale = np.exp(param_slice[:, :cols].numpy()) + 1e-6
                alpha = np.exp(param_slice[:, cols : 2 * cols].numpy()) + 1e-6
                samples = scale * (1 - rng.random(size=scale.shape)) ** (-1 / alpha)
                columns.append(samples.astype(np.float32))
                masks.append(np.ones_like(samples))
            else:
                mean = param_slice[:, :cols].numpy()
                log_var = param_slice[:, cols : 2 * cols].numpy()
                std = np.exp(0.5 * log_var)
                samples = mean + std * rng.standard_normal(size=mean.shape)
                columns.append(samples)
                masks.append(np.ones_like(samples))
        array = np.concatenate(columns, axis=1).astype(np.float32)
        mask = np.concatenate(masks, axis=1).astype(np.float32)
        return array, mask
