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
* synthetic sample generation by decoding perturbed latent representations,
  optionally guided by feature-wise distributional schemas that adjust both the
  reconstruction loss and the sampling process.

In addition, the implementation incorporates the regularization token described
in the original T-JEPA paper. The learnable token is optimised jointly with the
encoder and predictor to stabilise the joint-embedding objective and can be
scaled through the ``regularization_weight`` hyper-parameter.

Typical usage::

    >>> import pandas as pd
    >>> from t_jepa import TJEPA
    >>> df = pd.DataFrame({"x": [0.0, 0.1, 0.2, 1.8, 2.0, 2.1],
    ...                    "y": [0.1, 0.0, 0.2, 1.9, 2.2, 2.0]})
    >>> jepa = TJEPA(random_state=0,
    ...              feature_schema={"x": {"type": "gaussian"},
    ...                               "y": {"type": "beta"}})
    >>> augmented = jepa.fit_transform(df)
    >>> augmented[["x", "y", "pseudo_label", "is_synthetic"]].head()

The resulting dataframe combines the original and synthetic samples, annotated
with pseudo labels derived from UMAP clustering.

A categorical schema can be expressed with either integer-coded or one-hot
encoded columns.  For example::

    >>> schema = {
    ...     "city": {"type": "categorical", "encoding": "ordinal", "num_classes": 4},
    ...     "plan": {
    ...         "type": "categorical",
    ...         "encoding": "one_hot",
    ...         "columns": ["plan_basic", "plan_plus", "plan_premium"],
    ...     },
    ... }

When the grouped ``columns`` entry is supplied the model treats the referenced
one-hot columns as a single categorical feature during training, loss
computation, and synthetic sampling.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
except ImportError as exc:  # pragma: no cover - informative error for users
    raise ImportError(
        "t_jepa requires umap-learn to be installed. Install it with `pip install umap-learn`."
    ) from exc

try:  # pragma: no cover - depending on umap version this may fail
    from umap.umap_ import find_clusters  # type: ignore[attr-defined]
    _USING_FALLBACK_FIND_CLUSTERS = False
except Exception:  # pragma: no cover - provide a functional fallback
    _USING_FALLBACK_FIND_CLUSTERS = True
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:  # pragma: no cover - informative error for users
        raise ImportError(
            "The bundled fallback for `find_clusters` requires SciPy. Install it with `pip install scipy`."
        ) from exc

    try:  # pragma: no cover - optional, improves clustering quality if available
        import hdbscan  # type: ignore
    except ImportError:  # pragma: no cover - gracefully degrade without hdbscan
        hdbscan = None

    def find_clusters(
        graph,
        min_cluster_size: int = 10,
        cluster_selection_method: str = "eom",
        embedding: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback approximation to :func:`umap.umap_.find_clusters`.

        The helper disappeared from newer releases of UMAP, so this replacement
        attempts progressively richer clustering strategies before finally
        falling back to simple connected components.  When a low-dimensional
        embedding or the original feature array is supplied, HDBSCAN or DBSCAN
        is applied to recover multi-cluster structure.
        """

        feature_array: Optional[np.ndarray] = None
        if embedding is not None:
            feature_array = np.asarray(embedding, dtype=np.float32)
        elif data is not None:
            feature_array = np.asarray(data, dtype=np.float32)

        if (
            hdbscan is not None
            and feature_array is not None
            and feature_array.shape[0] >= 2
        ):
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, int(min_cluster_size)),
                cluster_selection_method=cluster_selection_method,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(feature_array)
            probs = getattr(clusterer, "probabilities_", None)
            probabilities = (
                np.asarray(probs, dtype=np.float32)
                if probs is not None
                else np.ones(labels.shape[0], dtype=np.float32)
            )
            return np.asarray(labels, dtype=int), probabilities

        if feature_array is not None and feature_array.shape[0] >= 2:
            try:
                from sklearn.cluster import DBSCAN
                from sklearn.neighbors import NearestNeighbors
            except ImportError as exc:  # pragma: no cover - propagate informative error
                raise ImportError(
                    "The fallback clustering path requires scikit-learn's clustering utilities."
                ) from exc

            n_neighbors = min(
                max(2, int(min_cluster_size)), feature_array.shape[0] - 1
            )
            if n_neighbors >= 1:
                nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1)
                nbrs.fit(feature_array)
                distances = nbrs.kneighbors(feature_array)[0][:, 1:]
                kth = np.sort(distances[:, -1])
                percentiles = [30, 50, 70, 90]
                candidate_eps = []
                for perc in percentiles:
                    value = np.percentile(kth, perc)
                    if np.isfinite(value) and value > 0:
                        candidate_eps.append(float(value))
                best_labels: Optional[np.ndarray] = None
                best_cluster_count = -1
                for eps in candidate_eps:
                    clustering = DBSCAN(
                        eps=float(eps),
                        min_samples=max(2, int(min_cluster_size)),
                    ).fit(feature_array)
                    labels = clustering.labels_
                    cluster_ids = set(labels.tolist())
                    cluster_count = len(cluster_ids - {-1})
                    if cluster_count > best_cluster_count:
                        best_cluster_count = cluster_count
                        best_labels = labels
                    if cluster_count > 1:
                        break
                if best_labels is not None:
                    labels = np.asarray(best_labels, dtype=int)
                    probabilities = np.ones(labels.shape[0], dtype=np.float32)
                    return labels, probabilities

        if graph is None:
            raise ValueError("graph must be provided for clustering.")
        if not isinstance(graph, csr_matrix):
            graph = csr_matrix(graph)
        sym_graph = graph.maximum(graph.T)
        _, labels = connected_components(sym_graph, directed=False)
        labels = np.asarray(labels, dtype=int)
        min_size = max(1, int(min_cluster_size))
        counts = np.bincount(labels, minlength=labels.max() + 1)
        for component, size in enumerate(counts):
            if size < min_size:
                labels[labels == component] = -1
        probabilities = np.ones(labels.shape[0], dtype=np.float32)
        return labels, probabilities


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
    """Simple :class:`torch.utils.data.Dataset` wrapper over paired arrays."""

    def __init__(self, scaled: ArrayLike, original: ArrayLike, mask: ArrayLike):
        self.scaled = torch.as_tensor(scaled, dtype=torch.float32)
        self.original = torch.as_tensor(original, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)
        if not (self.scaled.shape == self.original.shape == self.mask.shape):
            raise ValueError("Scaled, original, and mask arrays must share the same shape.")

    def __len__(self) -> int:
        return int(self.scaled.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.scaled[index], self.original[index], self.mask[index]


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
    reg_token: torch.Tensor
    history: list[Dict[str, float]]
    val_loss: Optional[float]


@dataclass
class _DistributionSpec:
    name: str
    param_count: int
    slc: slice
    column_indices: Tuple[int, ...]
    column_names: Tuple[str, ...]
    num_classes: Optional[int] = None
    df: Optional[float] = None
    xm: Optional[float] = None
    encoding: str = "scalar"


class TJEPA:
    """Neural self-supervised augmenter inspired by the original T-JEPA project."""

    _SUPPORTED_DISTRIBUTIONS = {
        "gaussian",
        "bernoulli",
        "categorical",
        "t",
        "poisson",
        "beta",
        "negative_binomial",
        "pareto",
    }

    _CONTINUOUS_SCALABLE = {"gaussian", "t", "pareto"}

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
        regularization_weight: float = 1.0,
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
        feature_schema: Optional[Dict[str, Dict[str, Any]]] = None,
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
        self.feature_schema_input = feature_schema or {}

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
            "regularization_weight": regularization_weight,
            "umap_n_neighbors": umap_n_neighbors,
            "umap_min_dist": umap_min_dist,
            "umap_n_components": umap_n_components,
        }

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.encoder: Optional[nn.Module] = None
        self.predictor: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.reg_token: Optional[torch.Tensor] = None
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
        self._original_columns: Optional[list[str]] = None
        self._scaled_data: Optional[ArrayLike] = None
        self._original_data: Optional[ArrayLike] = None
        self._observed_mask: Optional[ArrayLike] = None
        self._umap_model: Optional[umap.UMAP] = None
        self._rng = np.random.default_rng(random_state)
        self._fitted = False
        self._column_specs: Dict[str, _DistributionSpec] = {}
        self._column_scalers: Dict[str, Optional[StandardScaler]] = {}
        self._total_param_dim: Optional[int] = None
        self._feature_specs: list[_DistributionSpec] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "TJEPA":
        """Train the JEPA encoder and downstream clustering pipeline."""

        _set_global_seed(self.random_state)
        self._rng = np.random.default_rng(self.random_state)
        self.reg_token = None
        numeric = _validate_dataframe(df)
        if numeric.shape[0] < 2:
            raise ValueError("At least two samples are required to train TJEPA.")

        self._original_columns = list(numeric.columns)
        self._prepare_schema(numeric)
        scaled = self._scale_frame(numeric, fit=True)
        self._scaled_data = scaled.astype(np.float32)
        self._original_data = numeric.values.astype(np.float32)
        self._observed_mask = np.isfinite(self._original_data).astype(np.float32)

        tuned_params = self._maybe_tune(
            self._scaled_data, self._original_data, self._observed_mask
        )
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
            self._original_data,
            self._observed_mask,
            params=self.hparams,
            epochs=self.epochs,
            val_scaled=None,
            val_original=None,
            val_mask=None,
        )
        self.encoder = result.encoder
        self.predictor = result.predictor
        self.decoder = result.decoder
        self.reg_token = result.reg_token
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
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Model components are not initialised correctly.")
        if self._original_columns is None:
            raise RuntimeError("Training column metadata is missing.")

        if df is None:
            if self._original_data is None or self._scaled_data is None:
                raise RuntimeError("Training data was not cached.")
            numeric = pd.DataFrame(
                self._original_data,
                columns=self._original_columns,
            )
            scaled = self._scaled_data
        else:
            numeric = _validate_dataframe(df)
            if self._original_columns is not None:
                unexpected = [col for col in numeric.columns if col not in self._original_columns]
                if unexpected:
                    raise ValueError(
                        "Input dataframe includes unexpected columns: "
                        + ", ".join(unexpected)
                    )
                missing = [col for col in self._original_columns if col not in numeric.columns]
                if missing:
                    raise ValueError(
                        "Input dataframe is missing expected columns: "
                        + ", ".join(missing)
                    )
                numeric = numeric[self._original_columns]
            self._ensure_schema_consistency(numeric)
            scaled = self._scale_frame(numeric)

        if synthetic_multiplier is None:
            synthetic_multiplier = self.synthetic_multiplier
        if synthetic_multiplier < 0:
            raise ValueError("synthetic_multiplier must be non-negative")

        n_original = scaled.shape[0]
        n_synth = int(round(n_original * synthetic_multiplier))
        synthetic_original, base_indices = self._generate_synthetic_original(n_synth)
        synthetic_scaled = (
            self._scale_array(synthetic_original)
            if n_synth > 0
            else np.empty((0, scaled.shape[1]), dtype=np.float32)
        )

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

        combined_original = (
            np.vstack([numeric.values, synthetic_original])
            if n_synth > 0
            else numeric.values
        )
        combined_df = pd.DataFrame(combined_original, columns=self._original_columns)
        combined_df = self._postprocess_dataframe(combined_df)
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_schema(self, df: pd.DataFrame) -> None:
        overrides: Dict[str, Dict[str, Any]] = {}
        group_configs: Dict[str, Dict[str, Any]] = {}
        for key, raw_cfg in self.feature_schema_input.items():
            cfg = raw_cfg.copy()
            if "columns" in cfg:
                columns = list(cfg.get("columns", []))
                if not columns:
                    raise ValueError(
                        f"Grouped schema entry '{key}' must include a non-empty 'columns' list."
                    )
                group_configs[key] = {"columns": columns, "config": cfg}
            else:
                overrides[key] = cfg

        column_to_group: Dict[str, str] = {}
        for group_name, info in group_configs.items():
            for column in info["columns"]:
                if column not in df.columns:
                    raise ValueError(
                        f"Grouped schema '{group_name}' references unknown column '{column}'."
                    )
                if column in column_to_group:
                    raise ValueError(
                        f"Column '{column}' is assigned to multiple grouped schema entries."
                    )
                if column in overrides:
                    raise ValueError(
                        f"Column '{column}' cannot have both a grouped schema and an individual schema."
                    )
                column_to_group[column] = group_name

        total_params = 0
        schema: Dict[str, _DistributionSpec] = {}
        scalers: Dict[str, Optional[StandardScaler]] = {}
        feature_specs: list[_DistributionSpec] = []
        processed_groups: set[str] = set()

        for idx, col in enumerate(df.columns):
            if col in column_to_group:
                group_name = column_to_group[col]
                if group_name in processed_groups:
                    continue
                info = group_configs[group_name]
                cfg = info["config"]
                dist_name = str(cfg.get("type", "categorical")).lower()
                if dist_name != "categorical":
                    raise ValueError(
                        f"Grouped schema '{group_name}' must use type='categorical'."
                    )
                encoding = str(cfg.get("encoding", "one_hot")).lower()
                if encoding not in {"one_hot", "onehot"}:
                    raise ValueError(
                        f"Grouped categorical schema '{group_name}' must declare encoding='one_hot'."
                    )
                columns = list(info["columns"])
                indices = tuple(df.columns.get_loc(c) for c in columns)
                num_classes = cfg.get("num_classes", len(columns))
                if not isinstance(num_classes, int) or num_classes < 2:
                    raise ValueError(
                        f"Grouped categorical schema '{group_name}' requires an integer 'num_classes' >= 2."
                    )
                if num_classes != len(columns):
                    raise ValueError(
                        f"Grouped categorical schema '{group_name}' expects 'num_classes' == len(columns) ({len(columns)})."
                    )
                param_count = num_classes
                spec = _DistributionSpec(
                    name="categorical",
                    param_count=param_count,
                    slc=slice(total_params, total_params + param_count),
                    column_indices=indices,
                    column_names=tuple(columns),
                    num_classes=param_count,
                    encoding="one_hot",
                )
                feature_specs.append(spec)
                total_params += param_count
                for column in columns:
                    schema[column] = spec
                    scalers[column] = None
                processed_groups.add(group_name)
                continue

            cfg = overrides.get(col, {})
            dist_name = str(cfg.get("type", "gaussian")).lower()
            if dist_name not in self._SUPPORTED_DISTRIBUTIONS:
                raise ValueError(
                    f"Unsupported distribution '{dist_name}' for column '{col}'."
                )

            param_count: int
            num_classes: Optional[int] = None
            encoding = "scalar"
            if dist_name == "gaussian":
                param_count = 2
            elif dist_name == "bernoulli":
                param_count = 1
            elif dist_name == "categorical":
                encoding = str(cfg.get("encoding", "ordinal")).lower()
                if encoding not in {"ordinal", "index", "label"}:
                    raise ValueError(
                        "Categorical columns support encoding 'ordinal' (integer codes) or grouped 'one_hot'."
                    )
                finite = df[col].dropna()
                if finite.empty and "num_classes" not in cfg:
                    raise ValueError(
                        f"Unable to infer 'num_classes' for categorical column '{col}'. Provide it explicitly."
                    )
                if not finite.empty and not np.allclose(finite, np.round(finite)):
                    raise ValueError(
                        f"Categorical column '{col}' must contain integer codes when using ordinal encoding."
                    )
                inferred = int(np.max(finite.astype(int))) + 1 if not finite.empty else None
                num_classes = int(cfg.get("num_classes", inferred))
                if num_classes is None or num_classes < 2:
                    raise ValueError(
                        f"Categorical column '{col}' requires an integer 'num_classes' >= 2."
                    )
                param_count = num_classes
                encoding = "ordinal"
            elif dist_name == "t":
                param_count = 2
            elif dist_name == "poisson":
                param_count = 1
            elif dist_name == "beta":
                param_count = 2
            elif dist_name == "negative_binomial":
                param_count = 2
            elif dist_name == "pareto":
                param_count = 2
            else:  # pragma: no cover - exhaustive guard
                raise ValueError(f"Unhandled distribution '{dist_name}'")

            spec = _DistributionSpec(
                name=dist_name,
                param_count=param_count,
                slc=slice(total_params, total_params + param_count),
                column_indices=(idx,),
                column_names=(col,),
                num_classes=num_classes,
                df=cfg.get("df"),
                xm=cfg.get("xm"),
                encoding=encoding,
            )
            feature_specs.append(spec)
            total_params += param_count
            if dist_name in self._CONTINUOUS_SCALABLE:
                scalers[col] = StandardScaler()
            else:
                scalers[col] = None
            schema[col] = spec

        self._column_specs = schema
        self._column_scalers = scalers
        self._feature_specs = feature_specs
        self._total_param_dim = total_params
        self._validate_against_schema(df)

    def _validate_against_schema(self, df: pd.DataFrame) -> None:
        for spec in self._feature_specs:
            columns = list(spec.column_names)
            if spec.name == "bernoulli":
                for col in columns:
                    series = df[col]
                    invalid = ~series.dropna().isin({0, 1})
                    if invalid.any():
                        raise ValueError(
                            f"Bernoulli column '{col}' must contain only 0/1 values."
                        )
            elif spec.name == "categorical":
                if spec.encoding == "one_hot":
                    subset = df[columns]
                    values = subset.to_numpy(dtype=float)
                    if values.size == 0:
                        continue
                    finite_mask = np.isfinite(values)
                    if finite_mask.any():
                        clipped = values[finite_mask]
                        if (clipped < -1e-6).any() or (clipped > 1 + 1e-6).any():
                            raise ValueError(
                                f"One-hot categorical columns {columns} must contain probabilities between 0 and 1."
                            )
                    row_sums = np.nansum(values, axis=1)
                    if (row_sums > 1 + 1e-3).any():
                        raise ValueError(
                            f"Rows in one-hot group {columns} cannot sum to more than 1."
                        )
                else:
                    col = columns[0]
                    series = df[col]
                    if spec.num_classes is None:
                        raise ValueError(
                            f"Categorical schema for '{col}' must include 'num_classes'."
                        )
                    finite = series.dropna()
                    if not finite.empty and not np.allclose(finite, np.round(finite)):
                        raise ValueError(
                            f"Categorical column '{col}' must contain integer codes."
                        )
                    unique = finite.astype(int)
                    if unique.lt(0).any() or unique.ge(spec.num_classes).any():
                        raise ValueError(
                            f"Categorical column '{col}' must be integer encoded within [0, {spec.num_classes - 1}]."
                        )
            elif spec.name == "beta":
                col = columns[0]
                finite = df[col].dropna()
                if ((finite <= 0) | (finite >= 1)).any():
                    raise ValueError(
                        f"beta column '{col}' must be strictly within (0, 1)."
                    )
            elif spec.name == "pareto":
                col = columns[0]
                xm = spec.xm if spec.xm is not None else 1.0
                finite = df[col].dropna()
                if (finite < xm).any():
                    raise ValueError(
                        f"pareto column '{col}' must be >= xm (default 1.0)."
                    )

    def _ensure_schema_consistency(self, df: pd.DataFrame) -> None:
        missing = [col for col in self._original_columns or [] if col not in df.columns]
        if missing:
            raise ValueError(
                f"Input dataframe is missing expected columns: {', '.join(missing)}"
            )
        self._validate_against_schema(df)

    def _scale_values(
        self, values: np.ndarray, scaler: Optional[StandardScaler], *, fit: bool
    ) -> np.ndarray:
        mask = np.isfinite(values)
        scaled = np.zeros_like(values, dtype=np.float32)
        if scaler is not None:
            if fit:
                if mask.any():
                    scaler.fit(values[mask].reshape(-1, 1))
                else:
                    scaler.fit(np.array([[0.0]], dtype=np.float32))
            if mask.any():
                transformed = scaler.transform(values[mask].reshape(-1, 1)).reshape(-1)
                scaled[mask] = transformed.astype(np.float32)
            scaled[~mask] = 0.0
        else:
            scaled[mask] = values[mask].astype(np.float32)
            scaled[~mask] = 0.0
        return scaled.astype(np.float32)

    def _scale_frame(self, df: pd.DataFrame, *, fit: bool = False) -> ArrayLike:
        scaled_cols = []
        for col in df.columns:
            values = df[col].values.astype(np.float32)
            scaler = self._column_scalers.get(col)
            scaled = self._scale_values(values, scaler, fit=fit)
            scaled_cols.append(scaled)
        return np.column_stack(scaled_cols)

    def _scale_array(self, original: ArrayLike) -> ArrayLike:
        if self._original_columns is None:
            raise RuntimeError("Column metadata is unavailable for scaling.")
        scaled_cols = []
        for idx, col in enumerate(self._original_columns):
            values = original[:, idx].astype(np.float32)
            scaler = self._column_scalers.get(col)
            scaled = self._scale_values(values, scaler, fit=False)
            scaled_cols.append(scaled)
        return np.column_stack(scaled_cols)

    def _postprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        processed = df.copy()
        for spec in self._feature_specs:
            cols = list(spec.column_names)
            if spec.name in {"bernoulli", "poisson", "negative_binomial"}:
                for col in cols:
                    processed[col] = processed[col].round().astype(np.int64)
            elif spec.name == "categorical":
                if spec.encoding == "ordinal":
                    col = cols[0]
                    processed[col] = processed[col].round().astype(np.int64)
                else:
                    values = processed[cols].to_numpy(dtype=float)
                    if values.size == 0:
                        continue
                    row_sums = values.sum(axis=1)
                    max_idx = np.argmax(values, axis=1)
                    one_hot = np.zeros_like(values)
                    valid = row_sums > 0
                    if valid.any():
                        rows = np.arange(values.shape[0])[valid]
                        one_hot[rows, max_idx[valid]] = 1
                    processed.loc[:, cols] = one_hot.astype(np.int64)
            elif spec.name == "beta":
                col = cols[0]
                processed[col] = processed[col].clip(lower=1e-6, upper=1 - 1e-6)
            elif spec.name == "pareto":
                col = cols[0]
                xm = spec.xm if spec.xm is not None else 1.0
                processed[col] = processed[col].clip(lower=xm)
        return processed

    def _build_components(
        self, input_dim: int, params: Dict[str, float | int]
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        if self._total_param_dim is None:
            raise RuntimeError("Distribution schema must be prepared before building components.")
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
            output_dim=self._total_param_dim,
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

    def _distribution_reconstruction_loss(
        self,
        decoded: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._feature_specs:
            raise RuntimeError("Distribution schema unavailable for reconstruction loss.")
        losses = []
        masks = []
        for spec in self._feature_specs:
            params = decoded[:, spec.slc]
            if spec.encoding == "one_hot":
                indices = list(spec.column_indices)
                column_target = target[:, indices]
                feature_mask = mask[:, indices].amin(dim=1)
                row_sum = column_target.sum(dim=1)
                feature_mask = feature_mask * (row_sum > 0).float()
            else:
                idx = spec.column_indices[0]
                column_target = target[:, idx]
                feature_mask = mask[:, idx]
            loss = self._loss_for_spec(spec, params, column_target, feature_mask)
            losses.append(loss * feature_mask)
            masks.append(feature_mask)
        loss_stack = torch.stack(losses, dim=0)
        mask_stack = torch.stack(masks, dim=0)
        valid_counts = mask_stack.sum(dim=0).clamp_min(1.0)
        per_sample_loss = loss_stack.sum(dim=0) / valid_counts
        return per_sample_loss.mean(), per_sample_loss

    def _loss_for_spec(
        self,
        spec: _DistributionSpec,
        params: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        dist = spec.name
        eps = 1e-6
        valid_mask = mask > 0.5
        if target.dim() > 1:
            expanded_mask = valid_mask.unsqueeze(1)
        else:
            expanded_mask = valid_mask
        safe_target = torch.where(expanded_mask, target, torch.zeros_like(target))
        if dist == "gaussian":
            mean = params[:, 0]
            log_var = params[:, 1]
            var = torch.exp(log_var).clamp_min(1e-6)
            return 0.5 * (
                math.log(2 * math.pi) + log_var + (safe_target - mean) ** 2 / var
            )
        if dist == "bernoulli":
            return F.binary_cross_entropy_with_logits(
                params[:, 0], safe_target, reduction="none"
            )
        if dist == "categorical":
            logits = params
            if spec.encoding == "one_hot":
                target_long = safe_target.argmax(dim=1).long()
            else:
                target_long = safe_target.long().clamp_min(0)
            return F.cross_entropy(logits, target_long, reduction="none")
        if dist == "t":
            loc = params[:, 0]
            log_scale = params[:, 1]
            scale = F.softplus(log_scale) + eps
            df = torch.tensor(spec.df if spec.df is not None else 5.0, device=loc.device)
            y = (safe_target - loc) / scale
            log_norm = (
                torch.lgamma((df + 1) / 2)
                - torch.lgamma(df / 2)
                - 0.5 * (torch.log(df) + math.log(math.pi))
            )
            return -(log_norm - torch.log(scale) - ((df + 1) / 2) * torch.log1p((y ** 2) / df))
        if dist == "poisson":
            log_rate = params[:, 0]
            rate = torch.exp(log_rate).clamp_min(eps)
            return rate - safe_target * log_rate + torch.lgamma(safe_target + 1)
        if dist == "beta":
            log_alpha = params[:, 0]
            log_beta = params[:, 1]
            alpha = F.softplus(log_alpha) + eps
            beta = F.softplus(log_beta) + eps
            clipped_target = safe_target.clamp(eps, 1 - eps)
            return -(
                (alpha - 1) * torch.log(clipped_target)
                + (beta - 1) * torch.log1p(-clipped_target)
                - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))
            )
        if dist == "negative_binomial":
            log_mean = params[:, 0]
            log_disp = params[:, 1]
            mean = torch.exp(log_mean).clamp_min(eps)
            disp = F.softplus(log_disp) + eps
            return (
                torch.lgamma(safe_target + disp)
                - torch.lgamma(disp)
                - torch.lgamma(safe_target + 1)
                + disp * torch.log(disp / (disp + mean))
                + safe_target * torch.log(mean / (disp + mean))
            ) * -1
        if dist == "pareto":
            log_xm = params[:, 0]
            log_alpha = params[:, 1]
            xm = torch.exp(log_xm).clamp_min(eps)
            alpha = F.softplus(log_alpha) + eps
            clipped_target = torch.where(
                valid_mask, torch.max(safe_target, xm + eps), xm + eps
            )
            log_pdf = torch.log(alpha) + alpha * torch.log(xm) - (alpha + 1) * torch.log(clipped_target)
            return -log_pdf
        raise ValueError(f"Unsupported distribution {dist}")

    def _compute_loss(
        self,
        scaled_batch: torch.Tensor,
        original_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        encoder: nn.Module,
        predictor: nn.Module,
        decoder: nn.Module,
        params: Dict[str, float | int],
        reg_token: torch.Tensor,
        augment: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if augment:
            view_one = self._augment_batch(scaled_batch, params)
            view_two = self._augment_batch(scaled_batch, params)
        else:
            view_one = scaled_batch
            view_two = scaled_batch

        z_one = encoder(view_one)
        z_two = encoder(view_two)
        pred_one = predictor(z_one)
        pred_two = predictor(z_two)
        loss_jepa = self._byol_loss(pred_one, z_two) + self._byol_loss(pred_two, z_one)

        recon_weight = float(params["recon_weight"])
        regularization_weight = float(params.get("regularization_weight", 0.0))
        if recon_weight > 0:
            recon_one = decoder(z_one)
            recon_two = decoder(z_two)
            recon_loss_one, _ = self._distribution_reconstruction_loss(
                recon_one, original_batch, mask_batch
            )
            recon_loss_two, _ = self._distribution_reconstruction_loss(
                recon_two, original_batch, mask_batch
            )
            loss_recon = recon_loss_one + recon_loss_two
        else:
            loss_recon = torch.tensor(0.0, device=scaled_batch.device)

        if regularization_weight > 0:
            reg_token_expanded = reg_token.expand(scaled_batch.shape[0], -1)
            pred_reg_one = predictor(reg_token_expanded)
            pred_reg_two = predictor(reg_token_expanded)
            loss_reg = self._byol_loss(pred_reg_one, z_two) + self._byol_loss(
                pred_reg_two, z_one
            )
        else:
            loss_reg = torch.tensor(0.0, device=scaled_batch.device)

        total_loss = (
            loss_jepa
            + recon_weight * loss_recon
            + regularization_weight * loss_reg
        )
        return (
            total_loss,
            loss_jepa.detach(),
            loss_recon.detach(),
            loss_reg.detach(),
        )

    def _run_training(
        self,
        scaled_array: ArrayLike,
        original_array: ArrayLike,
        mask_array: ArrayLike,
        params: Dict[str, float | int],
        epochs: int,
        val_scaled: Optional[ArrayLike],
        val_original: Optional[ArrayLike],
        val_mask: Optional[ArrayLike],
    ) -> _TrainingResult:
        batch_size = max(1, min(int(params["batch_size"]), scaled_array.shape[0]))
        dataset = _TabularDataset(scaled_array, original_array, mask_array)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        encoder, predictor, decoder = self._build_components(scaled_array.shape[1], params)
        reg_token = nn.Parameter(
            torch.zeros(1, int(params["embedding_dim"]), device=self.device)
        )
        optim_params = list(encoder.parameters())
        optim_params += list(predictor.parameters())
        optim_params += list(decoder.parameters())
        optim_params.append(reg_token)
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
            total_reg = 0.0
            total_samples = 0
            for scaled_batch, original_batch, mask_batch in loader:
                scaled_batch = scaled_batch.to(self.device, non_blocking=True)
                original_batch = original_batch.to(self.device, non_blocking=True)
                mask_batch = mask_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss, loss_jepa, loss_recon, loss_reg = self._compute_loss(
                    scaled_batch,
                    original_batch,
                    mask_batch,
                    encoder,
                    predictor,
                    decoder,
                    params,
                    reg_token,
                    augment=True,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(optim_params, max_norm=5.0)
                optimizer.step()

                batch_size_now = scaled_batch.shape[0]
                total_loss += float(loss.item()) * batch_size_now
                total_jepa += float(loss_jepa.item()) * batch_size_now
                total_recon += float(loss_recon.item()) * batch_size_now
                total_reg += float(loss_reg.item()) * batch_size_now
                total_samples += batch_size_now

            if total_samples > 0:
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        "loss": total_loss / total_samples,
                        "jepa_loss": total_jepa / total_samples,
                        "recon_loss": total_recon / total_samples,
                        "reg_loss": total_reg / total_samples,
                    }
                )

        val_loss = None
        if (
            val_scaled is not None
            and val_original is not None
            and val_mask is not None
            and len(val_scaled) > 0
        ):
            val_loss = self._evaluate_loss(
                val_scaled,
                val_original,
                val_mask,
                encoder,
                predictor,
                decoder,
                params,
                reg_token,
            )

        encoder.eval()
        predictor.eval()
        decoder.eval()
        return _TrainingResult(
            encoder,
            predictor,
            decoder,
            reg_token.detach().cpu(),
            history,
            val_loss,
        )

    def _evaluate_loss(
        self,
        scaled_array: ArrayLike,
        original_array: ArrayLike,
        mask_array: ArrayLike,
        encoder: nn.Module,
        predictor: nn.Module,
        decoder: nn.Module,
        params: Dict[str, float | int],
        reg_token: torch.Tensor,
    ) -> float:
        dataset = _TabularDataset(scaled_array, original_array, mask_array)
        batch_size = max(1, min(int(params["batch_size"]), len(dataset)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        total_loss = 0.0
        total_samples = 0
        encoder.eval()
        predictor.eval()
        decoder.eval()
        with torch.no_grad():
            for scaled_batch, original_batch, mask_batch in loader:
                scaled_batch = scaled_batch.to(self.device, non_blocking=True)
                original_batch = original_batch.to(self.device, non_blocking=True)
                mask_batch = mask_batch.to(self.device, non_blocking=True)
                loss, _, _, _ = self._compute_loss(
                    scaled_batch,
                    original_batch,
                    mask_batch,
                    encoder,
                    predictor,
                    decoder,
                    params,
                    reg_token,
                    augment=False,
                )
                batch_size_now = scaled_batch.shape[0]
                total_loss += float(loss.item()) * batch_size_now
                total_samples += batch_size_now
        return total_loss / max(1, total_samples)

    def _train_and_score(
        self,
        train_scaled: ArrayLike,
        train_original: ArrayLike,
        train_mask: ArrayLike,
        val_scaled: ArrayLike,
        val_original: ArrayLike,
        val_mask: ArrayLike,
        params: Dict[str, float | int],
    ) -> Tuple[float, float]:
        if val_scaled.size == 0:
            raise ValueError("Validation data must be non-empty for tuning.")

        result = self._run_training(
            train_scaled,
            train_original,
            train_mask,
            params=params,
            epochs=min(self.tune_epochs, self.epochs),
            val_scaled=val_scaled,
            val_original=val_original,
            val_mask=val_mask,
        )
        encoder = result.encoder
        predictor = result.predictor
        decoder = result.decoder
        reg_token = result.reg_token.to(self.device)
        val_loss = result.val_loss
        if val_loss is None:
            val_loss = self._evaluate_loss(
                val_scaled,
                val_original,
                val_mask,
                encoder,
                predictor,
                decoder,
                params,
                reg_token,
            )

        with torch.no_grad():
            encoder.eval()
            tensor = torch.as_tensor(val_scaled, dtype=torch.float32, device=self.device)
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
            if _USING_FALLBACK_FIND_CLUSTERS:
                clusters = find_clusters(
                    reducer.graph_,
                    min_cluster_size=self.min_cluster_size,
                    cluster_selection_method=self.cluster_selection_method,
                    embedding=low_dim,
                    data=embeddings,
                )
            else:
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

    def _maybe_tune(
        self,
        scaled: ArrayLike,
        original: ArrayLike,
        mask: ArrayLike,
    ) -> Dict[str, float | int]:
        params = dict(self.hparams)
        if self.optuna_trials <= 0 or scaled.shape[0] < 4:
            return params

        val_size = max(1, int(round(scaled.shape[0] * self.tune_val_split)))
        if val_size >= scaled.shape[0]:
            val_size = scaled.shape[0] - 1
        if val_size <= 0:
            return params

        indices = np.arange(scaled.shape[0])
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=self.random_state,
            shuffle=True,
        )
        train_scaled = scaled[train_idx]
        train_original = original[train_idx]
        train_mask = mask[train_idx]
        val_scaled = scaled[val_idx]
        val_original = original[val_idx]
        val_mask = mask[val_idx]

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
            trial_params["regularization_weight"] = trial.suggest_float(
                "regularization_weight", 0.0, 3.0
            )
            trial_params["batch_size"] = trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256]
            )
            max_neighbors = max(5, min(60, train_scaled.shape[0] - 1))
            trial_params["umap_n_neighbors"] = trial.suggest_int(
                "umap_n_neighbors", 5, max_neighbors
            )
            trial_params["umap_min_dist"] = trial.suggest_float(
                "umap_min_dist", 0.0, 0.8
            )
            trial_params["umap_n_components"] = trial.suggest_int(
                "umap_n_components", 2, min(10, max(2, trial_params["embedding_dim"]))
            )

            val_loss, silhouette = self._train_and_score(
                train_scaled,
                train_original,
                train_mask,
                val_scaled,
                val_original,
                val_mask,
                trial_params,
            )
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

    def _generate_synthetic_original(
        self, n_samples: int
    ) -> Tuple[ArrayLike, ArrayLike]:
        if self._scaled_data is None or self._original_data is None:
            raise RuntimeError("Training data is not available for synthesis.")
        if n_samples <= 0:
            return (
                np.empty((0, self._scaled_data.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Model must be trained before generating synthetic samples.")

        indices = self._rng.integers(
            low=0,
            high=self._scaled_data.shape[0],
            size=n_samples,
            endpoint=False,
        )
        scaled_batch = torch.as_tensor(
            self._scaled_data[indices],
            dtype=torch.float32,
            device=self.device,
        )
        latent_noise_scale = float(self.hparams["latent_noise"]) * self.synthetic_temperature
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            latent = self.encoder(scaled_batch)
            noise = torch.randn_like(latent) * latent_noise_scale
            decoded_params = self.decoder(latent + noise)
            samples = self._sample_from_params(decoded_params.cpu())
        return samples.astype(np.float32), indices.astype(np.int64)

    def _sample_from_params(self, params: torch.Tensor) -> np.ndarray:
        if self._original_columns is None:
            raise RuntimeError("Column metadata unavailable for sampling.")
        params = params.to(torch.float32)
        device = params.device
        output = torch.zeros(
            (params.shape[0], len(self._original_columns)),
            dtype=torch.float32,
            device=device,
        )
        for spec in self._feature_specs:
            column_params = params[:, spec.slc]
            if spec.name == "gaussian":
                mean = column_params[:, 0]
                var = torch.exp(column_params[:, 1]).clamp_min(1e-6)
                std = torch.sqrt(var)
                dist = torch.distributions.Normal(mean, std)
                draw = dist.sample()
                output[:, spec.column_indices[0]] = draw
            elif spec.name == "bernoulli":
                dist = torch.distributions.Bernoulli(logits=column_params[:, 0])
                draw = dist.sample()
                output[:, spec.column_indices[0]] = draw
            elif spec.name == "categorical" and spec.encoding == "one_hot":
                dist = torch.distributions.Categorical(logits=column_params)
                draw = dist.sample()
                one_hot = F.one_hot(draw, num_classes=spec.param_count).to(torch.float32)
                output[:, list(spec.column_indices)] = one_hot
            elif spec.name == "categorical":
                dist = torch.distributions.Categorical(logits=column_params)
                draw = dist.sample().to(torch.float32)
                output[:, spec.column_indices[0]] = draw
            elif spec.name == "t":
                scale = F.softplus(column_params[:, 1]) + 1e-6
                df = torch.tensor(spec.df if spec.df is not None else 5.0, device=device)
                dist = torch.distributions.StudentT(df, loc=column_params[:, 0], scale=scale)
                draw = dist.sample()
                output[:, spec.column_indices[0]] = draw
            elif spec.name == "poisson":
                rate = torch.exp(column_params[:, 0]).clamp_min(1e-6)
                dist = torch.distributions.Poisson(rate)
                draw = dist.sample()
                output[:, spec.column_indices[0]] = draw
            elif spec.name == "beta":
                alpha = F.softplus(column_params[:, 0]) + 1e-6
                beta_param = F.softplus(column_params[:, 1]) + 1e-6
                dist = torch.distributions.Beta(alpha, beta_param)
                draw = dist.sample()
                output[:, spec.column_indices[0]] = draw
            elif spec.name == "negative_binomial":
                mean = torch.exp(column_params[:, 0]).clamp_min(1e-6)
                disp = F.softplus(column_params[:, 1]) + 1e-6
                probs = disp / (disp + mean)
                dist = torch.distributions.NegativeBinomial(total_count=disp, probs=probs)
                draw = dist.sample()
                output[:, spec.column_indices[0]] = draw
            elif spec.name == "pareto":
                xm = torch.exp(column_params[:, 0]).clamp_min(1e-6)
                alpha = F.softplus(column_params[:, 1]) + 1e-6
                dist = torch.distributions.Pareto(xm, alpha)
                draw = dist.sample()
                output[:, spec.column_indices[0]] = draw
            else:  # pragma: no cover - exhaustive guard
                raise ValueError(f"Unsupported distribution {spec.name}")
        return output.cpu().numpy()

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
            if _USING_FALLBACK_FIND_CLUSTERS:
                clusters = find_clusters(
                    reducer.graph_,
                    min_cluster_size=self.min_cluster_size,
                    cluster_selection_method=self.cluster_selection_method,
                    embedding=low_dim,
                    data=embeddings,
                )
            else:
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


__all__ = ["TJEPA"]
