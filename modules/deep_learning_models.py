
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
import hashlib
import json
import random
import os
from functools import wraps
from threadpoolctl import threadpool_limits

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

API_VERSION = 3


@dataclass(frozen=True)
class MLPConfig:
    # Original fields and positional order are preserved.
    name: str = "MLP 64-32"
    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 80
    patience: int = 12
    validation_fraction: float = 0.15
    balanced: bool = False
    preprocessing: str = "standard"
    architecture: str = "mlp"
    min_epochs: int = 20
    class_weight_power: float = 0.0
    drop_fnlwgt: bool = False
    residual_blocks: int = 2
    lr_step_size: int = 10
    lr_gamma: float = 0.5
    ensemble_size: int = 1
    numeric_bins: int = 32

    def __post_init__(self):
        if not self.name or not self.hidden_sizes or any(type(n) is not int or n < 1 for n in self.hidden_sizes):
            raise ValueError("A name and positive integer layer widths are required.")
        if self.architecture not in {"mlp", "residual"} or self.preprocessing not in {"standard", "log", "quantile", "piecewise"}:
            raise ValueError("Unsupported architecture or preprocessing.")
        if not 0 <= self.dropout < 1 or not 0 < self.validation_fraction < 0.5:
            raise ValueError("Invalid dropout or validation_fraction.")
        for n in (self.batch_size, self.max_epochs, self.min_epochs, self.patience,
                  self.residual_blocks, self.lr_step_size, self.ensemble_size, self.numeric_bins):
            if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
                raise ValueError("Batch size, epochs, patience and ensemble size must be positive integers.")
        numbers = [self.learning_rate, self.weight_decay, self.class_weight_power, self.lr_gamma]
        if not np.isfinite(numbers).all() or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings.")
        if not 0 <= self.class_weight_power <= 1 or not 0 < self.lr_gamma <= 1:
            raise ValueError("Invalid weighting or learning-rate decay.")


# Predeclared search space. Keep the two baseline names used by the notebook.
DEFAULT_CONFIGS = (
    MLPConfig(),
    MLPConfig(name="MLP 128-64", hidden_sizes=(128, 64), dropout=0.3),
    MLPConfig(name="MLP 64-32 balanced", balanced=True),
    MLPConfig(name="MLP log 128-64", hidden_sizes=(128, 64), preprocessing="log", dropout=0.1),
    MLPConfig(name="MLP log 256-128-64", hidden_sizes=(256, 128, 64), preprocessing="log", dropout=0.1),
    MLPConfig(name="MLP log slow", hidden_sizes=(128, 64), preprocessing="log", learning_rate=3e-4, dropout=0.05, weight_decay=1e-5),
    MLPConfig(name="MLP quantile 128-64", hidden_sizes=(128, 64), preprocessing="quantile", dropout=0.1),
    MLPConfig(name="MLP quantile 256-128", hidden_sizes=(256, 128), preprocessing="quantile", dropout=0.1),
    MLPConfig(name="MLP quantile weighted", hidden_sizes=(128, 64), preprocessing="quantile", dropout=0.1, class_weight_power=0.35),
    MLPConfig(name="Residual quantile 128", hidden_sizes=(128,), architecture="residual", preprocessing="quantile", dropout=0.1),
    MLPConfig(name="Residual log 128", hidden_sizes=(128,), architecture="residual", preprocessing="log", dropout=0.1),
    MLPConfig(name="MLP log without fnlwgt", hidden_sizes=(256, 128), preprocessing="log", dropout=0.05, weight_decay=1e-5, drop_fnlwgt=True),
    MLPConfig(name="MLP piecewise 128-64", hidden_sizes=(128, 64), preprocessing="piecewise", dropout=0.1),
    MLPConfig(name="MLP piecewise weighted", hidden_sizes=(128, 64), preprocessing="piecewise", dropout=0.1, class_weight_power=0.35),
    MLPConfig(name="MLP piecewise 256-128", hidden_sizes=(256, 128), preprocessing="piecewise", dropout=0.15, numeric_bins=64),
    MLPConfig(name="MLP piecewise weighted 256-128", hidden_sizes=(256, 128), preprocessing="piecewise", dropout=0.15, numeric_bins=64, class_weight_power=0.35),
)


def _limited_threads(function):
    # Avoid small tabular networks slowing down under large Colab CPU thread pools.
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(min(4, previous))
            with threadpool_limits(limits=4):
                return function(*args, **kwargs)
        finally:
            torch.set_num_threads(previous)
    return wrapped


def _resolve_device(device):
    if str(device) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; pass device='cpu' or device='auto'.")
    return str(result)


def seed_everything(seed=42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


class TailLogTransformer(TransformerMixin, BaseEstimator):
    """Signed log1p for fixed long-tail columns, with no learned statistics."""
    def __init__(self, positions=()):
        self.positions = positions

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float).copy()
        if self.positions:
            subset = values[:, list(self.positions)]
            values[:, list(self.positions)] = np.sign(subset) * np.log1p(np.abs(subset))
        return values


class PiecewiseLinearEncoder(TransformerMixin, BaseEstimator):
    """Train-fitted numerical bases, followed by the network's learned projection.

    Inspired by Gorishniy et al., NeurIPS 2022 (arxiv.org/abs/2203.05556).
    Knots use quantiles of DISTINCT observed values, so mostly-zero capital
    variables retain resolution among nonzero values. Each interval outputs
    clip((x-left)/(right-left), 0, 1). No labels are used to choose knots.
    Constant columns produce zeros; out-of-range values clamp to endpoints.
    """
    def __init__(self, n_bins=32):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = values.shape[1]
        self.edges_ = []
        for column in values.T:
            unique = np.unique(column)
            edges = np.unique(np.quantile(unique, np.linspace(0, 1, min(self.n_bins, len(unique) - 1) + 1)))
            self.edges_.append(edges)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=np.float64)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Numeric schema differs from fitted schema.")
        encoded = []
        for column, edges in zip(values.T, self.edges_):
            if len(edges) < 2:
                encoded.append(np.zeros((len(values), 1)))
            else:
                encoded.append(np.clip((column[:, None] - edges[:-1]) / np.diff(edges), 0, 1))
        return np.concatenate(encoded, axis=1)


def build_feature_preprocessor(X, config, *, seed=42):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if len(numeric) + len(categorical) != X.shape[1]:
        raise ValueError("Unsupported feature dtype.")
    if config.drop_fnlwgt:
        numeric = [name for name in numeric if name != "fnlwgt"]
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if config.preprocessing == "log":
        positions = tuple(i for i, name in enumerate(numeric) if name in {"fnlwgt", "capital_gain", "capital_loss", "capital-gain", "capital-loss"})
        steps.append(("tail_log", TailLogTransformer(positions)))
    if config.preprocessing == "piecewise":
        steps.append(("piecewise", PiecewiseLinearEncoder(config.numeric_bins)))
    elif config.preprocessing == "quantile":
        steps.append(("scale", QuantileTransformer(
            n_quantiles=min(512, len(X)), output_distribution="normal",
            random_state=seed, subsample=max(10000, len(X)),
        )))
    else:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], sparse_threshold=0)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(width, width), nn.Dropout(dropout),
        )

    def forward(self, values):
        return values + self.body(values)


def build_mlp(input_dim, config):
    if config.architecture == "residual":
        width = config.hidden_sizes[0]
        return nn.Sequential(
            nn.Linear(input_dim, width), nn.ReLU(),
            *[ResidualBlock(width, config.dropout) for _ in range(config.residual_blocks)],
            nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1),
        )
    layers, previous = [], input_dim
    for width in config.hidden_sizes:
        layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(config.dropout)])
        previous = width
    return nn.Sequential(*layers, nn.Linear(previous, 1))


def _validate_xy(X, y):
    if not isinstance(X, pd.DataFrame) or X.empty or "income" in X.columns or not X.columns.is_unique:
        raise ValueError("X must be a nonempty feature DataFrame with unique columns, without income.")
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(X) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Aligned binary labels containing both classes 0 and 1 are required.")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("X/y indexes are not aligned.")
    return labels.astype(np.float32)


def _features(transformer, X):
    values = np.asarray(transformer.transform(X), dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Features must be finite and two-dimensional.")
    return np.ascontiguousarray(values)


def _probabilities(network, values, device):
    network.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(values), 4096):
            batch = torch.from_numpy(values[start:start + 4096]).to(device)
            batches.append(torch.sigmoid(network(batch).squeeze(1)).cpu().numpy())
    return np.concatenate(batches) if batches else np.empty(0, dtype=np.float32)


def _fit_network(values, labels, config, *, seed, device, epochs, validation=None):
    seed_everything(seed)
    network = build_mlp(values.shape[1], config).to(device)
    power = 1.0 if config.balanced else config.class_weight_power
    ratio = float((labels == 0).sum() / (labels == 1).sum()) ** power
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    tensors = torch.from_numpy(values).to(device), torch.from_numpy(labels).to(device)
    generator = torch.Generator().manual_seed(seed)
    history, best_state = [], None
    best_loss, best_epoch, stale = np.inf, 0, 0
    for epoch in range(1, epochs + 1):
        # Same schedule in CV and full-train refitting; no test-dependent schedule.
        rate = max(1e-5, config.learning_rate * config.lr_gamma ** ((epoch - 1) // config.lr_step_size))
        rate = min(rate, config.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = rate
        network.train()
        order = torch.randperm(len(labels), generator=generator).to(device)
        total_loss = 0.0
        for start in range(0, len(labels), config.batch_size):
            indexes = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(tensors[0][indexes]).squeeze(1), tensors[1][indexes])
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite.")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indexes)
        row = {"epoch": epoch, "train_loss": total_loss / len(labels), "learning_rate": rate}
        if validation is not None:
            val_values, val_labels = validation
            probabilities = _probabilities(network, val_values, device)
            p = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
            # Monitor the smooth training objective, not noisy thresholded F1.
            val_loss = float(-(ratio * val_labels * np.log(p) + (1 - val_labels) * np.log1p(-p)).mean())
            row.update(validation_loss=val_loss, validation_f1_macro=float(f1_score(
                val_labels, probabilities >= 0.5, average="macro", zero_division=0)))
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = deepcopy(network.state_dict())
            else:
                stale += 1
        history.append(row)
        if validation is not None and epoch >= min(config.min_epochs, epochs) and stale >= config.patience:
            break
    if best_state is not None:
        network.load_state_dict(best_state)
    return network.eval(), pd.DataFrame(history), best_epoch or epochs


class ProbabilityEnsemble(nn.Module):
    """Average member probabilities; return logits for the existing predictor."""
    def __init__(self, networks):
        super().__init__()
        self.members = nn.ModuleList(networks)

    def forward(self, values):
        probability = torch.stack([torch.sigmoid(net(values)) for net in self.members]).mean(dim=0)
        return torch.logit(probability.clamp(1e-7, 1 - 1e-7))


@dataclass
class FittedMLP:
    preprocessor: object
    network: nn.Module
    config: MLPConfig
    feature_columns: tuple[str, ...]
    epochs: int
    seed: int
    device: str = "cpu"
    input_dim: int | None = None

    @property
    def classes_(self):
        return np.array([0, 1])

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or set(X.columns) != set(self.feature_columns):
            raise ValueError("Prediction columns do not match the training schema.")
        values = _features(self.preprocessor, X.loc[:, list(self.feature_columns)])
        positive = _probabilities(self.network, values, self.device)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _member_seed(seed, member):
    return int((seed + 1009 * member) % (2**32 - 1))


def _metrics(labels, probabilities):
    predictions = probabilities >= 0.5
    return {
        "cv_accuracy": accuracy_score(labels, predictions),
        "cv_f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "cv_f1_>50K": f1_score(labels, predictions, zero_division=0),
        "cv_precision_>50K": precision_score(labels, predictions, zero_division=0),
        "cv_recall_>50K": recall_score(labels, predictions, zero_division=0),
        "cv_roc_auc": roc_auc_score(labels, probabilities),
        "cv_average_precision": average_precision_score(labels, probabilities),
    }


def _fold(X, labels, config, train_idx, held_idx, *, seed, fold, device, cached=None):
    fit_idx, stop_idx = train_test_split(train_idx, test_size=config.validation_fraction,
        stratify=labels[train_idx], random_state=seed + fold)
    started = perf_counter()
    transformer = build_feature_preprocessor(X.iloc[fit_idx], config, seed=seed + fold).fit(X.iloc[fit_idx])
    fit_values, stop_values, held_values = [_features(transformer, X.iloc[index]) for index in (fit_idx, stop_idx, held_idx)]
    members = list(cached or [])
    for member in range(len(members), config.ensemble_size):
        member_start = perf_counter()
        network, history, best_epoch = _fit_network(fit_values, labels[fit_idx], config,
            seed=_member_seed(seed + fold, member), device=device, epochs=config.max_epochs,
            validation=(stop_values, labels[stop_idx]))
        members.append({"history": history, "best_epoch": best_epoch,
            "fit_probability": _probabilities(network, fit_values, device),
            "held_probability": _probabilities(network, held_values, device),
            "seconds": perf_counter() - member_start})
        del network
    probability = np.mean([m["held_probability"] for m in members], axis=0)
    train_probability = np.mean([m["fit_probability"] for m in members], axis=0)
    row = {"model": config.name, "fold": fold,
        "best_epoch": int(np.median([m["best_epoch"] for m in members])),
        **_metrics(labels[held_idx], probability),
        "train_f1_macro": f1_score(labels[fit_idx], train_probability >= 0.5, average="macro", zero_division=0),
        "mean_fit_seconds": perf_counter() - started + sum(m["seconds"] for m in (cached or [])),
        "n_fit": len(fit_idx), "n_early_stop": len(stop_idx), "n_held_out": len(held_idx),
        "ensemble_size": config.ensemble_size,
        "member_best_epochs": json.dumps([m["best_epoch"] for m in members]),
    }
    # A mean of member learning curves keeps one row/epoch/fold for old plots.
    history = pd.concat([m["history"] for m in members]).groupby("epoch", as_index=False).mean(numeric_only=True)
    history = history.assign(model=config.name, fold=fold, ensemble_size=config.ensemble_size)
    print(f"{config.name} | fold {fold}/3 | F1-macro={row['cv_f1_macro']:.4f} | epoch={row['best_epoch']}", flush=True)
    return row, history, members


def _summary(rows):
    folds = pd.DataFrame(rows)
    metric_cols = [c for c in folds if c.startswith("cv_")] + ["train_f1_macro", "mean_fit_seconds"]
    grouped = folds.groupby("model", sort=False)
    summary = grouped[metric_cols].mean()
    summary["cv_f1_macro_std"] = grouped["cv_f1_macro"].std(ddof=0)
    summary["refit_epochs"] = grouped["best_epoch"].median().astype(int)
    summary["f1_macro_gap"] = summary["train_f1_macro"] - summary["cv_f1_macro"]
    summary["ensemble_size"] = grouped["ensemble_size"].first()
    return summary.reset_index().sort_values("cv_f1_macro", ascending=False, kind="stable").reset_index(drop=True)


@_limited_threads
def run_deep_learning_comparison(X, y, *, configs=None, seed=42, device="cpu", ensemble_top_k=None):
    """Return (summary, folds, histories, selected_config), as in the old module.

    Default: 16 single-network configurations + ensembles of the best two.
    Explicit configs: use exactly those configs unless ensemble_top_k is supplied.
    Three fixed member seeds, not a search for the luckiest seed. Outer folds are
    not used for early stopping. Ensemble shortlist uses CV, so its CV scores are
    model-selection scores and should not be described as unbiased estimates.
    """
    labels = _validate_xy(X, y)
    device = _resolve_device(device)
    top_k = (2 if configs is None else 0) if ensemble_top_k is None else ensemble_top_k
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("ensemble_top_k must be a nonnegative integer.")
    configs = tuple(DEFAULT_CONFIGS if configs is None else configs)
    if not configs or len({c.name for c in configs}) != len(configs):
        raise ValueError("Provide configurations with unique names.")
    if np.bincount(labels.astype(int)).min() < 3:
        raise ValueError("3-fold CV requires at least three samples per class.")
    splits = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, labels))
    rows, histories, cache = [], [], {}
    for config in configs:
        for fold, (train_idx, held_idx) in enumerate(splits, 1):
            row, history, members = _fold(X, labels, config, train_idx, held_idx,
                seed=seed, fold=fold, device=device)
            rows.append(row)
            histories.append(history)
            cache[(config.name, fold)] = members
    eligible = [name for name in _summary(rows).model
                if next(c for c in configs if c.name == name).ensemble_size == 1][:top_k]
    expanded = list(configs)
    for name in eligible:
        base = next(c for c in configs if c.name == name)
        config = replace(base, name=f"{base.name} + ensemble 3 seeds", ensemble_size=3)
        if any(c.name == config.name for c in expanded):
            raise ValueError("Generated ensemble name conflicts with a supplied configuration.")
        expanded.append(config)
        for fold, (train_idx, held_idx) in enumerate(splits, 1):
            row, history, _ = _fold(X, labels, config, train_idx, held_idx,
                seed=seed, fold=fold, device=device, cached=cache[(base.name, fold)])
            rows.append(row)
            histories.append(history)
    summary = _summary(rows)
    selected = next(c for c in expanded if c.name == summary.iloc[0]["model"])
    print(f"Selected using train CV only: {selected.name}; threshold=0.5", flush=True)
    return summary, pd.DataFrame(rows), pd.concat(histories, ignore_index=True), selected


@_limited_threads
def fit_final_deep_learning(X, y, config, *, epochs, seed=42, device="cpu"):
    """Refit on all train rows at the CV-selected epoch budget; returns two items."""
    labels = _validate_xy(X, y)
    device = _resolve_device(device)
    if not isinstance(epochs, (int, np.integer)) or isinstance(epochs, bool) or not 1 <= epochs <= config.max_epochs:
        raise ValueError("epochs must be selected by CV and within max_epochs.")
    transformer = build_feature_preprocessor(X, config, seed=seed).fit(X)
    values = _features(transformer, X)
    networks, histories = [], []
    for member in range(config.ensemble_size):
        network, history, _ = _fit_network(values, labels, config,
            seed=_member_seed(seed, member), device=device, epochs=int(epochs))
        networks.append(network)
        histories.append(history)
        print(f"Full-train refit {member + 1}/{config.ensemble_size}: {epochs} epochs, {len(X)} rows.", flush=True)
    network = networks[0] if len(networks) == 1 else ProbabilityEnsemble(networks).to(device).eval()
    history = pd.concat(histories).groupby("epoch", as_index=False).mean(numeric_only=True)
    history["ensemble_size"] = config.ensemble_size
    return FittedMLP(transformer, network, config, tuple(X.columns), int(epochs), seed, device, values.shape[1]), history


def evaluate_deep_learning_test(model, X_test, y_test):
    """Evaluate a frozen model at threshold 0.5 without fitting or selection."""
    labels = _validate_xy(X_test, y_test)
    scores = model.predict_proba(X_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    metrics = pd.DataFrame([{
        "model": model.config.name,
        "test_accuracy": accuracy_score(labels, predictions),
        "test_f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "test_f1_>50K": f1_score(labels, predictions, zero_division=0),
        "test_precision_>50K": precision_score(labels, predictions, zero_division=0),
        "test_recall_>50K": recall_score(labels, predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(labels, scores),
        "test_average_precision": average_precision_score(labels, scores),
    }])
    report = classification_report(
        labels, predictions, labels=[0, 1], target_names=["<=50K", ">50K"],
        output_dict=True, zero_division=0,
    )
    return metrics, report, predictions


def save_deep_learning_model(model, output_dir):
    """Save all member weights, preprocessing and config; prediction stays at 0.5."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.network.state_dict().items()}, destination / "weights.pt")
    joblib.dump(model.preprocessor, destination / "preprocessor.joblib")
    input_dim = model.input_dim
    if input_dim is None:
        first = model.network.members[0] if isinstance(model.network, ProbabilityEnsemble) else model.network
        input_dim = first[0].in_features
    metadata = {"api_version": API_VERSION, "config": asdict(model.config),
        "feature_columns": list(model.feature_columns), "input_dim": input_dim,
        "epochs": model.epochs, "seed": model.seed, "threshold": 0.5,
        "member_seeds": [_member_seed(model.seed, i) for i in range(model.config.ensemble_size)],
        "torch": str(torch.__version__), "target_mapping": {"<=50K": 0, ">50K": 1},
        "selection": "Training-only 3-fold CV; fixed threshold 0.5"}
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_deep_learning_model(output_dir, *, device="cpu"):
    """Load trusted local checkpoints, including original single-MLP checkpoints."""
    device = _resolve_device(device)
    source = Path(output_dir)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("api_version", 1) not in {1, API_VERSION}:
        raise ValueError("Unsupported checkpoint format.")
    options = dict(metadata["config"])
    options["hidden_sizes"] = tuple(options["hidden_sizes"])
    config = MLPConfig(**options)
    networks = [build_mlp(metadata["input_dim"], config) for _ in range(config.ensemble_size)]
    network = networks[0] if len(networks) == 1 else ProbabilityEnsemble(networks)
    network.load_state_dict(torch.load(source / "weights.pt", map_location="cpu", weights_only=True))
    network.to(device).eval()
    return FittedMLP(joblib.load(source / "preprocessor.joblib"), network, config,
        tuple(metadata["feature_columns"]), metadata["epochs"], metadata["seed"], device, metadata["input_dim"])
