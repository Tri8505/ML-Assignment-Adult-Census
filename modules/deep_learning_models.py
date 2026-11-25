"""PyTorch MLP experiments for Adult Census, with leakage-free model selection.

Place this file in modules/ beside preprocessor.py. All selection uses training
data only. Each CV fold has its own inner early-stopping split and transformer.
The final network is refit on all training rows for the median selected epoch.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .preprocessor import build_preprocessor


@dataclass(frozen=True)
class MLPConfig:
    name: str = "MLP 64-32"
    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 7
    validation_fraction: float = 0.15
    balanced: bool = False

    def __post_init__(self):
        if not self.name or not self.hidden_sizes or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1
            for n in self.hidden_sizes
        ):
            raise ValueError("Provide a name and positive integer hidden sizes.")
        if not 0 <= self.dropout < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("Invalid dropout or validation_fraction.")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
               for n in (self.batch_size, self.max_epochs, self.patience)):
            raise ValueError("Batch size, epochs and patience must be positive integers.")
        if not np.isfinite([self.learning_rate, self.weight_decay]).all() or (
            self.learning_rate <= 0 or self.weight_decay < 0
        ):
            raise ValueError("Invalid optimizer settings.")


DEFAULT_CONFIGS = (
    MLPConfig(),
    MLPConfig(name="MLP 128-64", hidden_sizes=(128, 64), dropout=0.3),
    MLPConfig(name="MLP 64-32 balanced", balanced=True),
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU is the default reproducible execution path. CUDA may require
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def build_mlp(input_dim, config):
    layers = []
    previous = input_dim
    for width in config.hidden_sizes:
        layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(config.dropout)])
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def _validate_xy(X, y):
    if not isinstance(X, pd.DataFrame) or X.empty or "income" in X.columns:
        raise ValueError("X must be a nonempty feature DataFrame without income.")
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(X) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("y must be aligned with X and contain both binary classes 0/1.")
    if isinstance(y, pd.Series) and not X.index.equals(y.index):
        raise ValueError("X and y indexes must match in the same order.")
    return labels.astype(np.float32)


def _features(transformer, X):
    values = np.asarray(transformer.transform(X), dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite, dense, two-dimensional features.")
    return values


def _probabilities(network, values, device, batch_size=4096):
    network.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size]).to(device)
            output.append(torch.sigmoid(network(batch).squeeze(1)).cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def _fit_network(values, labels, config, *, seed, device, epochs, validation=None):
    seed_everything(seed)
    network = build_mlp(values.shape[1], config).to(device)
    ratio = float((labels == 0).sum() / (labels == 1).sum()) if config.balanced else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio], device=device))
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(labels)),
        batch_size=config.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    history, best_state = [], None
    best_score, best_epoch, stale = -np.inf, 0, 0
    for epoch in range(1, epochs + 1):
        network.train()
        total_loss = 0.0
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(network(batch).squeeze(1), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        row = {"epoch": epoch, "train_loss": total_loss / len(labels)}
        if validation is not None:
            val_values, val_labels = validation
            probabilities = _probabilities(network, val_values, device)
            score = f1_score(val_labels, probabilities >= 0.5, average="macro", zero_division=0)
            row["validation_f1_macro"] = float(score)
            if score > best_score + 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = deepcopy(network.state_dict())
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= config.patience:
            break
    if best_state is not None:
        network.load_state_dict(best_state)
    return network, pd.DataFrame(history), best_epoch or epochs


@dataclass
class FittedMLP:
    preprocessor: object
    network: nn.Module
    config: MLPConfig
    feature_columns: tuple[str, ...]
    epochs: int
    seed: int
    device: str = "cpu"

    @property
    def classes_(self):
        return np.array([0, 1])

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame) or set(X.columns) != set(self.feature_columns):
            raise ValueError("Prediction columns do not match the training schema.")
        values = _features(self.preprocessor, X.loc[:, list(self.feature_columns)])
        positive = _probabilities(self.network, values, self.device)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def run_deep_learning_comparison(X, y, *, configs=None, seed=42, device="cpu"):
    """Return summary, fold metrics, epoch histories, and the CV-selected config.

    Outer validation rows never fit preprocessing or determine early stopping.
    The early-stop split is stratified, and preprocessing fits its train rows only.
    """
    labels = _validate_xy(X, y)
    configs = tuple(DEFAULT_CONFIGS if configs is None else configs)
    if not configs or len({c.name for c in configs}) != len(configs):
        raise ValueError("Provide configurations with unique names.")
    if np.bincount(labels.astype(int)).min() < 3:
        raise ValueError("3-fold CV requires at least three samples per class.")
    rows, histories = [], []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for config in configs:
        for fold, (train_idx, held_idx) in enumerate(splitter.split(X, labels), 1):
            started = perf_counter()
            fit_idx, stop_idx = train_test_split(
                train_idx, test_size=config.validation_fraction,
                stratify=labels[train_idx], random_state=seed + fold,
            )
            transformer = build_preprocessor(X.iloc[fit_idx])
            transformer.fit(X.iloc[fit_idx])
            network, history, best_epoch = _fit_network(
                _features(transformer, X.iloc[fit_idx]), labels[fit_idx], config,
                seed=seed + fold, device=device, epochs=config.max_epochs,
                validation=(_features(transformer, X.iloc[stop_idx]), labels[stop_idx]),
            )
            fit_seconds = perf_counter() - started
            fitted = FittedMLP(transformer, network, config, tuple(X.columns), best_epoch, seed + fold, device)
            predictions = fitted.predict(X.iloc[held_idx])
            actual = labels[held_idx]
            rows.append({
                "model": config.name, "fold": fold, "best_epoch": best_epoch,
                "cv_accuracy": accuracy_score(actual, predictions),
                "cv_f1_macro": f1_score(actual, predictions, average="macro", zero_division=0),
                "cv_f1_>50K": f1_score(actual, predictions, zero_division=0),
                "cv_precision_>50K": precision_score(actual, predictions, zero_division=0),
                "cv_recall_>50K": recall_score(actual, predictions, zero_division=0),
                "train_f1_macro": f1_score(labels[fit_idx], fitted.predict(X.iloc[fit_idx]), average="macro", zero_division=0),
                "mean_fit_seconds": fit_seconds,
                "n_fit": len(fit_idx), "n_early_stop": len(stop_idx), "n_held_out": len(held_idx),
            })
            histories.append(history.assign(model=config.name, fold=fold))
            print(f"{config.name} | fold {fold}/3 | F1-macro={rows[-1]['cv_f1_macro']:.4f} | epoch={best_epoch}")
    folds = pd.DataFrame(rows)
    metric_cols = [c for c in folds if c.startswith("cv_")] + ["train_f1_macro", "mean_fit_seconds"]
    summary = folds.groupby("model", sort=False)[metric_cols].mean()
    summary["cv_f1_macro_std"] = folds.groupby("model")["cv_f1_macro"].std(ddof=0)
    summary["refit_epochs"] = folds.groupby("model")["best_epoch"].median().astype(int)
    summary["f1_macro_gap"] = summary["train_f1_macro"] - summary["cv_f1_macro"]
    summary = summary.reset_index().sort_values("cv_f1_macro", ascending=False, kind="stable").reset_index(drop=True)
    selected = next(c for c in configs if c.name == summary.iloc[0]["model"])
    return summary, folds, pd.concat(histories, ignore_index=True), selected


def fit_final_deep_learning(X, y, config, *, epochs, seed=42, device="cpu"):
    """Refit preprocessing and network on all train rows; no test input accepted."""
    labels = _validate_xy(X, y)
    if not isinstance(epochs, (int, np.integer)) or not 1 <= epochs <= config.max_epochs:
        raise ValueError("epochs must be a positive integer within max_epochs, selected by CV.")
    transformer = build_preprocessor(X).fit(X)
    network, history, _ = _fit_network(
        _features(transformer, X), labels, config,
        seed=seed, device=device, epochs=int(epochs),
    )
    return FittedMLP(transformer, network, config, tuple(X.columns), int(epochs), seed, device), history


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
    """Save CPU weights, preprocessing, configuration and prediction schema."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.network.state_dict().items()}, destination / "weights.pt")
    joblib.dump(model.preprocessor, destination / "preprocessor.joblib")
    metadata = {
        "config": asdict(model.config), "feature_columns": list(model.feature_columns),
        "input_dim": model.network[0].in_features, "epochs": model.epochs,
        "seed": model.seed, "threshold": 0.5, "torch": torch.__version__,
        "target_mapping": {"<=50K": 0, ">50K": 1},
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_deep_learning_model(output_dir, *, device="cpu"):
    """Load a trusted local artifact; joblib files must not come from strangers."""
    source = Path(output_dir)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    options = metadata["config"]
    options["hidden_sizes"] = tuple(options["hidden_sizes"])
    config = MLPConfig(**options)
    network = build_mlp(metadata["input_dim"], config)
    network.load_state_dict(torch.load(source / "weights.pt", map_location="cpu", weights_only=True))
    network.to(device).eval()
    return FittedMLP(
        joblib.load(source / "preprocessor.joblib"), network, config,
        tuple(metadata["feature_columns"]), metadata["epochs"], metadata["seed"], device,
    )
