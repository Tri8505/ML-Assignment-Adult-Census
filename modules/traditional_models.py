"""The original Adult Census experiments, expressed as reusable functions.

All selection uses stratified CV on training data. evaluate_test never fits.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, f1_score,
    make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocessor import build_preprocessor


def make_cv(seed=42):
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)


def make_scoring():
    return {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_positive": "f1",
        "precision_positive": make_scorer(precision_score, zero_division=0),
        "recall_positive": "recall",
    }


def build_model_pipeline(X, model, *, preprocessing=None, pca=None):
    """Build a fresh pipeline. A PCA fraction uses the full SVD solver."""
    return Pipeline([
        ("preprocessor", build_preprocessor(X, **(preprocessing or {}))),
        ("reduce", "passthrough" if pca is None else PCA(
            n_components=pca, svd_solver="full"
        )),
        ("classifier", clone(model)),
    ])


def evaluate_cv(name, pipeline, X, y, *, seed=42):
    """Return one summary row; all fitted preprocessing stays within each fold."""
    scores = cross_validate(
        pipeline, X, y, cv=make_cv(seed), scoring=make_scoring(),
        return_train_score=True, n_jobs=1, error_score="raise",
    )
    return {
        "model": name,
        "cv_accuracy": scores["test_accuracy"].mean(),
        "cv_f1_macro": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std(),
        "cv_f1_>50K": scores["test_f1_positive"].mean(),
        "cv_precision_>50K": scores["test_precision_positive"].mean(),
        "cv_recall_>50K": scores["test_recall_positive"].mean(),
        "train_f1_macro": scores["train_f1_macro"].mean(),
        "mean_fit_seconds": scores["fit_time"].mean(),
    }


def _sort_results(rows):
    results = pd.DataFrame(rows)
    results["f1_macro_gap"] = results["train_f1_macro"] - results["cv_f1_macro"]
    return results.sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)


def run_baselines(X, y, *, seed=42):
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
    }
    rows = []
    for name, model in models.items():
        print(f"Running: {name}")
        rows.append(evaluate_cv(name, build_model_pipeline(X, model), X, y, seed=seed))
    return _sort_results(rows)


def run_model_comparison(X, y, baseline_results, *, seed=42):
    """Reuse baseline scores and evaluate only the three additional models."""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "LinearSVC": LinearSVC(C=1.0, dual=False, max_iter=10000, random_state=seed),
        "Logistic Regression balanced": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed
        ),
    }
    rows = baseline_results.to_dict("records")
    for name, model in models.items():
        print(f"Running: {name}")
        rows.append(evaluate_cv(name, build_model_pipeline(X, model), X, y, seed=seed))
    return _sort_results(rows)


def run_preprocessing_comparison(X, y, baseline_results, *, seed=42):
    baseline = baseline_results.loc[
        baseline_results["model"] == "Logistic Regression"
    ]
    if len(baseline) != 1:
        raise ValueError("Expected exactly one Logistic Regression baseline row.")
    rows = baseline.to_dict("records")
    rows[0]["model"] = "LR baseline: Standard + mode"
    experiments = [
        ("LR + MinMax (0,1)", {"scaler": "minmax", "feature_range": (0, 1)}, None),
        ("LR + MinMax (-1,1)", {"scaler": "minmax", "feature_range": (-1, 1)}, None),
        ("LR + Unknown imputation", {"categorical_strategy": "constant"}, None),
        ("LR + PCA 90%", {}, 0.90),
        ("LR + PCA 95%", {}, 0.95),
    ]
    for name, options, pca in experiments:
        print(f"Running: {name}")
        pipeline = build_model_pipeline(
            X, LogisticRegression(max_iter=2000, random_state=seed),
            preprocessing=options, pca=pca,
        )
        rows.append(evaluate_cv(name, pipeline, X, y, seed=seed))
    results = _sort_results(rows)
    results["delta_f1_vs_baseline"] = (
        results["cv_f1_macro"] - float(baseline["cv_f1_macro"].iloc[0])
    )
    return results


def tune_random_forest(X, y, *, seed=42):
    """Evaluate the same 12 configurations as the original cell 9 and refit."""
    pipeline = build_model_pipeline(X, RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1
    ))
    grid = {
        "preprocessor__categorical__imputer__strategy": ["most_frequent", "constant"],
        "classifier__max_depth": [12, 20, None],
        "classifier__min_samples_leaf": [2, 5],
    }
    search = GridSearchCV(
        pipeline, grid, scoring=make_scoring(), refit="f1_macro",
        cv=make_cv(seed), return_train_score=True, n_jobs=1,
        verbose=1, error_score="raise",
    )
    search.fit(X, y)
    return search


def summarize_search(search):
    raw = pd.DataFrame(search.cv_results_)
    columns = {
        "rank_test_f1_macro": "rank",
        "param_preprocessor__categorical__imputer__strategy": "imputation",
        "param_classifier__max_depth": "max_depth",
        "param_classifier__min_samples_leaf": "min_samples_leaf",
        "mean_test_accuracy": "cv_accuracy",
        "mean_test_f1_macro": "cv_f1_macro",
        "std_test_f1_macro": "cv_f1_macro_std",
        "mean_test_f1_positive": "cv_f1_>50K",
        "mean_test_precision_positive": "cv_precision_>50K",
        "mean_test_recall_positive": "cv_recall_>50K",
        "mean_train_f1_macro": "train_f1_macro",
        "mean_fit_time": "mean_fit_seconds",
    }
    summary = raw[list(columns)].rename(columns=columns)
    summary["f1_macro_gap"] = summary["train_f1_macro"] - summary["cv_f1_macro"]
    return summary.sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)


def evaluate_test(model, X_test, y_test):
    """Evaluate a previously fitted final classifier. Never tune or fit here.

    Returns (metrics DataFrame, classification-report dict, predictions).
    """
    predictions = model.predict(X_test)
    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(X_test)[:, positive_index]
    metrics = pd.DataFrame([{
        "model": "Selected Random Forest",
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_f1_macro": f1_score(y_test, predictions, average="macro"),
        "test_f1_>50K": f1_score(y_test, predictions, zero_division=0),
        "test_precision_>50K": precision_score(y_test, predictions, zero_division=0),
        "test_recall_>50K": recall_score(y_test, predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, scores),
        "test_average_precision": average_precision_score(y_test, scores),
    }])
    report = classification_report(
        y_test, predictions, labels=[0, 1], target_names=["<=50K", ">50K"],
        output_dict=True, zero_division=0,
    )
    return metrics, report, predictions
