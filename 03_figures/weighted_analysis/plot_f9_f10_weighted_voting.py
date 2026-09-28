from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import data_root, fusion_results, publication_results

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from statsmodels.nonparametric.smoothers_lowess import lowess

from nature_style import (
    DATASET_LABELS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_LINESTYLES,
    MODEL_ORDER,
    apply_style,
    panel_label,
)


ANALYSIS_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = ANALYSIS_DIR.parent
DATA_DIR = fusion_results() / "data"
TABLE_DIR = fusion_results() / "tables"
FIGURE_DIR = publication_results() / "figures" / "main"

INPUT_A_CV = data_root() / "fusion_results" / "A_fusions_CV_holdout" / "fusion_predictions_A_CV.csv"
INPUT_A_HOLDOUT_FUSION = data_root() / "fusion_results" / "A_fusions_CV_holdout" / "fusion_predictions_A_holdout_clean.csv"
INPUT_NATIVE = data_root() / "fusion_results" / "native_predictions_all_eval.csv"
INPUT_BC = data_root() / "fusion_results" / "three_fusions_BC" / "fusion_predictions_B_C.csv"

DATASET_ORDER = [
    "A_dev_cv_clean",
    "A_holdout_clean",
    "B_external",
    "C_external",
]
FUSION_MODELS = MODEL_ORDER[:3]
ALL_MODELS = MODEL_ORDER
PRIMARY_MODEL = "weighted_voting_cv_auc"
BASELINE_MODELS = MODEL_ORDER[3:]


_BOOTSTRAP_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "manuscript_bootstrap.py").is_file())
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))
from common.manuscript_bootstrap import BOOTSTRAP_PROTOCOL, bootstrap_seed, canonical_rows, shared_bootstrap_indices, bootstrap_identity, verify_manuscript_threshold
from common.manuscript_calibration import calibration_summary

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260911
LOWESS_FRAC = 0.75
LOWESS_IT = 0
LOWESS_LOWER_Q = 0.025
LOWESS_UPPER_Q = 0.975
CALIBRATION_BINS = 10
DCA_THRESHOLDS = np.linspace(0.01, 0.80, 160)
Z_975 = float(norm.ppf(0.975))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def _common_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    model_column: str,
    probability_column: str,
    models: Iterable[str],
) -> pd.DataFrame:
    models = list(models)
    part = frame.loc[
        (frame["dataset"].astype(str) == dataset)
        & frame[model_column].astype(str).isin(models)
    ].copy()
    if part.empty:
        raise RuntimeError(f"No prediction rows found for {dataset} in requested input.")
    part = part.rename(columns={model_column: "model", probability_column: "probability"})
    if "source_record_id" not in part.columns:
        part["source_record_id"] = part["record_id"].astype(str)
    keep = ["dataset", "model", "record_id", "source_record_id", "y_SCLC", "probability"]
    if "fold" in part.columns:
        keep.append("fold")
    part = part[keep]
    part["record_id"] = part["record_id"].astype(str)
    part["source_record_id"] = part["source_record_id"].astype(str)
    part["y_SCLC"] = pd.to_numeric(part["y_SCLC"], errors="raise").astype(int)
    part["probability"] = pd.to_numeric(part["probability"], errors="raise").astype(float)
    if "fold" in part.columns:
        part["fold"] = pd.to_numeric(part["fold"], errors="raise").astype(int)
    return part


def load_predictions() -> pd.DataFrame:
    for path in [INPUT_A_CV, INPUT_A_HOLDOUT_FUSION, INPUT_NATIVE, INPUT_BC]:
        if not path.exists():
            raise FileNotFoundError(path)

    a_cv_raw = pd.read_csv(INPUT_A_CV)
    a_holdout_fusion_raw = pd.read_csv(INPUT_A_HOLDOUT_FUSION)
    native_raw = pd.read_csv(INPUT_NATIVE)
    bc_raw = pd.read_csv(INPUT_BC)

    a_cv = _common_frame(
        a_cv_raw,
        dataset="A_dev_cv_clean",
        model_column="fusion",
        probability_column="score",
        models=FUSION_MODELS,
    )
    a_holdout_fusion = _common_frame(
        a_holdout_fusion_raw,
        dataset="A_holdout_clean",
        model_column="fusion",
        probability_column="score",
        models=FUSION_MODELS,
    )
    a_holdout_base = _common_frame(
        native_raw,
        dataset="A_holdout_clean",
        model_column="model",
        probability_column="native_probability",
        models=BASELINE_MODELS,
    )
    b = _common_frame(
        bc_raw,
        dataset="B_external",
        model_column="model",
        probability_column="native_probability",
        models=ALL_MODELS,
    )
    c = _common_frame(
        bc_raw,
        dataset="C_external",
        model_column="model",
        probability_column="native_probability",
        models=ALL_MODELS,
    )
    predictions = pd.concat([a_cv, a_holdout_fusion, a_holdout_base, b, c], ignore_index=True)
    validate_predictions(predictions)
    return predictions


def expected_models(dataset: str) -> list[str]:
    return FUSION_MODELS if dataset == "A_dev_cv_clean" else ALL_MODELS


def validate_predictions(predictions: pd.DataFrame) -> None:
    if predictions[["dataset", "model", "record_id", "y_SCLC", "probability"]].isna().any().any():
        raise RuntimeError("Selected predictions contain missing required values.")
    if set(predictions["dataset"].unique()) != set(DATASET_ORDER):
        raise RuntimeError("Selected prediction datasets do not match the frozen four-dataset scope.")
    if not set(predictions["y_SCLC"].unique()).issubset({0, 1}):
        raise RuntimeError("Outcome is not binary 0/1.")
    if not np.isfinite(predictions["probability"].to_numpy(float)).all():
        raise RuntimeError("Selected predictions contain non-finite probabilities.")
    if not predictions["probability"].between(0.0, 1.0, inclusive="both").all():
        raise RuntimeError("Selected probabilities fall outside [0, 1].")

    for dataset in DATASET_ORDER:
        subset = predictions.loc[predictions["dataset"] == dataset]
        models = expected_models(dataset)
        if set(subset["model"].drop_duplicates()) != set(models):
            present = subset["model"].drop_duplicates().tolist()
            raise RuntimeError(f"Frozen model order mismatch for {dataset}: {present}")
        primary = (
            subset.loc[subset["model"] == PRIMARY_MODEL, ["record_id", "source_record_id", "y_SCLC"]]
            .sort_values("record_id")
            .reset_index(drop=True)
        )
        if primary["record_id"].duplicated().any():
            raise RuntimeError(f"Duplicate record_id values in {dataset}/{PRIMARY_MODEL}.")
        if primary["y_SCLC"].nunique() != 2:
            raise RuntimeError(f"Both outcome classes are not present in {dataset}.")
        for model in models:
            current = (
                subset.loc[subset["model"] == model, ["record_id", "source_record_id", "y_SCLC"]]
                .sort_values("record_id")
                .reset_index(drop=True)
            )
            if current["record_id"].duplicated().any():
                raise RuntimeError(f"Duplicate record_id values in {dataset}/{model}.")
            if not current[["record_id", "y_SCLC"]].equals(primary[["record_id", "y_SCLC"]]):
                raise RuntimeError(f"record_id/outcome mismatch in {dataset}/{model}.")
        if dataset == "A_dev_cv_clean":
            for model in models:
                folds = subset.loc[subset["model"] == model, "fold"]
                if set(folds.unique()) != set(range(1, 11)):
                    raise RuntimeError(f"A-development fold identifiers are not 1-10 for {model}.")


def dataset_wide(predictions: pd.DataFrame, dataset: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    subset = predictions.loc[predictions["dataset"] == dataset].copy()
    base = (
        subset.loc[subset["model"] == PRIMARY_MODEL, ["record_id", "y_SCLC"]]
        .sort_values("record_id")
        .reset_index(drop=True)
    )
    model_scores: dict[str, np.ndarray] = {}
    for model in expected_models(dataset):
        current = (
            subset.loc[subset["model"] == model, ["record_id", "probability"]]
            .sort_values("record_id")
            .reset_index(drop=True)
        )
        if not current["record_id"].equals(base["record_id"]):
            raise RuntimeError(f"Alignment failure for {dataset}/{model}.")
        model_scores[model] = current["probability"].to_numpy(float)
    return base["y_SCLC"].to_numpy(int), model_scores, base["record_id"].to_numpy(str)


def stratified_bootstrap_indices(y: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    y = np.asarray(y, dtype=int)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    if len(negative) == 0 or len(positive) == 0:
        raise RuntimeError("Stratified bootstrap requires both outcome classes.")
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        samples.append(idx)
    return samples


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def compute_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        y, model_scores, record_ids = dataset_wide(predictions, dataset)
        samples = shared_bootstrap_indices(y, dataset, record_ids)
        for model in expected_models(dataset):
            score = model_scores[model]
            boot_auc = np.empty(BOOTSTRAP_N, dtype=float)
            boot_auprc = np.empty(BOOTSTRAP_N, dtype=float)
            boot_brier = np.empty(BOOTSTRAP_N, dtype=float)
            for i, idx in enumerate(samples):
                boot_auc[i] = roc_auc_score(y[idx], score[idx])
                boot_auprc[i] = average_precision_score(y[idx], score[idx])
                boot_brier[i] = brier_score_loss(y[idx], score[idx])
            auc_ci = percentile_interval(boot_auc)
            auprc_ci = percentile_interval(boot_auprc)
            brier_ci = percentile_interval(boot_brier)
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "role": "Primary model" if model == PRIMARY_MODEL else ("Fusion comparator" if model in FUSION_MODELS else "Baseline single model"),
                    "n": int(len(y)),
                    "events": int(y.sum()),
                    "non_events": int((y == 0).sum()),
                    "prevalence": float(y.mean()),
                    "auc": float(roc_auc_score(y, score)),
                    "auc_ci_low": auc_ci[0],
                    "auc_ci_high": auc_ci[1],
                    "auprc": float(average_precision_score(y, score)),
                    "auprc_ci_low": auprc_ci[0],
                    "auprc_ci_high": auprc_ci[1],
                    "brier": float(brier_score_loss(y, score)),
                    "brier_ci_low": brier_ci[0],
                    "brier_ci_high": brier_ci[1],
                    "bootstrap_n": BOOTSTRAP_N,
                    "bootstrap_protocol": BOOTSTRAP_PROTOCOL,
                    "bootstrap_case_sha256": bootstrap_identity(y, record_ids),
                    "bootstrap_method": "Outcome-stratified row bootstrap percentile interval",
                    "evaluation_estimand": (
                        "Pooled out-of-fold patient-level predictions from 10-fold CV"
                        if dataset == "A_dev_cv_clean"
                        else "Patient-level predictions in the frozen evaluation dataset"
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_youden_threshold(predictions: pd.DataFrame, *, strict: bool = True) -> dict[str, object]:
    y, model_scores, _ = dataset_wide(predictions, "A_dev_cv_clean")
    return verify_manuscript_threshold(y, model_scores[PRIMARY_MODEL], strict=strict)


def threshold_metric_values(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()

    def ratio(num: float, den: float) -> float:
        return float(num / den) if den else float("nan")

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    ppv = ratio(tp, tp + fp)
    npv = ratio(tn, tn + fn)
    accuracy = ratio(tp + tn, len(y))
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "f1": f1,
    }


def load_all_probability_predictions(plot_predictions: pd.DataFrame) -> pd.DataFrame:
    native_members = ["logistic_regression", "gam", "knn", "gaussian_nb", "decision_tree", "random_forest", "extra_trees", "gbdt", "xgboost", "lightgbm", "adaboost", "rotation_forest", "mlp"]
    rows = [plot_predictions.loc[plot_predictions["model"].isin(FUSION_MODELS)].copy()]
    native = pd.read_csv(INPUT_NATIVE)
    external = pd.read_csv(INPUT_BC)
    for dataset, frame in [("A_holdout_clean", native), ("B_external", external), ("C_external", external)]:
        part = _common_frame(frame, dataset=dataset, model_column="model", probability_column="native_probability", models=native_members)
        if set(part["model"]) != set(native_members):
            raise RuntimeError(f"Regenerate all 13 native-probability models for {dataset} before threshold reporting")
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def compute_all_threshold_metrics(predictions: pd.DataFrame, threshold: float, max_j: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        primary = canonical_rows(predictions.loc[(predictions["dataset"] == dataset) & (predictions["model"] == PRIMARY_MODEL)])
        y = primary["y_SCLC"].to_numpy(int)
        record_ids = primary["record_id"].to_numpy(str)
        model_scores = {}
        for name, group in predictions.loc[predictions["dataset"] == dataset].groupby("model"):
            group = canonical_rows(group)
            if not group[["record_id", "y_SCLC"]].equals(primary[["record_id", "y_SCLC"]]):
                raise RuntimeError(f"Threshold comparison case mismatch: {dataset}/{name}")
            model_scores[name] = group["probability"].to_numpy(float)
        for model in model_scores:
            score = model_scores[model]
            points = threshold_metric_values(y, score, threshold)
            samples = shared_bootstrap_indices(y, dataset, record_ids)
            boot = {metric: np.empty(BOOTSTRAP_N, dtype=float) for metric in points}
            for i, idx in enumerate(samples):
                values = threshold_metric_values(y[idx], score[idx], threshold)
                for metric, value in values.items():
                    boot[metric][i] = value
            row: dict[str, object] = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "n": int(len(y)),
                "events": int(y.sum()),
                "threshold": threshold,
                "threshold_source": "development CV Youden maximum (weighted voting)",
                "development_youden_j": max_j,
                "bootstrap_n": BOOTSTRAP_N,
                        "bootstrap_protocol": BOOTSTRAP_PROTOCOL,
                        "bootstrap_case_sha256": bootstrap_identity(y, record_ids),
                "bootstrap_method": "Outcome-stratified row bootstrap percentile interval; threshold fixed",
            }
            for metric, value in points.items():
                ci = percentile_interval(boot[metric])
                row[metric] = value
                row[f"{metric}_ci_low"] = ci[0]
                row[f"{metric}_ci_high"] = ci[1]
            rows.append(row)
    return pd.DataFrame(rows)


def compute_primary_threshold_metrics(predictions: pd.DataFrame, threshold: float, max_j: float) -> pd.DataFrame:
    all_metrics = compute_all_threshold_metrics(predictions, threshold, max_j)
    return all_metrics.loc[all_metrics["model"] == PRIMARY_MODEL].reset_index(drop=True)


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    midranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        midranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    result = np.empty(len(x), dtype=float)
    result[order] = midranks
    return result


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    if m < 2 or n < 2:
        raise RuntimeError("DeLong comparison requires at least two observations in each class.")
    positives = predictions_sorted_transposed[:, :m]
    negatives = predictions_sorted_transposed[:, m:]
    tx = np.array([compute_midrank(row) for row in positives])
    ty = np.array([compute_midrank(row) for row in negatives])
    tz = np.array([compute_midrank(row) for row in predictions_sorted_transposed])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.asarray(covariance, dtype=float)


def delong_primary_minus_comparator(
    y: np.ndarray, primary: np.ndarray, comparator: np.ndarray
) -> dict[str, float]:
    order = np.argsort(-y, kind="mergesort")
    aucs, covariance = fast_delong(np.vstack([primary, comparator])[:, order], int(y.sum()))
    contrast = np.array([1.0, -1.0])
    delta = float(aucs[0] - aucs[1])
    variance = float(contrast @ covariance @ contrast)
    variance = max(variance, 0.0)
    se = math.sqrt(variance)
    if se > 0:
        z = delta / se
        p_value = float(2.0 * norm.sf(abs(z)))
        ci_low = max(-1.0, delta - Z_975 * se)
        ci_high = min(1.0, delta + Z_975 * se)
    else:
        z = 0.0 if np.isclose(delta, 0.0) else math.copysign(float("inf"), delta)
        p_value = 1.0 if np.isclose(delta, 0.0) else 0.0
        ci_low = delta
        ci_high = delta
    return {
        "auc_reference": float(aucs[0]),
        "auc_comparator": float(aucs[1]),
        "delta_auc": delta,
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "standard_error": float(se),
        "z": float(z),
        "p_raw": p_value,
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def compute_delong_holm(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        y, model_scores, record_ids = dataset_wide(predictions, dataset)
        comparators = [model for model in expected_models(dataset) if model != PRIMARY_MODEL]
        family: list[dict[str, object]] = []
        for comparator in comparators:
            result = delong_primary_minus_comparator(y, model_scores[PRIMARY_MODEL], model_scores[comparator])
            family.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "reference_model": PRIMARY_MODEL,
                    "reference_label": MODEL_LABELS[PRIMARY_MODEL],
                    "comparator_model": comparator,
                    "comparator_label": MODEL_LABELS[comparator],
                    "direction": "Weighted voting minus comparator",
                    **result,
                    "holm_family": f"{dataset}: primary versus frozen displayed comparators",
                    "holm_family_n": len(comparators),
                }
            )
        adjusted = holm_adjust(np.array([float(row["p_raw"]) for row in family]))
        for row, p_holm in zip(family, adjusted):
            row["p_holm"] = float(p_holm)
            row["holm_significant_0_05"] = bool(p_holm < 0.05)
            rows.append(row)
    return pd.DataFrame(rows)


def equal_frequency_bins(y: np.ndarray, probability: np.ndarray, n_bins: int) -> pd.DataFrame:
    order = np.argsort(probability, kind="mergesort")
    groups = np.array_split(order, n_bins)
    rows = []
    for index, group in enumerate(groups, start=1):
        if len(group) == 0:
            continue
        rows.append(
            {
                "bin": index,
                "mean_predicted": float(np.mean(probability[group])),
                "observed_fraction": float(np.mean(y[group])),
                "n_bin": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def calibration_lowess(y: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    lower, upper = np.quantile(probability, [LOWESS_LOWER_Q, LOWESS_UPPER_Q])
    keep = (probability >= lower) & (probability <= upper)
    if keep.sum() < 5 or np.isclose(lower, upper):
        raise RuntimeError("Insufficient central prediction range for LOWESS calibration.")
    smoothed = lowess(
        endog=y[keep].astype(float),
        exog=probability[keep].astype(float),
        frac=LOWESS_FRAC,
        it=LOWESS_IT,
        delta=0.0,
        is_sorted=False,
        return_sorted=True,
    )
    lowess_frame = pd.DataFrame({"x": smoothed[:, 0], "y": smoothed[:, 1]}).groupby("x", as_index=False)["y"].mean()
    dense_x = np.linspace(float(lowess_frame["x"].min()), float(lowess_frame["x"].max()), 300)
    dense_y = np.interp(dense_x, lowess_frame["x"].to_numpy(float), lowess_frame["y"].to_numpy(float))
    return dense_x, np.clip(dense_y, 0.0, 1.0), float(lower), float(upper)


def build_curve_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roc_rows: list[dict[str, object]] = []
    pr_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    dca_rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        y, model_scores, record_ids = dataset_wide(predictions, dataset)
        prevalence = float(y.mean())
        for model in expected_models(dataset):
            probability = model_scores[model]
            fpr, tpr, thresholds = roc_curve(y, probability)
            for point, (x, value, threshold) in enumerate(zip(fpr, tpr, thresholds)):
                roc_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "point": point,
                        "fpr": float(x),
                        "tpr": float(value),
                        "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
                    }
                )
            precision, recall, thresholds_pr = precision_recall_curve(y, probability)
            for point, (x, value) in enumerate(zip(recall, precision)):
                pr_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "point": point,
                        "recall": float(x),
                        "precision": float(value),
                        "threshold": float(thresholds_pr[point]) if point < len(thresholds_pr) else np.nan,
                    }
                )
            lowess_x, lowess_y, lower, upper = calibration_lowess(y, probability)
            for point, (x, value) in enumerate(zip(lowess_x, lowess_y)):
                calibration_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "point_kind": "LOWESS",
                        "point": point,
                        "mean_predicted": float(x),
                        "observed_fraction": float(value),
                        "n_bin": np.nan,
                        "lowess_range_low": lower,
                        "lowess_range_high": upper,
                        "lowess_frac": LOWESS_FRAC,
                        "lowess_it": LOWESS_IT,
                    }
                )
            bins = equal_frequency_bins(y, probability, CALIBRATION_BINS)
            for _, bin_row in bins.iterrows():
                calibration_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "point_kind": "equal-frequency bin",
                        "point": int(bin_row["bin"]),
                        "mean_predicted": float(bin_row["mean_predicted"]),
                        "observed_fraction": float(bin_row["observed_fraction"]),
                        "n_bin": int(bin_row["n_bin"]),
                        "lowess_range_low": lower,
                        "lowess_range_high": upper,
                        "lowess_frac": LOWESS_FRAC,
                        "lowess_it": LOWESS_IT,
                    }
                )
            for threshold in DCA_THRESHOLDS:
                predicted = probability >= threshold
                tp = float(np.sum(predicted & (y == 1)))
                fp = float(np.sum(predicted & (y == 0)))
                net_benefit = tp / len(y) - fp / len(y) * threshold / (1.0 - threshold)
                dca_rows.append(
                    {
                        "dataset": dataset,
                        "curve_type": "model",
                        "model": model,
                        "threshold": float(threshold),
                        "net_benefit": float(net_benefit),
                        "threshold_grid_n": len(DCA_THRESHOLDS),
                    }
                )
        for threshold in DCA_THRESHOLDS:
            treat_all = prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
            dca_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "curve_type": "treat_all",
                        "model": "treat_all",
                        "threshold": float(threshold),
                        "net_benefit": float(treat_all),
                        "threshold_grid_n": len(DCA_THRESHOLDS),
                    },
                    {
                        "dataset": dataset,
                        "curve_type": "treat_none",
                        "model": "treat_none",
                        "threshold": float(threshold),
                        "net_benefit": 0.0,
                        "threshold_grid_n": len(DCA_THRESHOLDS),
                    },
                ]
            )
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows), pd.DataFrame(calibration_rows), pd.DataFrame(dca_rows)


def metric_lookup(metrics: pd.DataFrame, dataset: str, model: str) -> pd.Series:
    selected = metrics.loc[(metrics["dataset"] == dataset) & (metrics["model"] == model)]
    if len(selected) != 1:
        raise RuntimeError(f"Metric lookup returned {len(selected)} rows for {dataset}/{model}.")
    return selected.iloc[0]


def legend_handle(model: str, linewidth: float = 1.6) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=MODEL_COLORS[model],
        linestyle=MODEL_LINESTYLES[model],
        linewidth=linewidth,
    )


def setup_plot_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(direction="out", pad=2)


def plot_roc(ax: plt.Axes, legend_ax: plt.Axes, dataset: str, models: list[str], roc_points: pd.DataFrame, metrics: pd.DataFrame) -> None:
    setup_plot_axis(ax)
    for model in models:
        curve = roc_points.loc[(roc_points["dataset"] == dataset) & (roc_points["model"] == model)].sort_values("point")
        ax.plot(curve["fpr"], curve["tpr"], color=MODEL_COLORS[model], linestyle=MODEL_LINESTYLES[model], linewidth=2.15 if model == PRIMARY_MODEL else 1.35, zorder=3 if model == PRIMARY_MODEL else 2)
    ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=0.9, zorder=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="1 - Specificity", ylabel="Sensitivity")
    handles, labels = [], []
    for model in models:
        row = metric_lookup(metrics, dataset, model)
        handles.append(legend_handle(model, 2.0 if model == PRIMARY_MODEL else 1.35))
        labels.append(f"{MODEL_LABELS[model]}: AUC {row.auc:.3f} [{row.auc_ci_low:.3f}-{row.auc_ci_high:.3f}]")
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=6.2, handlelength=2.4, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.24)
    legend_ax.text(0.0, 0.02, "95% CI: outcome-stratified bootstrap (2,000 resamples).", transform=legend_ax.transAxes, fontsize=6.2, va="bottom")


def plot_pr(ax: plt.Axes, legend_ax: plt.Axes, dataset: str, models: list[str], pr_points: pd.DataFrame, metrics: pd.DataFrame) -> None:
    setup_plot_axis(ax)
    prevalence = float(metric_lookup(metrics, dataset, PRIMARY_MODEL)["prevalence"])
    for model in models:
        curve = pr_points.loc[(pr_points["dataset"] == dataset) & (pr_points["model"] == model)].sort_values("point")
        ax.plot(curve["recall"], curve["precision"], color=MODEL_COLORS[model], linestyle=MODEL_LINESTYLES[model], linewidth=2.15 if model == PRIMARY_MODEL else 1.35, zorder=3 if model == PRIMARY_MODEL else 2)
    ax.axhline(prevalence, color="#777777", linestyle=":", linewidth=0.9, zorder=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    handles, labels = [], []
    for model in models:
        row = metric_lookup(metrics, dataset, model)
        handles.append(legend_handle(model, 2.0 if model == PRIMARY_MODEL else 1.35))
        labels.append(f"{MODEL_LABELS[model]}: AUPRC {row.auprc:.3f} [{row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f}]")
    handles.append(Line2D([0], [0], color="#777777", linestyle=":", linewidth=0.9))
    labels.append(f"Prevalence: {prevalence:.3f}")
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=6.2, handlelength=2.4, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.24)
    legend_ax.text(0.0, 0.02, "95% CI: outcome-stratified bootstrap (2,000 resamples).", transform=legend_ax.transAxes, fontsize=6.2, va="bottom")


def plot_calibration(ax: plt.Axes, legend_ax: plt.Axes, dataset: str, models: list[str], calibration_points: pd.DataFrame) -> None:
    setup_plot_axis(ax)
    ax.plot([0, 1], [0, 1], color="#666666", linestyle=":", linewidth=1.0, zorder=1)
    for model in models:
        curve = calibration_points.loc[(calibration_points["dataset"] == dataset) & (calibration_points["model"] == model) & (calibration_points["point_kind"] == "LOWESS")].sort_values("point")
        bins = calibration_points.loc[(calibration_points["dataset"] == dataset) & (calibration_points["model"] == model) & (calibration_points["point_kind"] == "equal-frequency bin")].sort_values("point")
        ax.plot(curve["mean_predicted"], curve["observed_fraction"], color=MODEL_COLORS[model], linestyle=MODEL_LINESTYLES[model], linewidth=2.15 if model == PRIMARY_MODEL else 1.35, zorder=3 if model == PRIMARY_MODEL else 2)
        ax.scatter(bins["mean_predicted"], bins["observed_fraction"], s=13 if model == PRIMARY_MODEL else 8, facecolor=MODEL_COLORS[model], edgecolor="white", linewidth=0.35, alpha=0.85, zorder=4)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed proportion")
    handles = [legend_handle(model, 2.0 if model == PRIMARY_MODEL else 1.35) for model in models]
    labels = [MODEL_LABELS[model] for model in models]
    handles.append(Line2D([0], [0], color="#666666", linestyle=":", linewidth=1.0))
    labels.append("Ideal")
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=6.2, handlelength=2.4, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.24)
    legend_ax.text(0.0, 0.02, "Local-linear LOWESS (tricube), frac=0.75, it=0,\ncentral 2.5%-97.5%; circles: 10 equal-frequency bins.", transform=legend_ax.transAxes, fontsize=6.2, va="bottom")


def plot_dca(ax: plt.Axes, legend_ax: plt.Axes, dataset: str, models: list[str], dca_points: pd.DataFrame, metrics: pd.DataFrame) -> None:
    setup_plot_axis(ax)
    for model in models:
        curve = dca_points.loc[(dca_points["dataset"] == dataset) & (dca_points["model"] == model)].sort_values("threshold")
        ax.plot(curve["threshold"], curve["net_benefit"], color=MODEL_COLORS[model], linestyle=MODEL_LINESTYLES[model], linewidth=2.15 if model == PRIMARY_MODEL else 1.35, zorder=3 if model == PRIMARY_MODEL else 2)
    treat_all = dca_points.loc[(dca_points["dataset"] == dataset) & (dca_points["curve_type"] == "treat_all")].sort_values("threshold")
    treat_none = dca_points.loc[(dca_points["dataset"] == dataset) & (dca_points["curve_type"] == "treat_none")].sort_values("threshold")
    ax.plot(treat_all["threshold"], treat_all["net_benefit"], color="#777777", linestyle="--", linewidth=1.0, zorder=1)
    ax.plot(treat_none["threshold"], treat_none["net_benefit"], color="#111111", linestyle=":", linewidth=1.0, zorder=1)
    prevalence = float(metric_lookup(metrics, dataset, PRIMARY_MODEL)["prevalence"])
    model_values = dca_points.loc[(dca_points["dataset"] == dataset) & (dca_points["curve_type"] == "model"), "net_benefit"].to_numpy(float)
    lower = min(-0.04, float(np.quantile(model_values, 0.04)) - 0.015)
    lower = max(lower, -0.18)
    upper = min(0.45, max(prevalence + 0.055, float(np.quantile(model_values, 0.99)) + 0.02))
    ax.set(xlim=(0.01, 0.80), ylim=(lower, upper), xlabel="Threshold probability", ylabel="Net benefit")
    handles = [legend_handle(model, 2.0 if model == PRIMARY_MODEL else 1.35) for model in models]
    labels = [MODEL_LABELS[model] for model in models]
    handles.extend(
        [
            Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.0),
            Line2D([0], [0], color="#111111", linestyle=":", linewidth=1.0),
        ]
    )
    labels.extend(["Treat all", "Treat none"])
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=6.2, handlelength=2.4, handletextpad=0.5, borderaxespad=0.0, labelspacing=0.24)
    legend_ax.text(0.0, 0.02, "160 fixed thresholds (0.01-0.80); no Youden line.", transform=legend_ax.transAxes, fontsize=6.2, va="bottom")


def plot_brier(ax: plt.Axes, legend_ax: plt.Axes, dataset: str, models: list[str], metrics: pd.DataFrame) -> None:
    setup_plot_axis(ax)
    rows = [metric_lookup(metrics, dataset, model) for model in models]
    values = np.array([float(row["brier"]) for row in rows])
    lows = np.array([float(row["brier_ci_low"]) for row in rows])
    highs = np.array([float(row["brier_ci_high"]) for row in rows])
    y_positions = np.arange(len(models))
    for y_position, model, value, low, high in zip(y_positions, models, values, lows, highs):
        ax.errorbar(value, y_position, xerr=np.array([[value - low], [high - value]]), fmt="o", color=MODEL_COLORS[model], ecolor=MODEL_COLORS[model], elinewidth=1.5 if model == PRIMARY_MODEL else 1.0, capsize=2.4, markersize=5.2 if model == PRIMARY_MODEL else 4.0, markeredgecolor="white", markeredgewidth=0.45, zorder=3)
    labels = [MODEL_LABELS[model] for model in models]
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    for tick, model in zip(ax.get_yticklabels(), models):
        tick.set_fontsize(6.2)
        if model == PRIMARY_MODEL:
            tick.set_fontweight("bold")
    padding = max(0.006, 0.12 * float(highs.max() - lows.min()))
    ax.set_xlim(max(0.0, float(lows.min() - padding)), min(1.0, float(highs.max() + padding)))
    ax.set_xlabel("Brier score")
    ax.set_ylabel("")
    legend_ax.axis("off")
    legend_ax.text(0.0, 0.82, "Point estimate and 95% CI", transform=legend_ax.transAxes, fontsize=6.2, fontweight="bold", va="top")
    legend_ax.text(0.0, 0.56, "Outcome-stratified bootstrap; 2,000 resamples.", transform=legend_ax.transAxes, fontsize=6.2, va="top", wrap=True)
    legend_ax.text(0.0, 0.28, "Lower values indicate better overall probabilistic accuracy.", transform=legend_ax.transAxes, fontsize=6.2, va="top", wrap=True)


def make_composite_figure(
    *,
    datasets: list[str],
    basename: str,
    roc_points: pd.DataFrame,
    pr_points: pd.DataFrame,
    calibration_points: pd.DataFrame,
    dca_points: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[Path, Path]:
    if len(datasets) != 2:
        raise ValueError("Composite F9/F10 figures require exactly two dataset rows.")
    fig = plt.figure(figsize=(17.6, 11.2), facecolor="white")
    grid = fig.add_gridspec(
        nrows=4,
        ncols=5,
        height_ratios=[4.4, 2.10, 4.4, 2.10],
        left=0.055,
        right=0.995,
        top=0.975,
        bottom=0.025,
        wspace=0.43,
        hspace=0.22,
    )
    metric_names = ["ROC", "PR", "Calibration", "DCA", "Brier"]
    plotters = [plot_roc, plot_pr, plot_calibration, plot_dca, plot_brier]
    letters = "ABCDEFGHIJ"
    for row_index, dataset in enumerate(datasets):
        plot_row = row_index * 2
        legend_row = plot_row + 1
        models = expected_models(dataset)
        for column, (metric_name, plotter) in enumerate(zip(metric_names, plotters)):
            ax = fig.add_subplot(grid[plot_row, column])
            legend_ax = fig.add_subplot(grid[legend_row, column])
            panel_label(ax, letters[row_index * 5 + column], x=-0.16, y=1.02, fontsize=12)
            if metric_name == "ROC":
                plotter(ax, legend_ax, dataset, models, roc_points, metrics)
            elif metric_name == "PR":
                plotter(ax, legend_ax, dataset, models, pr_points, metrics)
            elif metric_name == "Calibration":
                plotter(ax, legend_ax, dataset, models, calibration_points)
            elif metric_name == "DCA":
                plotter(ax, legend_ax, dataset, models, dca_points, metrics)
            else:
                plotter(ax, legend_ax, dataset, models, metrics)
        y_coordinate = 0.752 if row_index == 0 else 0.275
        fig.text(0.011, y_coordinate, DATASET_LABELS[dataset], rotation=90, ha="center", va="center", fontsize=10.2, fontweight="bold")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{basename}.pdf"
    png_path = FIGURE_DIR / f"{basename}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def build_figure_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = [
        (
            "F9",
            "unnumbered_A_model_comparison_not_manuscript_fig4",
            ["A_dev_cv_clean", "A_holdout_clean"],
        ),
        (
            "F10",
            "unnumbered_BC_six_model_comparison_not_manuscript_fig5",
            ["B_external", "C_external"],
        ),
    ]
    metrics = ["ROC", "PR", "Calibration", "DCA", "Brier"]
    letters = "ABCDEFGHIJ"
    sources_by_dataset = {
        "A_dev_cv_clean": str(INPUT_A_CV),
        "A_holdout_clean": f"{INPUT_A_HOLDOUT_FUSION}; {INPUT_NATIVE}",
        "B_external": str(INPUT_BC),
        "C_external": str(INPUT_BC),
    }
    for figure, basename, datasets in specifications:
        for row_index, dataset in enumerate(datasets):
            models = expected_models(dataset)
            for column, metric in enumerate(metrics):
                rows.append(
                    {
                        "figure": figure,
                        "panel": letters[row_index * 5 + column],
                        "row": row_index + 1,
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "metric": metric,
                        "models_in_display_order": " | ".join(models),
                        "primary_model": PRIMARY_MODEL,
                        "data_source": sources_by_dataset[dataset],
                        "deterministic_recalculation": "ROC/PR/calibration/DCA points; AUC/AUPRC/Brier bootstrap CI",
                        "output_png": str(FIGURE_DIR / f"{basename}.png"),
                        "output_pdf": str(FIGURE_DIR / f"{basename}.pdf"),
                    }
                )
    return pd.DataFrame(rows)


def write_run_manifest(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    threshold: float,
    max_j: float,
    outputs: list[Path],
) -> Path:
    input_paths = [INPUT_A_CV, INPUT_A_HOLDOUT_FUSION, INPUT_NATIVE, INPUT_BC]
    manifest = {
        "analysis": "F9-F10 Weighted-voting primary-model figure revision",
        "scope": "Deterministic analysis of saved patient-level predictions; no predictive model fitted",
        "model_selection_rule": "Highest A-centre development 10-fold CV performance among frozen candidate fusions",
        "primary_model": PRIMARY_MODEL,
        "fusion_models": FUSION_MODELS,
        "baseline_single_models": BASELINE_MODELS,
        "model_order": MODEL_ORDER,
        "datasets": DATASET_ORDER,
        "dataset_labels": {dataset: DATASET_LABELS[dataset] for dataset in DATASET_ORDER},
        "sample_sizes": {
            dataset: {
                "n": int(metric_lookup(metrics, dataset, PRIMARY_MODEL)["n"]),
                "events": int(metric_lookup(metrics, dataset, PRIMARY_MODEL)["events"]),
            }
            for dataset in DATASET_ORDER
        },
        "confidence_intervals": {
            "metrics": ["AUC", "AUPRC", "Brier score"],
            "method": "Outcome-stratified row bootstrap percentile interval",
            "resamples": BOOTSTRAP_N,
            "base_seed": BOOTSTRAP_SEED,
        },
        "A_development_display_estimand": (
            "Pooled out-of-fold patient-level AUC/AUPRC/Brier from the saved 10-fold CV predictions. "
            "This is distinct from, and does not overwrite, the previously stored unweighted mean of ten fold-specific AUCs."
        ),
        "calibration": {
            "method": "Local-linear LOWESS with tricube neighbourhood weights applied to individual binary outcomes",
            "frac": LOWESS_FRAC,
            "robust_iterations": LOWESS_IT,
            "display_prediction_range_quantiles": [LOWESS_LOWER_Q, LOWESS_UPPER_Q],
            "original_equal_frequency_points": CALIBRATION_BINS,
            "ideal_line_retained": True,
        },
        "dca": {
            "threshold_start": float(DCA_THRESHOLDS[0]),
            "threshold_stop": float(DCA_THRESHOLDS[-1]),
            "threshold_count": int(len(DCA_THRESHOLDS)),
            "treat_all": True,
            "treat_none": True,
            "youden_line_displayed": False,
        },
        "primary_threshold": {
            "source_dataset": "A_dev_cv_clean",
            "method": "Maximum Youden index on development CV OOF weighted-voting predictions, rounded to four decimals",
            "threshold": threshold,
            "youden_j": max_j,
            "applied_unchanged_to": DATASET_ORDER,
        },
        "delong": {
            "direction": "Weighted voting minus comparator",
            "holm_family_definition": "Within dataset; primary versus all frozen comparators shown in that dataset",
            "A_development_family_n": 2,
            "A_holdout_B_C_family_n": 5,
        },
        "input_files": [{"path": str(path), "sha256": sha256(path)} for path in input_paths],
        "selected_prediction_rows": int(len(predictions)),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "style": {
            "language": "English",
            "font_family": "Times New Roman",
            "minimum_legend_and_method_text_size_pt": 6.2,
            "pdf_fonttype": 42,
            "ps_fonttype": 42,
            "png_dpi": 600,
            "figure_title": False,
            "top_right_spines": False,
        },
        "outputs": [str(path) for path in outputs],
    }
    path = DATA_DIR / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def render_figures_only() -> None:
    apply_style()
    metrics_path = TABLE_DIR / "model_metrics_with_ci.csv"
    roc_path = DATA_DIR / "roc_curve_points.csv"
    pr_path = DATA_DIR / "pr_curve_points.csv"
    calibration_path = DATA_DIR / "calibration_curve_points.csv"
    dca_path = DATA_DIR / "dca_curve_points.csv"
    required = [metrics_path, roc_path, pr_path, calibration_path, dca_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Figure-only refresh is missing frozen inputs: {missing}")

    metrics = pd.read_csv(metrics_path)
    roc_points = pd.read_csv(roc_path)
    pr_points = pd.read_csv(pr_path)
    calibration_points = pd.read_csv(calibration_path)
    dca_points = pd.read_csv(dca_path)

    f9_png, f9_pdf = make_composite_figure(
        datasets=["A_dev_cv_clean", "A_holdout_clean"],
        basename="unnumbered_A_model_comparison_not_manuscript_fig4",
        roc_points=roc_points,
        pr_points=pr_points,
        calibration_points=calibration_points,
        dca_points=dca_points,
        metrics=metrics,
    )
    f10_png, f10_pdf = make_composite_figure(
        datasets=["B_external", "C_external"],
        basename="unnumbered_BC_six_model_comparison_not_manuscript_fig5",
        roc_points=roc_points,
        pr_points=pr_points,
        calibration_points=calibration_points,
        dca_points=dca_points,
        metrics=metrics,
    )

    figure_files = {
        "F9": {"png": f9_png, "pdf": f9_pdf},
        "F10": {"png": f10_png, "pdf": f10_pdf},
    }
    figure_manifest = build_figure_manifest()
    figure_manifest["minimum_legend_and_method_text_size_pt"] = 6.2
    figure_manifest["png_sha256"] = figure_manifest["figure"].map(
        {figure: sha256(paths["png"]) for figure, paths in figure_files.items()}
    )
    figure_manifest["pdf_sha256"] = figure_manifest["figure"].map(
        {figure: sha256(paths["pdf"]) for figure, paths in figure_files.items()}
    )
    figure_manifest_path = TABLE_DIR / "F9_F10_manifest.csv"
    save_csv(figure_manifest, figure_manifest_path)

    caption_path = TABLE_DIR / "F9_F10_figure_captions.md"
    run_manifest_path = DATA_DIR / "run_manifest.json"
    if not caption_path.exists() or not run_manifest_path.exists():
        raise FileNotFoundError("Caption or existing run manifest is missing.")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8-sig"))
    run_manifest.setdefault("style", {})["minimum_legend_and_method_text_size_pt"] = 6.2
    run_manifest["figure_only_revision"] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Independent QC: raise all legend and method-note text to at least 6.2 pt",
        "statistics_recomputed": False,
        "predictions_or_curve_points_modified": False,
        "canvas_inches": [17.6, 11.2],
        "legend_row_height_ratio": 2.10,
    }
    run_manifest["figure_files"] = [
        {
            "figure": figure,
            "png": str(paths["png"]),
            "png_sha256": sha256(paths["png"]),
            "pdf": str(paths["pdf"]),
            "pdf_sha256": sha256(paths["pdf"]),
        }
        for figure, paths in figure_files.items()
    ]
    run_manifest["caption_file"] = {"path": str(caption_path), "sha256": sha256(caption_path)}
    run_manifest["panel_manifest_file"] = {
        "path": str(figure_manifest_path),
        "sha256": sha256(figure_manifest_path),
    }
    run_manifest["figure_code_file"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__).resolve()),
    }
    for path in [caption_path, figure_manifest_path, f9_png, f9_pdf, f10_png, f10_pdf]:
        if str(path) not in run_manifest.setdefault("outputs", []):
            run_manifest["outputs"].append(str(path))
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Figure-only refresh completed; statistical and curve-point files were not rewritten.")
    print(f"F9: {f9_png} | {f9_pdf}")
    print(f"F10: {f10_png} | {f10_pdf}")
    print(f"Run manifest: {run_manifest_path}")


def main() -> None:
    apply_style()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    metrics = compute_model_metrics(predictions)
    threshold_info = compute_youden_threshold(predictions, strict="--allow-threshold-mismatch" not in sys.argv[1:])
    threshold, max_j = float(threshold_info["threshold"]), float(threshold_info["youden_j"])
    save_csv(pd.DataFrame([threshold_info]), TABLE_DIR / "decision_threshold_derivation.csv")
    all_probability_predictions = load_all_probability_predictions(predictions)
    all_threshold_metrics = compute_all_threshold_metrics(all_probability_predictions, threshold, max_j)
    calibration_metrics = calibration_summary(all_probability_predictions)
    calibration_values = calibration_metrics.drop(columns=["bootstrap_n", "bootstrap_protocol", "bootstrap_case_sha256"])
    metrics = metrics.merge(calibration_values, on=["dataset", "model"], validate="one_to_one")
    threshold_metrics = all_threshold_metrics.loc[all_threshold_metrics["model"] == PRIMARY_MODEL].reset_index(drop=True)
    delong = compute_delong_holm(predictions)
    roc_points, pr_points, calibration_points, dca_points = build_curve_tables(predictions)

    metrics_path = TABLE_DIR / "model_metrics_with_ci.csv"
    primary_path = TABLE_DIR / "primary_weighted_voting_summary.csv"
    threshold_path = TABLE_DIR / "primary_threshold_metrics.csv"
    delong_path = TABLE_DIR / "delong_holm_vs_weighted_voting.csv"
    dca_path = DATA_DIR / "dca_curve_points.csv"
    roc_path = DATA_DIR / "roc_curve_points.csv"
    pr_path = DATA_DIR / "pr_curve_points.csv"
    calibration_path = DATA_DIR / "calibration_curve_points.csv"
    manifest_path = TABLE_DIR / "F9_F10_manifest.csv"

    save_csv(metrics, metrics_path)
    primary_summary = metrics.loc[metrics["model"] == PRIMARY_MODEL].copy()
    primary_summary = primary_summary.merge(
        threshold_metrics.drop(columns=["dataset_label", "model", "model_label", "n", "events", "bootstrap_protocol", "bootstrap_case_sha256", "bootstrap_n", "bootstrap_method"]),
        on="dataset",
        how="left",
        validate="one_to_one",
    )
    save_csv(primary_summary, primary_path)
    save_csv(threshold_metrics, threshold_path)
    save_csv(all_threshold_metrics, TABLE_DIR / "all_probability_model_threshold_metrics.csv")
    save_csv(calibration_metrics, TABLE_DIR / "all_probability_model_calibration_metrics.csv")
    save_csv(delong, delong_path)
    save_csv(dca_points, dca_path)
    save_csv(roc_points, roc_path)
    save_csv(pr_points, pr_path)
    save_csv(calibration_points, calibration_path)
    figure_manifest = build_figure_manifest()
    save_csv(figure_manifest, manifest_path)

    f9_png, f9_pdf = make_composite_figure(
        datasets=["A_dev_cv_clean", "A_holdout_clean"],
        basename="unnumbered_A_model_comparison_not_manuscript_fig4",
        roc_points=roc_points,
        pr_points=pr_points,
        calibration_points=calibration_points,
        dca_points=dca_points,
        metrics=metrics,
    )
    f10_png, f10_pdf = make_composite_figure(
        datasets=["B_external", "C_external"],
        basename="unnumbered_BC_six_model_comparison_not_manuscript_fig5",
        roc_points=roc_points,
        pr_points=pr_points,
        calibration_points=calibration_points,
        dca_points=dca_points,
        metrics=metrics,
    )

    outputs = [
        TABLE_DIR / "all_probability_model_threshold_metrics.csv",
        TABLE_DIR / "all_probability_model_calibration_metrics.csv",
        metrics_path,
        primary_path,
        threshold_path,
        delong_path,
        dca_path,
        roc_path,
        pr_path,
        calibration_path,
        manifest_path,
        f9_png,
        f9_pdf,
        f10_png,
        f10_pdf,
    ]
    run_manifest = write_run_manifest(predictions, metrics, threshold, max_j, outputs)
    print(f"Validated patient-level prediction rows: {len(predictions)}")
    print(f"Weighted-voting development CV Youden threshold: {threshold:.4f} (derived {threshold_info['derived_threshold']:.6f}, J={max_j:.4f})")
    print(f"Metric rows: {len(metrics)}; DeLong comparisons: {len(delong)}")
    print(f"F9: {f9_png} | {f9_pdf}")
    print(f"F10: {f10_png} | {f10_pdf}")
    print(f"Manifest: {run_manifest}")


if __name__ == "__main__":
    if "--figures-only" in sys.argv[1:]:
        render_figures_only()
    else:
        main()
