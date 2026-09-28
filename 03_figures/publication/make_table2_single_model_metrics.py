from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import fusion_results, publication_results

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score, roc_curve

from common.manuscript_bootstrap import (
    BOOTSTRAP_N,
    BOOTSTRAP_PROTOCOL,
    bootstrap_identity,
    canonical_rows,
    shared_bootstrap_indices,
    threshold_metric_values,
    verify_manuscript_threshold,
)
from common.manuscript_calibration import calibration_stats


DATASET = "A_holdout_clean"
PRIMARY_FUSION = "weighted_voting_cv_auc"
NATIVE_PREDICTIONS = fusion_results() / "native_predictions_all_eval.csv"
A_CV_FUSION_PREDICTIONS = fusion_results() / "A_fusions_CV_holdout" / "fusion_predictions_A_CV.csv"
TABLE_OUT = publication_results() / "tables" / "table2_model_performance_ci.csv"
CURVE_DIR = publication_results() / "curves"

SINGLE_MODELS = [
    "logistic_regression", "gam", "knn", "rbf_svm", "gaussian_nb", "decision_tree", "random_forest",
    "extra_trees", "gbdt", "xgboost", "lightgbm", "adaboost", "rotation_forest", "mlp",
]
DISPLAY_NAME = {
    "logistic_regression": "Logistic Regression", "gam": "GAM", "knn": "KNN", "rbf_svm": "RBF-SVM",
    "gaussian_nb": "Gaussian NB", "decision_tree": "Decision Tree", "random_forest": "Random Forest",
    "extra_trees": "Extra Trees", "gbdt": "GBDT", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "adaboost": "AdaBoost", "rotation_forest": "Rotation Forest", "mlp": "MLP",
}
THRESHOLD_METRICS = ("accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy")


def _threshold_values(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    values = threshold_metric_values(y, probability, threshold)
    values["balanced_accuracy"] = float((values["sensitivity"] + values["specificity"]) / 2.0)
    return values


def _probability_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
    }


def _interval(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    low, high = np.quantile(array, [0.025, 0.975])
    return float(low), float(high)


def load_single_model_predictions(path: Path = NATIVE_PREDICTIONS) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path)
    frame = frame.loc[(frame["dataset"] == DATASET) & frame["model"].isin(SINGLE_MODELS)].copy()
    missing = sorted(set(SINGLE_MODELS) - set(frame["model"]))
    if missing:
        raise RuntimeError(f"Single-model predictions missing for {DATASET}: {missing}")
    by_model = {model: canonical_rows(frame.loc[frame["model"] == model]) for model in SINGLE_MODELS}
    reference = by_model[SINGLE_MODELS[0]][["record_id", "y_SCLC"]]
    for model, group in by_model.items():
        if not group[["record_id", "y_SCLC"]].equals(reference):
            raise RuntimeError(f"Case or outcome mismatch for {model}")
        kinds = set(group["score_kind"])
        expected = {"decision_function"} if model == "rbf_svm" else {"probability"}
        if kinds != expected:
            raise RuntimeError(f"{model}: score_kind {kinds}, expected {expected}")
    return by_model


def development_threshold(strict: bool) -> dict[str, object]:
    frame = pd.read_csv(A_CV_FUSION_PREDICTIONS)
    primary = canonical_rows(frame.loc[frame["fusion"] == PRIMARY_FUSION])
    return verify_manuscript_threshold(
        primary["y_SCLC"].to_numpy(dtype=int), primary["score"].to_numpy(dtype=float), strict=strict
    )


def compute_table2(by_model: dict[str, pd.DataFrame], threshold: float) -> pd.DataFrame:
    reference = by_model[SINGLE_MODELS[0]]
    y = reference["y_SCLC"].to_numpy(dtype=int)
    record_ids = reference["record_id"].to_numpy(str)
    samples = shared_bootstrap_indices(y, DATASET, record_ids)
    rows = []
    for model in SINGLE_MODELS:
        group = by_model[model]
        raw = group["raw_score"].to_numpy(dtype=float)
        is_probability = model != "rbf_svm"
        points: dict[str, float] = {
            "auc": float(roc_auc_score(y, raw)),
            "auprc": float(average_precision_score(y, raw)),
        }
        row: dict[str, object] = {
            "dataset": DATASET,
            "model": model,
            "display_name": DISPLAY_NAME[model],
            "n": int(len(y)),
            "positive_n": int(y.sum()),
            "negative_n": int(len(y) - y.sum()),
            "score_kind": "probability" if is_probability else "decision_function",
            "threshold_used": threshold if is_probability else np.nan,
        }
        if is_probability:
            probability = group["native_probability"].to_numpy(dtype=float)
            points.update(_probability_values(y, probability))
            points.update(_threshold_values(y, probability, threshold))
            calibration = calibration_stats(y, probability)
            row.update(calibration)
            row["observed_expected_ratio"] = float(y.mean() / probability.mean())
        draws: dict[str, list[float]] = {name: [] for name in points}
        for index in samples:
            draws["auc"].append(roc_auc_score(y[index], raw[index]))
            draws["auprc"].append(average_precision_score(y[index], raw[index]))
            if is_probability:
                for name, value in {**_probability_values(y[index], probability[index]),
                                    **_threshold_values(y[index], probability[index], threshold)}.items():
                    draws[name].append(value)
        for name in ("auc", "auprc", "brier", "log_loss", *THRESHOLD_METRICS):
            if name in points:
                low, high = _interval(draws[name])
                row.update({name: points[name], f"{name}_ci_low": low, f"{name}_ci_high": high})
            else:
                row.update({name: np.nan, f"{name}_ci_low": np.nan, f"{name}_ci_high": np.nan})
        row.update({
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_protocol": BOOTSTRAP_PROTOCOL,
            "bootstrap_case_sha256": bootstrap_identity(y, record_ids),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def compute_curves(by_model: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    roc_rows, pr_rows = [], []
    for model in SINGLE_MODELS:
        group = by_model[model]
        y = group["y_SCLC"].to_numpy(dtype=int)
        raw = group["raw_score"].to_numpy(dtype=float)
        fpr, tpr, roc_thresholds = roc_curve(y, raw)
        roc_thresholds = np.where(np.isfinite(roc_thresholds), roc_thresholds, np.nan)
        for point, (x, t, cut) in enumerate(zip(fpr, tpr, roc_thresholds)):
            roc_rows.append({"dataset": DATASET, "model": model, "point": point, "fpr": x, "tpr": t, "threshold": cut})
        precision, recall, pr_thresholds = precision_recall_curve(y, raw)
        pr_thresholds = np.append(pr_thresholds, np.nan)
        for point, (p, r, cut) in enumerate(zip(precision, recall, pr_thresholds)):
            pr_rows.append({"dataset": DATASET, "model": model, "point": point, "precision": p, "recall": r, "threshold": cut})
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-threshold-mismatch", action="store_true")
    args = parser.parse_args()
    threshold_info = development_threshold(strict=not args.allow_threshold_mismatch)
    by_model = load_single_model_predictions()
    table = compute_table2(by_model, float(threshold_info["threshold"]))
    roc, pr = compute_curves(by_model)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_OUT, index=False, encoding="utf-8-sig", float_format="%.12g")
    roc.to_csv(CURVE_DIR / "roc_curve_points.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    pr.to_csv(CURVE_DIR / "pr_curve_points.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    print(f"Threshold {threshold_info['threshold']:.4f} (derived {threshold_info['derived_threshold']:.6f})")
    print(f"WROTE {TABLE_OUT}")


if __name__ == "__main__":
    main()
