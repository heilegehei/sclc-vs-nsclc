from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, data_root


import sys as _cohort_sys
from pathlib import Path as _CohortPath
_cohort_root = next(p for p in _CohortPath(__file__).resolve().parents if (p / "common" / "manuscript_cohorts.py").is_file())
if str(_cohort_root) not in _cohort_sys.path:
    _cohort_sys.path.insert(0, str(_cohort_root))
from common.manuscript_cohorts import split_center_a_indices, validate_cohort, validate_base_cohorts, split_metadata, require_split_design

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = data_root()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


_ARCHIVE_MODEL_SRC = Path(__file__).resolve().parents[2] / "01_methods" / "models" / "src"
sys.path.insert(0, str(_ARCHIVE_MODEL_SRC))
from fold_selection import fit_scope_selection, SELECTION_PROTOCOL, require_selection_protocol

import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents
                     if (p / "common" / "manuscript_source_paths.py").is_file())
if str(_archive_root) not in _archive_sys.path:
    _archive_sys.path.insert(0, str(_archive_root))
from common.manuscript_source_paths import activate_archive_sources
activate_archive_sources(__file__)

from model_factory import MODEL_NAMES, build_model, derive_seed as model_seed
from stage4_pipeline import ALL_FEATURES, DIRECT_FEATURES, FrozenPreprocessor, load_base_workbook


MASTER_SEED = 20260902
CV_SEED = model_seed(MASTER_SEED, "10fold", "train80")
INPUT_WORKBOOK = cohort_workbook()
OUTPUT_DEFAULT = data_root() / "fusion_results" / "A_fusions_CV_holdout"
NATIVE_OUTPUT_DEFAULT = data_root() / "fusion_results" / "native_predictions_all_eval.csv"

EXPECTED_DIRECT = [
    "WBC", "PLT", "LDH", "PCT", "EO#", "EO%", "ALB", "TP", "GLB", "TC", "LDL-C", "MCH", "MCHC",
]
FIXED_COMPOSITES = ["SIRI", "LMR", "GAR", "PNI", "HALP"]
FUSION_NAMES = (
    "soft_voting_native_probability",
    "weighted_voting_cv_auc",
    "stacking_in_sample_A_train",
)
NATIVE_FUSION_MODELS = tuple(name for name in MODEL_NAMES if name != "rbf_svm")


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_safe(x) for x in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    text = frame.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _subset(frame: pd.DataFrame, indices: Sequence[int]) -> pd.DataFrame:
    return frame.iloc[np.asarray(indices, dtype=int)].reset_index(drop=True)


def _prepare_center(frame: pd.DataFrame, center: str) -> pd.DataFrame:
    out = frame.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    if set(out["center"].astype(str)) != {center}:
        raise ValueError(f"unexpected center in {center} frame")
    if "missing_direct_n" in out.columns:
        out = out.loc[out["missing_direct_n"].astype(float) < len(DIRECT_FEATURES)].copy()
    if len(out) < 100 or out["y_SCLC"].nunique() != 2:
        raise ValueError(f"unexpected {center} analysis cohort: n={len(out)}")
    return out.reset_index(drop=True)


def _score(estimator: Any, matrix: pd.DataFrame) -> tuple[np.ndarray, str, np.ndarray | None]:
    if hasattr(estimator, "predict_proba"):
        values = np.asarray(estimator.predict_proba(matrix), dtype=float)[:, 1]
        return values, "probability", np.clip(values, 0.0, 1.0)
    if hasattr(estimator, "decision_function"):
        values = np.asarray(estimator.decision_function(matrix), dtype=float).reshape(-1)
        return values, "decision_function", None
    values = np.asarray(estimator.predict(matrix), dtype=float).reshape(-1)
    return values, "predict", None


def _point_metrics(y: np.ndarray, score: np.ndarray, probability: np.ndarray | None) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {
        "n": int(len(y)),
        "positive_n": int(np.sum(y)),
        "negative_n": int(len(y) - np.sum(y)),
        "auc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }
    if probability is None:
        row.update({"brier": None, "log_loss": None})
    else:
        p = np.clip(probability, 1e-6, 1 - 1e-6)
        row.update({"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))})
    return row


def _fit_scope(
    train_frame: pd.DataFrame,
    apply_frame: pd.DataFrame,
    *,
    scope_id: str,
    seed_tag: str,
    model_names: Sequence[str],
    smoke: bool,
    selection_fold: int | None = None,
) -> dict[str, Any]:

    if not smoke and tuple(model_names) != tuple(MODEL_NAMES):
        raise RuntimeError("Formal analysis requires all 14 prespecified candidate models")
    y_train = train_frame["y_SCLC"].to_numpy(dtype=int)
    y_apply = apply_frame["y_SCLC"].to_numpy(dtype=int)
    pre = FrozenPreprocessor.fit(train_frame)
    x_train, train_qc = pre.transform(train_frame)
    x_apply, apply_qc = pre.transform(apply_frame)
    if list(x_train.columns) != list(ALL_FEATURES) or list(x_apply.columns) != list(ALL_FEATURES):
        raise RuntimeError("Stage-4 feature order mismatch")
    if not np.isfinite(x_train.to_numpy(dtype=float)).all() or not np.isfinite(x_apply.to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite Stage-4 values remain")

    selection = fit_scope_selection(x_train, y_train, selection_fold)

    fitted: dict[str, Any] = {}
    train_scores: dict[str, np.ndarray] = {}
    apply_scores: dict[str, np.ndarray] = {}
    train_probabilities: dict[str, np.ndarray | None] = {}
    apply_probabilities: dict[str, np.ndarray | None] = {}
    score_kinds: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    for model_name in model_names:
        try:
            params: dict[str, Any] | None = None
            if smoke and model_name in {"random_forest", "extra_trees", "xgboost", "lightgbm", "rotation_forest"}:
                params = {"n_estimators": 20}
            if smoke and model_name == "mlp":
                params = {"max_iter": 80}
            if smoke and model_name == "gam":
                params = {"max_iter": 50}
            estimator = build_model(
                model_name,
                seed=(model_seed(MASTER_SEED, "holdout_fit", "model", model_name) if selection_fold is None else model_seed(MASTER_SEED, "cv", int(selection_fold), "model", model_name)),
                selected_direct=selection["selected_direct"],
                scope_id=scope_id,
                params=params,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(x_train, y_train)
            train_score, train_kind, train_probability = _score(estimator, x_train)
            apply_score, apply_kind, apply_probability = _score(estimator, x_apply)
            if train_kind != apply_kind:
                raise RuntimeError(f"score kind changed between train/apply: {train_kind} vs {apply_kind}")
            fitted[model_name] = estimator
            train_scores[model_name] = train_score
            apply_scores[model_name] = apply_score
            train_probabilities[model_name] = train_probability
            apply_probabilities[model_name] = apply_probability
            score_kinds[model_name] = apply_kind
        except Exception as exc:
            errors.append({"scope": scope_id, "model": model_name, "error": f"{type(exc).__name__}: {exc}"})

    if not smoke and (errors or set(fitted) != set(MODEL_NAMES)):
        raise RuntimeError(f"Formal candidate fitting failed; no partial fusion is permitted: {errors}")
    native_members = [
        name for name in model_names
        if name in fitted and train_probabilities.get(name) is not None and apply_probabilities.get(name) is not None
    ]
    if len(native_members) < 2:
        raise RuntimeError(f"fewer than two native-probability members available in {scope_id}: {native_members}")
    return {
        "selection": selection,
        "smoke": smoke,
        "preprocessor": pre,
        "train_qc": train_qc,
        "apply_qc": apply_qc,
        "fitted": fitted,
        "train_scores": train_scores,
        "apply_scores": apply_scores,
        "train_probabilities": train_probabilities,
        "apply_probabilities": apply_probabilities,
        "score_kinds": score_kinds,
        "native_members": native_members,
        "errors": errors,
        "y_train": y_train,
        "y_apply": y_apply,
    }


def _normalize_auc_weights(auc_by_model, native_members):
    names = [name for name in native_members if name in auc_by_model and np.isfinite(auc_by_model[name]) and float(auc_by_model[name]) > 0]
    if len(names) < 2:
        raise RuntimeError("weighted voting needs at least two positive development CV AUCs from the same fit")
    values = np.asarray([float(auc_by_model[name]) for name in names], dtype=float)
    values = values / values.sum()
    return {name: float(value) for name, value in zip(names, values)}


def _fit_stacker(matrix, y, seed):
    stacker = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacker.fit(matrix, y)
    return stacker


def _member_matrix(probabilities, members):
    return np.column_stack([np.asarray(probabilities[name], dtype=float) for name in members])


def _apply_fixed_fusions(probabilities, members, weights, stacker):
    matrix = _member_matrix(probabilities, members)
    vector = np.asarray([weights[name] for name in members], dtype=float)
    return {
        FUSION_NAMES[0]: np.mean(matrix, axis=1),
        FUSION_NAMES[1]: np.sum(matrix * vector.reshape(1, -1), axis=1),
        FUSION_NAMES[2]: stacker.predict_proba(matrix)[:, 1],
    }


def _clean_folds(train: pd.DataFrame, n_folds: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    y = train["y_SCLC"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CV_SEED)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    assignment = np.full(len(train), -1, dtype=int)
    dummy = np.zeros((len(train), 1), dtype=np.uint8)
    for fold_number, (tr_i, va_i) in enumerate(splitter.split(dummy, y), start=1):
        folds.append((tr_i, va_i))
        assignment[va_i] = fold_number
    if (assignment < 0).any():
        raise RuntimeError("some development rows were not assigned to a CV fold")
    return folds, assignment


def _run_cv(
    train: pd.DataFrame,
    *,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    dataset_name: str,
    model_names: Sequence[str],
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    stored: list[dict[str, Any]] = []
    effective_folds = len(folds)
    for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
        if set(np.asarray(train_idx, dtype=int)) & set(np.asarray(valid_idx, dtype=int)):
            raise RuntimeError(f"fold {fold_number} training and validation indices overlap")
        fold_train = _subset(train, train_idx)
        fold_valid = _subset(train, valid_idx)
        if set(fold_train["record_id"].astype(str)) & set(fold_valid["record_id"].astype(str)):
            raise RuntimeError(f"fold {fold_number} training and validation record ids overlap")
        scope_id = f"A_dev_{dataset_name}_fold_{fold_number}"
        started = time.perf_counter()
        fit = _fit_scope(
            fold_train,
            fold_valid,
            scope_id=scope_id,
            seed_tag=dataset_name,
            model_names=model_names,
            smoke=smoke,
            selection_fold=fold_number,
        )
        members = list(fit["native_members"])
        if not smoke and tuple(members) != NATIVE_FUSION_MODELS:
            raise RuntimeError("Formal fusion requires all 13 native-probability members")
        stored.append({
            "fold": int(fold_number),
            "started": started,
            "train_n": int(len(fold_train)),
            "record_id": fold_valid["record_id"].astype(str).tolist(),
            "y": np.asarray(fit["y_apply"], dtype=int),
            "y_train": np.asarray(fit["y_train"], dtype=int),
            "members": members,
            "probabilities": {name: np.asarray(fit["apply_probabilities"][name], dtype=float) for name in members},
            "train_probabilities": {name: np.asarray(fit["train_probabilities"][name], dtype=float) for name in members},
            "fit": fit,
        })
        errors.extend([{"fold": fold_number, **item} for item in fit["errors"]])
    if not stored:
        raise RuntimeError("no CV folds were stored")
    members = list(stored[0]["members"])
    auc_by_model = {}
    for name in members:
        auc_by_model[name] = float(np.mean([roc_auc_score(item["y"], item["probabilities"][name]) for item in stored]))
    weights = _normalize_auc_weights(auc_by_model, members)
    members = [name for name in members if name in weights]
    for item in stored:
        train_matrix = _member_matrix(item["train_probabilities"], members)
        stacker = _fit_stacker(train_matrix, item["y_train"], model_seed(MASTER_SEED, "stacker", "in_sample", item["fold"]))
        fused_scores = _apply_fixed_fusions(item["probabilities"], members, weights, stacker)
        for fusion_name, score in fused_scores.items():
            metrics = _point_metrics(item["y"], score, score)
            fold_rows.append({
                "dataset": dataset_name,
                "fold": item["fold"],
                "fusion": fusion_name,
                "train_n": item["train_n"],
                "validation_n": int(len(item["y"])),
                "validation_positive_n": int(item["y"].sum()),
                "auc": metrics["auc"],
                "auprc": metrics["auprc"],
                "brier": metrics["brier"],
                "log_loss": metrics["log_loss"],
                "native_member_n": int(len(members)),
                "runtime_seconds": float(time.perf_counter() - item["started"]),
            })
            for record_id, y_value, value in zip(item["record_id"], item["y"], score):
                prediction_rows.append({
                    "selection_protocol": SELECTION_PROTOCOL if not smoke else "non-manuscript-run",
                    "dataset": dataset_name,
                    "fold": item["fold"],
                    "fusion": fusion_name,
                    "record_id": record_id,
                    "y_SCLC": int(y_value),
                    "score": float(value),
                })
        provenance.append({
            "dataset": dataset_name,
            "fold": item["fold"],
            "preprocessor_fingerprint": item["fit"]["preprocessor"].fingerprint,
            "train_qc": item["fit"]["train_qc"],
            "validation_qc": item["fit"]["apply_qc"],
            "native_members": members,
            "weights": weights,
            "model_errors": item["fit"]["errors"],
        })
    fold_df = pd.DataFrame(fold_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    return fold_df, prediction_df, errors, {
        "fold_provenance": provenance,
        "effective_folds": effective_folds,
        "weights": weights,
        "member_auc": auc_by_model,
        "native_members": members,
        "stacker_training": "in-sample probabilities from the same training rows used to fit the base learners",
    }


def _summarize_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (dataset, fusion), group in fold_df.groupby(["dataset", "fusion"], sort=False):
        values = group["auc"].to_numpy(dtype=float)
        rows.append({
            "dataset": str(dataset),
            "fusion": str(fusion),
            "folds_completed": int(len(values)),
            "cv_auc_mean": float(np.mean(values)),
            "cv_auc_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "cv_auc_min": float(np.min(values)),
            "cv_auc_max": float(np.max(values)),
            "cv_auprc_mean": float(group["auprc"].mean()),
            "cv_brier_mean": float(group["brier"].mean()),
            "cv_log_loss_mean": float(group["log_loss"].mean()),
            "validation_n_total": int(group["validation_n"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "fusion"]).reset_index(drop=True)


def _holdout_fusions(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    model_names: Sequence[str],
    smoke: bool,
    weights: Mapping[str, float],
    native_members: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fit = _fit_scope(
        train,
        holdout,
        scope_id="A_holdout_fit",
        seed_tag="holdout",
        model_names=model_names,
        smoke=smoke,
    )
    stacker = _fit_stacker(
        _member_matrix(fit["train_probabilities"], native_members),
        fit["y_train"],
        model_seed(MASTER_SEED, "stacker", "development_in_sample"),
    )
    scores = _apply_fixed_fusions(fit["apply_probabilities"], native_members, weights, stacker)
    info = {"native_members": list(native_members), "weights": dict(weights)}
    source_ids = (holdout["source_record_id"].astype(str) if "source_record_id" in holdout.columns
                  else holdout["record_id"].astype(str)).tolist()
    native_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        probability = fit["apply_probabilities"].get(model_name)
        for index, (record_id, y_value, raw_value) in enumerate(
            zip(holdout["record_id"].astype(str), fit["y_apply"], fit["apply_scores"][model_name])
        ):
            native_rows.append({
                "selection_protocol": SELECTION_PROTOCOL if not smoke else "non-manuscript-run",
                "dataset": "A_holdout_clean",
                "model": model_name,
                "record_id": record_id,
                "source_record_id": source_ids[index],
                "y_SCLC": int(y_value),
                "raw_score": float(raw_value),
                "native_probability": float(probability[index]) if probability is not None else None,
                "score_kind": fit["score_kinds"][model_name],
            })
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fusion_name, score in scores.items():
        metrics = _point_metrics(fit["y_apply"], score, score)
        rows.append({
            "dataset": "A_holdout_clean",
            "fusion": fusion_name,
            **metrics,
            "native_member_n": int(len(info["native_members"])),
            "member_names": "|".join(info["native_members"]),
        })
        for record_id, y_value, value in zip(holdout["record_id"].astype(str), fit["y_apply"], score):
            predictions.append({
                "selection_protocol": SELECTION_PROTOCOL if not smoke else "non-manuscript-run",
                "dataset": "A_holdout_clean",
                "fusion": fusion_name,
                "record_id": record_id,
                "y_SCLC": int(y_value),
                "score": float(value),
            })
    return pd.DataFrame(rows), pd.DataFrame(predictions), pd.DataFrame(native_rows), {
        "preprocessor_fingerprint": fit["preprocessor"].fingerprint,
        "train_qc": fit["train_qc"],
        "holdout_qc": fit["apply_qc"],
        "native_members": info["native_members"],
        "weights": info["weights"],
        "model_errors": fit["errors"],
    }


def run(
    *,
    output_dir: Path,
    input_workbook: Path = INPUT_WORKBOOK,
    n_folds: int = 10,
    fold_limit: int | None = None,
    smoke: bool = False,
    native_output: Path = NATIVE_OUTPUT_DEFAULT,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_direct = list(EXPECTED_DIRECT)
    selected_composites = list(FIXED_COMPOSITES)
    data = validate_base_cohorts(load_base_workbook(input_workbook))
    a = _prepare_center(data["A"], "A")
    all_indices = np.arange(len(a), dtype=int)
    train_indices, holdout_indices = split_center_a_indices(a)
    train_indices = np.sort(train_indices)
    holdout_indices = np.sort(holdout_indices)
    train = a.iloc[train_indices].reset_index(drop=True)
    holdout = a.iloc[holdout_indices].reset_index(drop=True)
    if set(train["record_id"].astype(str)) & set(holdout["record_id"].astype(str)):
        raise RuntimeError("A development and holdout record ids overlap")

    requested_folds = int(n_folds)
    if requested_folds < 2 or requested_folds > int(np.bincount(train["y_SCLC"].to_numpy(dtype=int)).min()):
        raise ValueError("n_folds is incompatible with the A-development class counts")
    folds, _fold_assignment = _clean_folds(train, requested_folds)
    effective_limit = requested_folds if fold_limit is None else max(1, min(int(fold_limit), requested_folds))
    folds = folds[:effective_limit]
    model_names = list(MODEL_NAMES[:4]) if smoke else list(MODEL_NAMES)
    if smoke:
        model_names = ["logistic_regression", "random_forest", "extra_trees", "gbdt"]

    clean_fold_df, clean_pred_df, clean_errors, clean_provenance = _run_cv(
        train,
        folds=folds,
        dataset_name="A_dev_cv_clean",
        model_names=model_names,
        smoke=smoke,
    )
    holdout_metrics, holdout_predictions, native_predictions, holdout_provenance = _holdout_fusions(
        train,
        holdout,
        model_names=model_names,
        smoke=smoke,
        weights=clean_provenance["weights"],
        native_members=clean_provenance["native_members"],
    )
    fold_summary = _summarize_folds(clean_fold_df)
    holdout_metrics.to_csv(output_dir / "fusion_metrics_A_holdout_clean.csv", index=False, encoding="utf-8-sig")
    holdout_predictions.to_csv(output_dir / "fusion_predictions_A_holdout_clean.csv", index=False, encoding="utf-8-sig")
    native_output.parent.mkdir(parents=True, exist_ok=True)
    native_predictions.to_csv(native_output, index=False, encoding="utf-8-sig")
    clean_fold_df.to_csv(output_dir / "fusion_fold_metrics_A_CV.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(output_dir / "fusion_metrics_A_CV.csv", index=False, encoding="utf-8-sig")
    clean_pred_df.to_csv(output_dir / "fusion_predictions_A_CV.csv", index=False, encoding="utf-8-sig")

    formal_paths_written: list[str] = [str(native_output)]
    manifest = {
        "split_design": split_metadata(),
        "run_name": "A_three_fusions_cv_and_holdout",
        "purpose": "Compute A-centre three-fusion metrics with fold-local CV and a clean holdout.",
        "formal_results_unchanged": True,
        "formal_paths_written_by_script": formal_paths_written,
        "formal_scope_not_updated": True,
        "input_workbook": str(input_workbook),
        "input_workbook_sha256": _sha256_file(input_workbook),
        "formal_input_sha256_reference": None,
        "master_seed": MASTER_SEED, "split_design": split_metadata(),
        "split": {
            "strategy": "stratified_train_test_split",
            "train_fraction": split_metadata()["train_fraction"],
            "holdout_fraction": split_metadata()["test_fraction"],
            "random_state": MASTER_SEED,
            "train_n": int(len(train)),
            "holdout_n": int(len(holdout)),
            "train_record_id_hash": _frame_hash(train[["record_id"]]),
            "holdout_record_id_hash": _frame_hash(holdout[["record_id"]]),
        },
        "cv": {
            "strategy": "StratifiedKFold",
            "n_splits_requested": requested_folds,
            "n_folds_run": effective_limit,
            "shuffle": True,
            "random_state": CV_SEED,
            "fold_local_preprocessor": True,
        },
        "frozen_features": {
            "direct": EXPECTED_DIRECT,
            "direct_n": len(EXPECTED_DIRECT),
            "forced_composites": FIXED_COMPOSITES,
            "total_n": len(EXPECTED_DIRECT) + len(FIXED_COMPOSITES),
            "source_formal_manifest": None,
        },
        "fusion_definitions": {
            FUSION_NAMES[0]: "uniform mean of all fitted native-probability base-model outputs (RBF-SVM decision output excluded)",
            FUSION_NAMES[1]: "one weight vector, normalized from the same development 10-fold CV ROC-AUC means as the out-of-fold probabilities",
            FUSION_NAMES[2]: "L2 logistic regression, C=1.0, lbfgs, max_iter=2000, no raw-feature passthrough, fit on in-sample probabilities from the same training rows",
        },
        "models_run": model_names,
        "strict_oof": False,
        "stacker_training": "in-sample training-row member probabilities",
        "decision_threshold": 0.2076,
        "platt_calibration": False,
        "bootstrap": False,
        "dca": False,
        "errors": clean_errors + holdout_provenance.get("model_errors", []),
        "provenance": {
            "clean_cv": clean_provenance,
            "holdout": holdout_provenance,
        },
        "created_unix": time.time(),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--input-workbook", type=Path, default=INPUT_WORKBOOK)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--native-output", type=Path, default=NATIVE_OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        input_workbook=args.input_workbook,
        n_folds=args.n_folds,
        fold_limit=args.fold_limit,
        smoke=args.smoke,
        native_output=args.native_output,
    )


if __name__ == "__main__":
    main()
