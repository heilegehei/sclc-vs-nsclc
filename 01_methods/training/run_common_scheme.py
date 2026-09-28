from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook


import sys as _cohort_sys
from pathlib import Path as _CohortPath
_cohort_root = next(p for p in _CohortPath(__file__).resolve().parents if (p / "common" / "manuscript_cohorts.py").is_file())
if str(_cohort_root) not in _cohort_sys.path:
    _cohort_sys.path.insert(0, str(_cohort_root))
from common.manuscript_cohorts import split_center_a_indices, validate_cohort, validate_base_cohorts, split_metadata

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent


import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents
                     if (p / "common" / "manuscript_source_paths.py").is_file())
if str(_archive_root) not in _archive_sys.path:
    _archive_sys.path.insert(0, str(_archive_root))
from common.manuscript_source_paths import activate_archive_sources
activate_archive_sources(__file__)
from fold_selection import fit_selectors as strict_fit_selectors, EXPECTED_DIRECT, SELECTION_PROTOCOL
from common.manuscript_bootstrap import MANUSCRIPT_DECISION_THRESHOLD

from stage4_pipeline import (
    ALL_FEATURES,
    COMPOSITE_FEATURES,
    DIRECT_FEATURES,
    FrozenPreprocessor,
    load_base_workbook,
)
from model_factory import (
    MODEL_DISPLAY_NAMES,
    MODEL_NAMES,
    build_model,
    dependency_versions,
    derive_seed as model_seed,
    environment_info,
)


MASTER_SEED = 20260902
SPLIT_NAMESPACE = "SCLC_NSCLC|common_scheme_32direct_10fold_80_20"
RFE_N_FEATURES = 10
DEFAULT_BORUTA_ESTIMATORS = 200
DEFAULT_BORUTA_MAX_ITER = 30


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(x) for x in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subset(frame: pd.DataFrame, indices: Sequence[int]) -> pd.DataFrame:
    return frame.iloc[np.asarray(indices, dtype=int)].reset_index(drop=True)


def _fit_stage4(train: pd.DataFrame, apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:

    pre = FrozenPreprocessor.fit(train)
    x_train, train_qc = pre.transform(train)
    x_apply, apply_qc = pre.transform(apply)
    if list(x_train.columns) != list(ALL_FEATURES) or list(x_apply.columns) != list(ALL_FEATURES):
        raise RuntimeError("Stage-4 feature order mismatch")
    if not np.isfinite(x_train.to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite values remain in transformed training matrix")
    if not np.isfinite(x_apply.to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite values remain in transformed application matrix")
    return x_train, x_apply, {
        "preprocessor_fingerprint": pre.fingerprint,
        "train_qc": train_qc,
        "apply_qc": apply_qc,
    }


def _scores_and_metrics(estimator: Any, x: pd.DataFrame, y: np.ndarray) -> tuple[dict[str, float], str]:
    if hasattr(estimator, "predict_proba"):
        output = np.asarray(estimator.predict_proba(x), dtype=float)
        score = output[:, 1]
        score_kind = "predict_proba"
        probability = np.clip(score, 0.0, 1.0)
        threshold_pred = (probability >= MANUSCRIPT_DECISION_THRESHOLD).astype(int)
    elif hasattr(estimator, "decision_function"):
        score = np.asarray(estimator.decision_function(x), dtype=float).reshape(-1)
        score_kind = "decision_function"
        probability = None
        threshold_pred = None
    else:
        threshold_pred = None
        score = np.asarray(estimator.predict(x), dtype=float).reshape(-1)
        score_kind = "predict"
        probability = None
    metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }
    if threshold_pred is None:
        for key in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1"):
            metrics[key] = float("nan")
    else:
        tn, fp, fn, tp = confusion_matrix(y, threshold_pred, labels=[0, 1]).ravel()
        metrics.update({
            "accuracy": float(accuracy_score(y, threshold_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, threshold_pred)),
            "sensitivity": float(recall_score(y, threshold_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "precision": float(precision_score(y, threshold_pred, zero_division=0)),
            "f1": float(f1_score(y, threshold_pred, zero_division=0)),
        })
    if probability is not None:
        metrics["brier"] = float(brier_score_loss(y, probability))
    else:
        metrics["brier"] = float("nan")
    return metrics, score_kind


def _model_params_for_run(name: str, smoke: bool) -> dict[str, Any] | None:
    if not smoke:
        return None


    if name in {"random_forest", "extra_trees", "xgboost", "lightgbm", "rotation_forest"}:
        return {"n_estimators": 20}
    if name == "mlp":
        return {"max_iter": 80}
    if name == "gam":
        return {"max_iter": 50}
    return None


def _fit_one_model(
    name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_apply: pd.DataFrame,
    y_apply: np.ndarray,
    selected_direct: Sequence[str],
    seed: int,
    *,
    smoke: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    params = _model_params_for_run(name, smoke)
    try:
        estimator = build_model(
            name, seed=int(seed), selected_direct=list(selected_direct), scope_id="common_scheme",
            params=params,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(x_train, y_train)
        metrics, score_kind = _scores_and_metrics(estimator, x_apply, y_apply)
        return {
            "status": "PASS", "error": None, "score_kind": score_kind,
            "runtime_seconds": time.perf_counter() - started,
            **metrics,
        }
    except Exception as exc:
        return {
            "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
            "score_kind": None, "runtime_seconds": time.perf_counter() - started,
            **{metric: float("nan") for metric in ("roc_auc", "average_precision", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier")},
        }


def _prepare_cohort(input_workbook: Path) -> pd.DataFrame:
    data = validate_base_cohorts(load_base_workbook(input_workbook))
    a = data["A"].sort_values("record_id", kind="mergesort").reset_index(drop=True)
    if set(a["center"].astype(str)) != {"A"}:
        raise ValueError("A sheet contains a non-A center")


    if "missing_direct_n" in a.columns:
        a = a.loc[a["missing_direct_n"].astype(float) < len(DIRECT_FEATURES)].copy()
    if len(a) < 100 or a["y_SCLC"].nunique() != 2:
        raise ValueError(f"unexpected A analysis cohort: n={len(a)}")
    return a.reset_index(drop=True)


def run(
    *,
    output_dir: Path,
    input_workbook: Path,
    n_folds: int = 10,
    fold_limit: int | None = None,
    model_names: Sequence[str] = MODEL_NAMES,
    boruta_estimators: int = DEFAULT_BORUTA_ESTIMATORS,
    boruta_max_iter: int = DEFAULT_BORUTA_MAX_ITER,
    smoke: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    a = _prepare_cohort(input_workbook)
    y = a["y_SCLC"].to_numpy(dtype=int)
    indices = np.arange(len(a), dtype=int)
    train_idx, holdout_idx = split_center_a_indices(a)
    train_idx = np.sort(train_idx)
    holdout_idx = np.sort(holdout_idx)
    train_frame = _subset(a, train_idx)
    holdout_frame = _subset(a, holdout_idx)
    y_train_all = train_frame["y_SCLC"].to_numpy(dtype=int)
    y_holdout = holdout_frame["y_SCLC"].to_numpy(dtype=int)
    if len(train_idx) + len(holdout_idx) != len(a) or set(train_idx) & set(holdout_idx):
        raise RuntimeError("Frozen split is not a complete disjoint partition")

    split_rows: list[dict[str, Any]] = []
    for original_position, split_name in [(i, "train80") for i in train_idx] + [(i, "holdout20") for i in holdout_idx]:
        split_rows.append({
            "original_position": int(original_position),
            "record_id": str(a.iloc[int(original_position)]["record_id"]),
            "split": split_name,
            "y_SCLC": int(a.iloc[int(original_position)]["y_SCLC"]),
        })
    pd.DataFrame(split_rows).sort_values("original_position").to_csv(output_dir / "split_records.csv", index=False, encoding="utf-8-sig")

    effective_folds = min(int(n_folds), int(np.bincount(y_train_all).min()))
    if effective_folds < 2:
        raise ValueError("training partition cannot support stratified CV")
    if effective_folds != int(n_folds):
        raise ValueError(f"requested {n_folds} folds but smallest class has only {effective_folds} training rows")
    splitter = StratifiedKFold(
        n_splits=effective_folds, shuffle=True,
        random_state=model_seed(MASTER_SEED, "10fold", "train80"),
    )
    x_dummy = np.zeros((len(train_frame), 1), dtype=np.uint8)
    requested_limit = int(fold_limit) if fold_limit is not None else effective_folds
    if requested_limit < 1 or requested_limit > effective_folds:
        raise ValueError("fold_limit must be between 1 and n_folds")

    cv_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    fold_manifest_rows: list[dict[str, Any]] = []
    for fold_number, (fold_train_rel, fold_valid_rel) in enumerate(splitter.split(x_dummy, y_train_all), start=1):
        if fold_number > requested_limit:
            break
        fold_train = _subset(train_frame, fold_train_rel)
        fold_valid = _subset(train_frame, fold_valid_rel)
        fold_y_train = fold_train["y_SCLC"].to_numpy(dtype=int)
        fold_y_valid = fold_valid["y_SCLC"].to_numpy(dtype=int)
        x_fold_train, x_fold_valid, prep_info = _fit_stage4(fold_train, fold_valid)
        selector_info = strict_fit_selectors(
            x_fold_train, fold_y_train, model_seed(MASTER_SEED, "cv", fold_number),
            boruta_estimators=boruta_estimators, boruta_max_iter=boruta_max_iter,
        )
        selector_json = json.dumps(_json_safe(selector_info), ensure_ascii=False, sort_keys=True)
        for method_record in selector_info["methods"]:
            selection_rows.append({
                "partition": "cv", "fold": fold_number, "method": method_record["method"],
                "backend": method_record.get("backend"), "selected_direct": "|".join(method_record["selected_features"]),
                "n_selected_direct": method_record["n_selected"], "fallback_used": method_record.get("fallback_used"),
            })
        selection_rows.append({
            "partition": "cv", "fold": fold_number, "method": "consensus_2of3", "backend": None,
            "selected_direct": "|".join(selector_info["selected_direct"]),
            "n_selected_direct": selector_info["n_selected_direct"], "fallback_used": False,
        })
        fold_manifest_rows.append({
            "partition": "cv", "fold": fold_number, "train_n": len(fold_train), "validation_n": len(fold_valid),
            "train_SCLC": int(fold_y_train.sum()), "validation_SCLC": int(fold_y_valid.sum()),
            "preprocessor_fingerprint": prep_info["preprocessor_fingerprint"],
            "selected_direct": "|".join(selector_info["selected_direct"]),
            "selected_total": "|".join(selector_info["selected_features"]),
            "selected_n_direct": selector_info["n_selected_direct"],
            "selector_json": selector_json,
        })
        for model_name in model_names:
            result = _fit_one_model(
                str(model_name), x_fold_train, fold_y_train, x_fold_valid, fold_y_valid,
                selector_info["selected_direct"], model_seed(MASTER_SEED, "cv", fold_number, "model", model_name),
                smoke=smoke,
            )
            cv_rows.append({
                "partition": "cv", "fold": fold_number, "model": str(model_name),
                "display_name": MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name)),
                "selected_n_direct": selector_info["n_selected_direct"],
                "selected_direct": "|".join(selector_info["selected_direct"]),
                **result,
            })
        print(f"completed CV fold {fold_number}/{requested_limit}: {len(selector_info['selected_direct'])} direct features", flush=True)


    x_train_final, x_holdout, final_prep_info = _fit_stage4(train_frame, holdout_frame)
    final_selector = strict_fit_selectors(
        x_train_final, y_train_all, model_seed(MASTER_SEED, "holdout_fit"),
        boruta_estimators=boruta_estimators, boruta_max_iter=boruta_max_iter,
    )
    if not smoke and list(final_selector["selected_direct"]) != list(EXPECTED_DIRECT):
        raise RuntimeError("Full-development strict consensus differs from the manuscript final predictor set")
    for method_record in final_selector["methods"]:
        selection_rows.append({
            "partition": "holdout_fit", "fold": 0, "method": method_record["method"],
            "backend": method_record.get("backend"), "selected_direct": "|".join(method_record["selected_features"]),
            "n_selected_direct": method_record["n_selected"], "fallback_used": method_record.get("fallback_used"),
        })
    selection_rows.append({
        "partition": "holdout_fit", "fold": 0, "method": "consensus_2of3", "backend": None,
        "selected_direct": "|".join(final_selector["selected_direct"]),
        "n_selected_direct": final_selector["n_selected_direct"], "fallback_used": False,
    })
    fold_manifest_rows.append({
        "partition": "holdout_fit", "fold": 0, "train_n": len(train_frame), "validation_n": len(holdout_frame),
        "train_SCLC": int(y_train_all.sum()), "validation_SCLC": int(y_holdout.sum()),
        "preprocessor_fingerprint": final_prep_info["preprocessor_fingerprint"],
        "selected_direct": "|".join(final_selector["selected_direct"]),
        "selected_total": "|".join(final_selector["selected_features"]),
        "selected_n_direct": final_selector["n_selected_direct"],
        "selector_json": json.dumps(_json_safe(final_selector), ensure_ascii=False, sort_keys=True),
    })
    holdout_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        result = _fit_one_model(
            str(model_name), x_train_final, y_train_all, x_holdout, y_holdout,
            final_selector["selected_direct"], model_seed(MASTER_SEED, "holdout_fit", "model", model_name),
            smoke=smoke,
        )
        holdout_rows.append({
            "partition": "holdout20", "fold": 0, "model": str(model_name),
            "display_name": MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name)),
            "selected_n_direct": final_selector["n_selected_direct"],
            "selected_direct": "|".join(final_selector["selected_direct"]),
            **result,
        })
        print(f"completed holdout {model_name}: {result['status']}", flush=True)

    cv_df = pd.DataFrame(cv_rows)
    holdout_df = pd.DataFrame(holdout_rows)
    selection_df = pd.DataFrame(selection_rows)
    fold_manifest_df = pd.DataFrame(fold_manifest_rows)
    cv_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    holdout_df.to_csv(output_dir / "holdout_metrics.csv", index=False, encoding="utf-8-sig")
    selection_df.to_csv(output_dir / "selection_records.csv", index=False, encoding="utf-8-sig")
    fold_manifest_df.to_csv(output_dir / "fold_manifest.csv", index=False, encoding="utf-8-sig")

    metric_names = ["roc_auc", "average_precision", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier"]
    summary_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        rows = cv_df.loc[(cv_df["model"] == str(model_name)) & (cv_df["status"] == "PASS")] if not cv_df.empty else pd.DataFrame()
        holdout_row = holdout_df.loc[holdout_df["model"] == str(model_name)].iloc[0] if not holdout_df.empty else None
        row: dict[str, Any] = {
            "model": str(model_name), "display_name": MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name)),
            "cv_folds_completed": int(len(rows)), "cv_failures": int((cv_df.loc[cv_df["model"] == str(model_name), "status"] == "FAIL").sum()) if not cv_df.empty else 0,
            "final_selected_n_direct": final_selector["n_selected_direct"],
            "final_selected_direct": "|".join(final_selector["selected_direct"]),
        }
        for metric in metric_names:
            row[f"cv_{metric}_mean"] = float(rows[metric].mean()) if not rows.empty else float("nan")
            row[f"cv_{metric}_sd"] = float(rows[metric].std(ddof=1)) if len(rows) > 1 else float("nan")
            row[f"holdout_{metric}"] = float(holdout_row[metric]) if holdout_row is not None else float("nan")
        row["holdout_status"] = holdout_row["status"] if holdout_row is not None else "MISSING"
        row["holdout_error"] = holdout_row["error"] if holdout_row is not None else None
        summary_rows.append(row)
    protocol = SELECTION_PROTOCOL if (not smoke and n_folds == 10 and requested_limit == 10 and boruta_estimators == 200 and boruta_max_iter == 30 and tuple(model_names) == tuple(MODEL_NAMES) and len(cv_df) == 140 and len(holdout_df) == 14 and cv_df["status"].eq("PASS").all() and holdout_df["status"].eq("PASS").all()) else "non-manuscript-run"
    summary_df = pd.DataFrame(summary_rows)
    summary_df["selection_protocol"] = protocol
    summary_df.to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "split_design": split_metadata(),
        "selection_protocol": protocol,
        "run_name": "common_medical_ml_scheme_32direct_lasso_rfe_boruta_10fold_80_20",
        "status": "SMOKE" if smoke else "FULL",
        "master_seed": MASTER_SEED, "split_design": split_metadata(),
        "split": {"strategy": "stratified_train_test_split", "train_fraction": split_metadata()["train_fraction"], "holdout_fraction": split_metadata()["test_fraction"], "random_state": MASTER_SEED},
        "cv": {"strategy": "StratifiedKFold", "n_splits": effective_folds, "shuffle": True, "random_state": model_seed(MASTER_SEED, "10fold", "train80"), "folds_run": requested_limit},
        "cohort": {"center": "A", "analysis_n": len(a), "train_n": len(train_frame), "holdout_n": len(holdout_frame), "class_counts": a["y_SCLC"].value_counts().sort_index().to_dict()},
        "candidate_direct_features": list(DIRECT_FEATURES),
        "candidate_direct_n": len(DIRECT_FEATURES),
        "forced_composites": list(COMPOSITE_FEATURES),
        "selection": {
            "methods": ["L1-LASSO", "logistic-RFE", "BorutaPy"],
            "primary_rule": "at_least_2_of_3",
            "rfe_n_features": RFE_N_FEATURES,
            "boruta_estimators": int(boruta_estimators),
            "boruta_max_iter": int(boruta_max_iter),
            "fit_scope": "each_cv_training_fold_and_train80_final_fit",
        },
        "models": list(model_names),
        "model_count": len(model_names),
        "calibration": "none; AUC uses native probability or decision output",
        "preprocessing": "Stage4 E1-E8 transform fit within each training scope; no holdout fitting",
        "input_workbook": str(input_workbook),
        "input_sha256": _sha256_file(input_workbook),
        "environment": environment_info(),
        "dependency_versions": dependency_versions(),
        "final_holdout_selector": _json_safe(final_selector),
        "final_preprocessor": final_prep_info,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-workbook", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    parser.add_argument("--boruta-estimators", type=int, default=DEFAULT_BORUTA_ESTIMATORS)
    parser.add_argument("--boruta-max-iter", type=int, default=DEFAULT_BORUTA_MAX_ITER)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    input_workbook = cohort_workbook() if args.input_workbook is None else args.input_workbook
    fold_limit = args.fold_limit
    if args.smoke:
        fold_limit = 1 if fold_limit is None else min(int(fold_limit), 1)
    run(
        output_dir=args.output_dir,
        input_workbook=input_workbook,
        n_folds=args.n_folds,
        fold_limit=fold_limit,
        model_names=args.models,
        boruta_estimators=args.boruta_estimators if not args.smoke else min(args.boruta_estimators, 100),
        boruta_max_iter=args.boruta_max_iter if not args.smoke else min(args.boruta_max_iter, 15),
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    main()
