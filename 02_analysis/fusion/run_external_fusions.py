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
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = data_root()
for item in (HERE,):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents
                     if (p / "common" / "manuscript_source_paths.py").is_file())
if str(_archive_root) not in _archive_sys.path:
    _archive_sys.path.insert(0, str(_archive_root))
from common.manuscript_source_paths import activate_archive_sources
activate_archive_sources(__file__)
from fold_selection import fit_scope_selection, SELECTION_PROTOCOL, require_selection_protocol

from model_factory import MODEL_NAMES, build_model, derive_seed as model_seed
from stage4_pipeline import FrozenPreprocessor, load_base_workbook
from analysis_utils import (
    EXPECTED_DIRECT,
    FIXED_COMPOSITES,
    _prepare_center,
    _safe,
    _sha,
)
from run_a_center_fusions import _apply_fixed_fusions, _clean_folds, _fit_stacker, _member_matrix, _run_cv


MASTER_SEED = 20260902
OUTPUT_DEFAULT = data_root() / "fusion_results" / "three_fusions_BC"
FUSION_NAMES = (
    "soft_voting_native_probability",
    "weighted_voting_cv_auc",
    "stacking_in_sample_A_train",
)


def _score(estimator: Any, matrix: pd.DataFrame) -> tuple[np.ndarray, str, np.ndarray | None]:
    if hasattr(estimator, "predict_proba"):
        probability = np.asarray(estimator.predict_proba(matrix), dtype=float)[:, 1]
        return probability, "probability", np.clip(probability, 0.0, 1.0)
    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(matrix), dtype=float).reshape(-1)
        return decision, "decision_function", None
    prediction = np.asarray(estimator.predict(matrix), dtype=float).reshape(-1)
    return prediction, "predict", None


def _point_metrics(y: np.ndarray, score: np.ndarray, probability: np.ndarray | None, model: str, dataset: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "n": int(len(y)),
        "positive_n": int(y.sum()),
        "negative_n": int(len(y) - y.sum()),
        "score_kind": "probability" if probability is not None else "decision_function",
        "auc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }
    if probability is None:
        row.update({"brier": None, "log_loss": None})
    else:
        row.update({
            "brier": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6), labels=[0, 1])),
        })
    return row


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = list(EXPECTED_DIRECT)

    data = validate_base_cohorts(load_base_workbook(cohort_workbook()))
    a = _prepare_center(data["A"], "A")
    b = _prepare_center(data["B"], "B")
    c = _prepare_center(data["C"], "C")
    y_a = a["y_SCLC"].to_numpy(dtype=int)
    all_idx = np.arange(len(a), dtype=int)
    train_idx, holdout_idx = split_center_a_indices(a)
    train_idx, holdout_idx = np.sort(train_idx), np.sort(holdout_idx)
    train = a.iloc[train_idx].reset_index(drop=True)
    holdout = a.iloc[holdout_idx].reset_index(drop=True)
    if set(train["record_id"].astype(str)) & set(holdout["record_id"].astype(str)):
        raise RuntimeError("A development and holdout record ids overlap")
    y_train = train["y_SCLC"].to_numpy(dtype=int)
    pre = FrozenPreprocessor.fit(train)
    x_train, train_qc = pre.transform(train)
    selection = fit_scope_selection(x_train, y_train)
    frozen = list(selection["selected_direct"])
    x_b, b_qc = pre.transform(b)
    x_c, c_qc = pre.transform(c)

    estimators: dict[str, Any] = {}
    train_prob: dict[str, np.ndarray | None] = {}
    external_prob: dict[str, dict[str, np.ndarray | None]] = {"B_external": {}, "C_external": {}}
    external_raw: dict[str, dict[str, np.ndarray]] = {"B_external": {}, "C_external": {}}
    errors: list[dict[str, str]] = []
    matrices = {"B_external": x_b, "C_external": x_c}
    for model in MODEL_NAMES:
        try:
            estimator = build_model(
                model,
                seed=model_seed(MASTER_SEED, "holdout_fit", "model", model),
                selected_direct=frozen,
                scope_id="common_scheme",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(x_train, y_train)
            estimators[model] = estimator
            _train_raw, train_kind, train_prob[model] = _score(estimator, x_train)
            for dataset, matrix in matrices.items():
                raw, kind, probability = _score(estimator, matrix)
                if train_kind != kind:
                    raise RuntimeError(f"score-kind mismatch for {model} on {dataset}")
                external_raw[dataset][model] = raw
                external_prob[dataset][model] = probability
        except Exception as exc:
            errors.append({"stage": "fit_A_development", "model": model, "error": f"{type(exc).__name__}: {exc}"})
    if errors or set(estimators) != set(MODEL_NAMES):
        raise RuntimeError(f"Formal candidate fitting failed; all 14 models are required: {errors}")
    fitted = [m for m in MODEL_NAMES if m in estimators]
    native_members = [m for m in fitted if train_prob.get(m) is not None and all(external_prob[d].get(m) is not None for d in matrices)]
    if len(native_members) < 2:
        raise RuntimeError("not enough native-probability models for fusion")

    folds, _assignment = _clean_folds(train, 10)
    _fold_df, _pred_df, _errors, contract = _run_cv(
        train, folds=folds, dataset_name="A_dev_cv_clean", model_names=list(MODEL_NAMES), smoke=False,
    )
    native_members = list(contract["native_members"])
    weight_map = dict(contract["weights"])
    stacker = _fit_stacker(
        _member_matrix(train_prob, native_members),
        y_train,
        model_seed(MASTER_SEED, "stacker", "development_in_sample"),
    )
    if tuple(native_members) != tuple(m for m in MODEL_NAMES if m != "rbf_svm"):
        raise RuntimeError("External fusion requires all 13 native-probability members and valid CV AUC weights")

    labels = {
        "B_external": b["y_SCLC"].to_numpy(dtype=int),
        "C_external": c["y_SCLC"].to_numpy(dtype=int),
    }
    frames = {"B_external": b, "C_external": c}
    fusions: dict[str, dict[str, np.ndarray]] = {}
    for dataset in matrices:
        fusions[dataset] = _apply_fixed_fusions(external_prob[dataset], native_members, weight_map, stacker)

    pred_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for dataset, frame in frames.items():
        scores = {**external_raw[dataset], **fusions[dataset]}
        for model, score in scores.items():
            if model in fusions[dataset]:
                probability = fusions[dataset][model]
            else:
                probability = external_prob[dataset].get(model)
            metric_rows.append(_point_metrics(labels[dataset], score, probability, model, dataset))
            for i, record_id in enumerate(frame["record_id"].astype(str)):
                pred_rows.append({
                    "selection_protocol": SELECTION_PROTOCOL,
                    "dataset": dataset,
                    "model": model,
                    "record_id": record_id,
                    "y_SCLC": int(labels[dataset][i]),
                    "raw_score": float(score[i]),
                    "native_probability": float(probability[i]) if probability is not None else None,
                })
    pd.DataFrame(pred_rows).to_csv(output_dir / "fusion_predictions_B_C.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(output_dir / "fusion_point_metrics_B_C.csv", index=False, encoding="utf-8-sig")

    formal_paths = []
    manifest = {
        "split_design": split_metadata(),
        "selection_protocol": SELECTION_PROTOCOL,
        "selection": selection,
        "run_name": "three_fusions_BC_external",
        "purpose": "Evaluate three fusion models on clean external B and C.",
        "formal_results_unchanged": True,
        "formal_holdout_untouched": True,
        "formal_external_results_untouched": True,
        "formal_input_workbook_unchanged": True,
        "evaluated_datasets_only": ["B_external", "C_external"],
        "fusion_definitions": {
            "soft_voting_native_probability": "unweighted mean of native probabilities across native-probability base models",
            "weighted_voting_cv_auc": "fixed weights from the same A-development 10-fold CV ROC-AUC means as the out-of-fold probabilities",
            "stacking_in_sample_A_train": "L2 logistic regression fit on in-sample A-development member probabilities; C=1.0, lbfgs, max_iter=2000, no raw-feature passthrough",
        },
        "native_members": native_members,
        "weighted_voting_weights": weight_map,
        "frozen_features": {"direct": frozen, "forced_composites": FIXED_COMPOSITES, "total_n": len(frozen) + len(FIXED_COMPOSITES)},
        "models": fitted,
        "strict_OOF": False,
        "Platt_calibration": "SKIPPED_BY_USER",
        "model_errors": errors,
        "preprocessing": {
            "fit_scope": "A development only",
            "B_C_scope": "transform only; no B/C fitting or tuning",
            "train_qc": train_qc,
            "B_qc": b_qc,
            "C_qc": c_qc,
            "preprocessor_fingerprint": pre.fingerprint,
        },
        "integrity": {
            "input_workbook": str(cohort_workbook()),
            "input_workbook_sha256": _sha(cohort_workbook()),
            "formal_reference_hashes_after_run": {str(p): _sha(p) for p in formal_paths if p.exists()},
            "formal_paths_written_by_script": [],
            "development_holdout_disjoint": True,
        },
        "outputs": [str(p.relative_to(output_dir)) for p in sorted(output_dir.rglob("*")) if p.is_file() and p.name != "run_manifest.json"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    manifest = run(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "errors": manifest["model_errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
