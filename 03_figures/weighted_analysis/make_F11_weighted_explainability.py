from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, data_root, fusion_results, publication_results


import sys as _cohort_sys
from pathlib import Path as _CohortPath
_cohort_root = next(p for p in _CohortPath(__file__).resolve().parents if (p / "common" / "manuscript_cohorts.py").is_file())
if str(_cohort_root) not in _cohort_sys.path:
    _cohort_sys.path.insert(0, str(_cohort_root))
from common.manuscript_cohorts import split_center_a_indices, validate_cohort, validate_base_cohorts, split_metadata, require_split_design

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
FINAL_ROOT = ANALYSIS_DIR.parent
PROJECT_ROOT = data_root()
for _path in (HERE,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from nature_style import FEATURE_GROUP_COLORS, apply_style, panel_label, save_fig

import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents
                     if (p / "common" / "manuscript_source_paths.py").is_file())
if str(_archive_root) not in _archive_sys.path:
    _archive_sys.path.insert(0, str(_archive_root))
from common.manuscript_source_paths import activate_archive_sources
activate_archive_sources(__file__)
from fold_selection import require_selection_protocol

from model_factory import MODEL_NAMES, build_model, derive_seed as model_seed
from stage4_pipeline import (
    ALL_FEATURES,
    DIRECT_FEATURES,
    FrozenPreprocessor,
    load_base_workbook,
)


MASTER_SEED = 20260902
SHAP_SEED = 20260911
ALE_SEED = 20260912
BACKGROUND_N = 40
MAX_EVALS = 37
ALE_BINS = 5
FROZEN_DIRECT = [
    "WBC", "PLT", "LDH", "PCT", "EO#", "EO%", "ALB", "TP", "GLB",
    "TC", "LDL-C", "MCH", "MCHC",
]
FIXED_COMPOSITES = ["SIRI", "LMR", "GAR", "PNI", "HALP"]
EXPLAIN_FEATURES = FROZEN_DIRECT + FIXED_COMPOSITES
NATIVE_MEMBERS = [name for name in MODEL_NAMES if name != "rbf_svm"]
FEATURE_GROUPS = {
    "WBC": "I", "PLT": "I", "LDH": "I", "PCT": "I", "SIRI": "I",
    "EO#": "M", "EO%": "M", "LMR": "M", "GAR": "M",
    "ALB": "N", "TP": "N", "GLB": "N", "TC": "N", "LDL-C": "N",
    "MCH": "N", "MCHC": "N", "PNI": "N", "HALP": "N",
}
DISPLAY_FEATURE = {"EO#": "EO count", "EO%": "EO %", "LDL-C": "LDL-C"}

DATA_DIR = fusion_results() / "data"
TABLE_DIR = fusion_results() / "tables"
MODEL_DIR = fusion_results() / "model"
FIGURE_DIR = publication_results() / "figures" / "main"
for _directory in (DATA_DIR, TABLE_DIR, MODEL_DIR, FIGURE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

INPUT_WORKBOOK = cohort_workbook()
A_REFERENCE = fusion_results() / "A_fusions_CV_holdout" / "fusion_predictions_A_holdout_clean.csv"
A_FUSION_MANIFEST = fusion_results() / "A_fusions_CV_holdout" / "run_manifest.json"


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_center(frame: pd.DataFrame, center: str) -> pd.DataFrame:
    out = frame.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    if set(out["center"].astype(str)) != {center}:
        raise RuntimeError(f"{center} data contain an unexpected center label")
    if "missing_direct_n" in out.columns:
        out = out.loc[out["missing_direct_n"].astype(float) < len(DIRECT_FEATURES)].copy()
    if out["record_id"].duplicated().any() or out["y_SCLC"].nunique() != 2:
        raise RuntimeError(f"{center} cohort identity or outcome check failed")
    return out.reset_index(drop=True)


def _read_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _metadata(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["record_id"] = frame["record_id"].astype(str)
    out["source_record_id"] = frame.get("source_record_id", frame["record_id"]).astype(str)
    out["duplicate_flag"] = _read_bool(frame.get("duplicate_flag", pd.Series(False, index=frame.index)))
    out["y_SCLC"] = frame["y_SCLC"].astype(int)
    keys = ["record_id", "source_record_id", "duplicate_flag", "y_SCLC"]
    out["key_occurrence"] = out.groupby(keys, dropna=False).cumcount()
    return out


def _expanded_stage4(values: Any) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        array = values.loc[:, EXPLAIN_FEATURES].to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(EXPLAIN_FEATURES):
        raise ValueError(f"expected an (n, {len(EXPLAIN_FEATURES)}) transformed matrix")
    full = pd.DataFrame(0.0, index=np.arange(array.shape[0]), columns=list(ALL_FEATURES))
    full.loc[:, EXPLAIN_FEATURES] = array
    return full


class WeightedPredictor:

    def __init__(self, estimators: dict[str, Any], weights: dict[str, float]):
        self.estimators = estimators
        self.weights = weights
        self.member_order = list(weights)

    def predict(self, values: Any) -> np.ndarray:
        full = _expanded_stage4(values)
        matrix = np.column_stack(
            [np.asarray(self.estimators[name].predict_proba(full), dtype=float)[:, 1] for name in self.member_order]
        )
        weight_vector = np.asarray([self.weights[name] for name in self.member_order], dtype=float)
        return np.clip(matrix @ weight_vector, 0.0, 1.0)

    def predict_full_stage4(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = np.column_stack(
            [np.asarray(self.estimators[name].predict_proba(frame), dtype=float)[:, 1] for name in self.member_order]
        )
        weight_vector = np.asarray([self.weights[name] for name in self.member_order], dtype=float)
        return np.clip(matrix @ weight_vector, 0.0, 1.0)


def _load_and_fit() -> dict[str, Any]:
    if not INPUT_WORKBOOK.is_file():
        raise FileNotFoundError("Cohort workbook is not available.")
    frozen_manifest = {}
    sheets = validate_base_cohorts(load_base_workbook(INPUT_WORKBOOK))
    a = _prepare_center(sheets["A"], "A")
    indices = np.arange(len(a), dtype=int)
    train_idx, holdout_idx = split_center_a_indices(a)
    train = a.iloc[np.sort(train_idx)].reset_index(drop=True)
    clean_holdout = a.iloc[np.sort(holdout_idx)].reset_index(drop=True)
    b_input = _prepare_center(sheets["B"], "B")
    c_input = _prepare_center(sheets["C"], "C")

    preprocessor = FrozenPreprocessor.fit(train)
    transformed: dict[str, pd.DataFrame] = {}
    qc: dict[str, dict[str, int]] = {}
    for name, frame in {
        "A_development": train,
        "A_holdout_clean": clean_holdout,
        "B_external": b_input,
        "C_external": c_input,
    }.items():
        transformed[name], qc[name] = preprocessor.transform(frame)
        if not np.isfinite(transformed[name].to_numpy(dtype=float)).all():
            raise RuntimeError(f"non-finite Stage-4 value remains in {name}")

    y_train = train["y_SCLC"].to_numpy(dtype=int)
    estimators: dict[str, Any] = {}
    for name in NATIVE_MEMBERS:
        estimator = build_model(
            name,
            seed=model_seed(MASTER_SEED, "holdout_fit", "model", name),
            selected_direct=FROZEN_DIRECT,
            scope_id="common_scheme",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(transformed["A_development"], y_train)
        if not hasattr(estimator, "predict_proba"):
            raise RuntimeError(f"native-probability member {name} lacks predict_proba")
        estimators[name] = estimator
    if list(estimators) != NATIVE_MEMBERS:
        raise RuntimeError("the complete 13-member Weighted-voting ensemble was not fitted")

    frozen_weights = json.loads(A_FUSION_MANIFEST.read_text(encoding="utf-8"))["provenance"]["clean_cv"]["weights"]
    weights = {name: float(frozen_weights[name]) for name in NATIVE_MEMBERS}
    if not np.isfinite(list(weights.values())).all() or any(value <= 0 for value in weights.values()):
        raise RuntimeError("invalid frozen manuscript voting weight")
    if not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-8):
        raise RuntimeError("frozen manuscript voting weights do not sum to 1")
    predictor = WeightedPredictor(estimators, weights)

    input_frames = {
        "A_holdout_clean": clean_holdout,
        "B_external": b_input,
        "C_external": c_input,
    }
    predictions = {
        name: predictor.predict_full_stage4(transformed[name])
        for name in input_frames
    }
    return {
        "frozen_manifest": frozen_manifest,
        "train": train,
        "clean_holdout": clean_holdout,
        "input_frames": input_frames,
        "transformed": transformed,
        "qc": qc,
        "preprocessor": preprocessor,
        "estimators": estimators,
        "weights": weights,
        "predictor": predictor,
        "predictions": predictions,
    }


def _align_and_compare(dataset: str, input_frame: pd.DataFrame, predicted: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed = _metadata(input_frame)
    observed["refit_weighted_probability"] = np.asarray(predicted, dtype=float)
    if dataset != "A_holdout_clean":
        audit = {
            "dataset": dataset,
            "n": int(len(observed)),
            "max_abs_difference": 0.0,
            "mean_abs_difference": 0.0,
            "tolerance": 1e-10,
            "status": "PASS",
        }
        return observed, audit
    reference = pd.read_csv(A_REFERENCE)
    reference = reference.loc[
        (reference["dataset"].astype(str) == dataset)
        & (reference["fusion"].astype(str) == "weighted_voting_cv_auc")
    ].copy()
    reference_score = "score"
    reference_meta = _metadata(reference)
    reference_meta["saved_weighted_probability"] = pd.to_numeric(reference[reference_score], errors="raise")
    keys = ["record_id", "source_record_id", "duplicate_flag", "y_SCLC", "key_occurrence"]
    merged = observed.merge(reference_meta, on=keys, how="outer", validate="one_to_one", indicator=True)
    if len(merged) != len(observed) or not (merged["_merge"] == "both").all():
        raise RuntimeError(f"{dataset}: refit/reference row identities do not align")
    merged["abs_difference"] = (
        merged["refit_weighted_probability"] - merged["saved_weighted_probability"]
    ).abs()
    max_abs = float(merged["abs_difference"].max())
    audit = {
        "dataset": dataset,
        "n": int(len(merged)),
        "max_abs_difference": max_abs,
        "mean_abs_difference": float(merged["abs_difference"].mean()),
        "tolerance": 1e-10,
        "status": "PASS" if max_abs <= 1e-10 else "FAIL",
    }
    return merged.drop(columns="_merge"), audit


def _prediction_audit(fit: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_predictions: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for dataset, frame in fit["input_frames"].items():
        aligned, audit = _align_and_compare(dataset, frame, fit["predictions"][dataset])
        aligned.insert(0, "dataset", dataset)
        all_predictions.append(aligned)
        audit_rows.append(audit)
    predictions = pd.concat(all_predictions, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    predictions.to_csv(DATA_DIR / "F11_weighted_voting_refit_predictions.csv", index=False)
    audit.to_csv(DATA_DIR / "F11_prediction_reproducibility_audit.csv", index=False)
    if (audit["status"] != "PASS").any():
        details = audit.loc[audit["status"] != "PASS", ["dataset", "max_abs_difference"]].to_dict("records")
        raise RuntimeError(f"refitted Weighted-voting probabilities do not reproduce saved predictions: {details}")
    return predictions, audit


def _choose_background(train: pd.DataFrame, x_train_18: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SHAP_SEED)

    chosen: list[int] = []
    for outcome in (0, 1):
        candidates = np.flatnonzero(train["y_SCLC"].to_numpy(dtype=int) == outcome)
        n_class = int(round(BACKGROUND_N * len(candidates) / len(train)))
        chosen.extend(rng.choice(candidates, size=n_class, replace=False).tolist())
    if len(chosen) != BACKGROUND_N:
        remaining = np.setdiff1d(np.arange(len(train)), np.asarray(chosen, dtype=int))
        if len(chosen) < BACKGROUND_N:
            chosen.extend(rng.choice(remaining, size=BACKGROUND_N - len(chosen), replace=False).tolist())
        else:
            chosen = chosen[:BACKGROUND_N]
    chosen = sorted(chosen)
    meta = train.iloc[chosen][["record_id", "y_SCLC"]].reset_index(drop=True)
    return x_train_18.iloc[chosen].reset_index(drop=True), meta


def _run_shap(fit: dict[str, Any]) -> dict[str, Any]:
    train = fit["train"]
    holdout = fit["input_frames"]["A_holdout_clean"].reset_index(drop=True)
    x_train_18 = fit["transformed"]["A_development"].loc[:, EXPLAIN_FEATURES].reset_index(drop=True)
    x_explain = fit["transformed"]["A_holdout_clean"].loc[:, EXPLAIN_FEATURES].reset_index(drop=True)
    background, background_meta = _choose_background(train, x_train_18)
    np.random.seed(SHAP_SEED)
    masker = shap.maskers.Independent(background, max_samples=BACKGROUND_N)
    explainer = shap.Explainer(
        fit["predictor"].predict,
        masker,
        algorithm="permutation",
        feature_names=EXPLAIN_FEATURES,
        seed=SHAP_SEED,
    )
    explanation = explainer(x_explain, max_evals=MAX_EVALS, batch_size=256)
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if values.shape != (len(holdout), len(EXPLAIN_FEATURES)):
        raise RuntimeError(f"unexpected SHAP shape {values.shape}")
    if base_values.size == 1:
        base_values = np.repeat(base_values, len(holdout))
    predicted = fit["predictions"]["A_holdout_clean"]
    reconstructed = base_values + values.sum(axis=1)
    additivity_max_abs = float(np.max(np.abs(reconstructed - predicted)))
    if additivity_max_abs > 1e-6:
        raise RuntimeError(f"permutation SHAP additivity check failed: {additivity_max_abs}")

    meta = _metadata(holdout)
    long_rows: list[dict[str, Any]] = []
    for row_index in range(len(holdout)):
        for feature_index, feature in enumerate(EXPLAIN_FEATURES):
            long_rows.append({
                "sample_index": row_index,
                "record_id": meta.loc[row_index, "record_id"],
                "source_record_id": meta.loc[row_index, "source_record_id"],
                "duplicate_flag": bool(meta.loc[row_index, "duplicate_flag"]),
                "y_SCLC": int(meta.loc[row_index, "y_SCLC"]),
                "weighted_probability": float(predicted[row_index]),
                "base_value": float(base_values[row_index]),
                "feature": feature,
                "feature_group": FEATURE_GROUPS[feature],
                "transformed_value": float(x_explain.iloc[row_index, feature_index]),
                "shap_value": float(values[row_index, feature_index]),
            })
    long = pd.DataFrame(long_rows)
    long.to_csv(DATA_DIR / "F11_permutation_shap_long.csv", index=False)

    global_rows = []
    for feature_index, feature in enumerate(EXPLAIN_FEATURES):
        global_rows.append({
            "feature": feature,
            "feature_group": FEATURE_GROUPS[feature],
            "mean_abs_shap": float(np.mean(np.abs(values[:, feature_index]))),
            "mean_shap": float(np.mean(values[:, feature_index])),
            "shap_sd": float(np.std(values[:, feature_index], ddof=1)),
        })
    global_table = pd.DataFrame(global_rows).sort_values(
        ["mean_abs_shap", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    global_table.insert(0, "rank", np.arange(1, len(global_table) + 1))
    global_table.to_csv(TABLE_DIR / "F11_shap_global_importance.csv", index=False)

    base_table = meta.copy()
    base_table["sample_index"] = np.arange(len(base_table), dtype=int)
    base_table["base_value"] = base_values
    base_table["shap_sum"] = values.sum(axis=1)
    base_table["reconstructed_probability"] = reconstructed
    base_table["weighted_probability"] = predicted
    base_table["abs_additivity_difference"] = np.abs(reconstructed - predicted)
    base_table.to_csv(DATA_DIR / "F11_shap_base_values.csv", index=False)

    representatives = []
    y = holdout["y_SCLC"].to_numpy(dtype=int)
    for outcome, label in ((0, "NSCLC"), (1, "SCLC")):
        candidates = np.flatnonzero(y == outcome)
        median_probability = float(np.median(predicted[candidates]))
        candidate_table = pd.DataFrame({
            "sample_index": candidates,
            "distance": np.abs(predicted[candidates] - median_probability),
            "record_id": meta.loc[candidates, "record_id"].to_numpy(),
            "source_record_id": meta.loc[candidates, "source_record_id"].to_numpy(),
        }).sort_values(["distance", "record_id", "source_record_id"], kind="mergesort")
        selected = int(candidate_table.iloc[0]["sample_index"])
        representatives.append({
            "true_class": outcome,
            "class_label": label,
            "selection_rule": "probability closest to the within-class median; record_id tie-break",
            "within_class_median_probability": median_probability,
            "sample_index": selected,
            "record_id": meta.loc[selected, "record_id"],
            "source_record_id": meta.loc[selected, "source_record_id"],
            "duplicate_flag": bool(meta.loc[selected, "duplicate_flag"]),
            "weighted_probability": float(predicted[selected]),
            "base_value": float(base_values[selected]),
        })
    representatives_table = pd.DataFrame(representatives)
    representatives_table.to_csv(TABLE_DIR / "F11_local_representatives.csv", index=False)

    parameters = {
        "split_design": split_metadata(),
        "analysis": "Weighted-voting model-agnostic explanation",
        "explainer": "Permutation SHAP with an Independent masker",
        "algorithm": "permutation",
        "max_evals": MAX_EVALS,
        "n_features": len(EXPLAIN_FEATURES),
        "background_n": BACKGROUND_N,
        "background_selection": "deterministic proportional stratified sample from A development",
        "background_record_ids": background_meta["record_id"].astype(str).tolist(),
        "background_class_counts": background_meta["y_SCLC"].value_counts().sort_index().to_dict(),
        "explanation_dataset": "A_holdout_clean",
        "explanation_n": len(holdout),
        "feature_scale": "Stage-4 transformed (training median imputation, clipping, selected log1p)",
        "features": EXPLAIN_FEATURES,
        "numpy_seed": SHAP_SEED,
        "additivity_max_abs_difference": additivity_max_abs,
        "local_case_rule": "within each true class, probability closest to that class median",
        "software": {"shap": shap.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    (DATA_DIR / "F11_explainer_parameters.json").write_text(
        json.dumps(_safe(parameters), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "x_train_18": x_train_18,
        "x_explain": x_explain,
        "values": values,
        "base_values": base_values,
        "predicted": predicted,
        "global_table": global_table,
        "representatives": representatives_table,
        "parameters": parameters,
    }


def _strict_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1), method="linear").astype(float)
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = np.nextafter(edges[index - 1], np.inf)
    return edges


def _weighted_average(values: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    numerator = np.sum(values * weights, axis=axis)
    denominator = np.sum(weights, axis=axis)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def _ale_pair(predictor: WeightedPredictor, frame: pd.DataFrame, first: str, second: str) -> dict[str, Any]:
    x = frame.loc[:, EXPLAIN_FEATURES].reset_index(drop=True)
    first_values = x[first].to_numpy(dtype=float)
    second_values = x[second].to_numpy(dtype=float)
    first_edges = _strict_quantile_edges(first_values, ALE_BINS)
    second_edges = _strict_quantile_edges(second_values, ALE_BINS)
    first_bin = np.clip(np.searchsorted(first_edges, first_values, side="right") - 1, 0, ALE_BINS - 1)
    second_bin = np.clip(np.searchsorted(second_edges, second_values, side="right") - 1, 0, ALE_BINS - 1)
    counts = np.zeros((ALE_BINS, ALE_BINS), dtype=int)
    local = np.full((ALE_BINS, ALE_BINS), np.nan, dtype=float)
    for i in range(ALE_BINS):
        for j in range(ALE_BINS):
            idx = np.flatnonzero((first_bin == i) & (second_bin == j))
            counts[i, j] = len(idx)
            if len(idx) == 0:
                continue
            base = x.iloc[idx].copy()
            lo_lo, hi_lo, lo_hi, hi_hi = base.copy(), base.copy(), base.copy(), base.copy()
            lo_lo[first], lo_lo[second] = first_edges[i], second_edges[j]
            hi_lo[first], hi_lo[second] = first_edges[i + 1], second_edges[j]
            lo_hi[first], lo_hi[second] = first_edges[i], second_edges[j + 1]
            hi_hi[first], hi_hi[second] = first_edges[i + 1], second_edges[j + 1]
            contrast = (
                predictor.predict(hi_hi) - predictor.predict(lo_hi)
                - predictor.predict(hi_lo) + predictor.predict(lo_lo)
            )
            local[i, j] = float(np.mean(contrast))

    accumulated = np.cumsum(np.cumsum(np.nan_to_num(local, nan=0.0), axis=0), axis=1)
    weights = counts.astype(float)
    row_mean = _weighted_average(accumulated, weights, axis=1)
    col_mean = _weighted_average(accumulated, weights, axis=0)
    total = weights.sum()
    overall = float(np.sum(accumulated * weights) / total) if total > 0 else 0.0
    interaction = accumulated - row_mean[:, None] - col_mean[None, :] + overall
    interaction[counts == 0] = np.nan
    finite = counts > 0
    strength = float(np.sqrt(np.sum(weights[finite] * interaction[finite] ** 2) / np.sum(weights[finite])))
    first_centers = (first_edges[:-1] + first_edges[1:]) / 2.0
    second_centers = (second_edges[:-1] + second_edges[1:]) / 2.0
    rows = []
    for i in range(ALE_BINS):
        for j in range(ALE_BINS):
            rows.append({
                "feature_1": first,
                "feature_2": second,
                "feature_1_bin": i + 1,
                "feature_2_bin": j + 1,
                "feature_1_lower": first_edges[i],
                "feature_1_upper": first_edges[i + 1],
                "feature_1_center": first_centers[i],
                "feature_2_lower": second_edges[j],
                "feature_2_upper": second_edges[j + 1],
                "feature_2_center": second_centers[j],
                "cell_n": int(counts[i, j]),
                "local_second_difference": local[i, j],
                "accumulated_effect": accumulated[i, j],
                "interaction_ale": interaction[i, j],
                "empty_cell_masked": bool(counts[i, j] == 0),
            })
    return {
        "feature_1": first,
        "feature_2": second,
        "first_edges": first_edges,
        "second_edges": second_edges,
        "counts": counts,
        "interaction": interaction,
        "strength": strength,
        "rows": rows,
    }


def _run_interactions(fit: dict[str, Any], shap_result: dict[str, Any]) -> dict[str, Any]:
    top_features = shap_result["global_table"].head(6)["feature"].tolist()
    surfaces = []
    strength_rows = []
    surface_rows = []
    for first, second in combinations(top_features, 2):
        result = _ale_pair(fit["predictor"], shap_result["x_explain"], first, second)
        surfaces.append(result)
        strength_rows.append({
            "feature_1": first,
            "feature_2": second,
            "feature_1_group": FEATURE_GROUPS[first],
            "feature_2_group": FEATURE_GROUPS[second],
            "interaction_strength": result["strength"],
            "n_bins_each_axis": ALE_BINS,
            "occupied_cells": int(np.sum(result["counts"] > 0)),
            "empty_cells": int(np.sum(result["counts"] == 0)),
            "method": "model-agnostic second-order 2D ALE; not formal SHAP interaction",
        })
        surface_rows.extend(result["rows"])
    strengths = pd.DataFrame(strength_rows).sort_values(
        ["interaction_strength", "feature_1", "feature_2"], ascending=[False, True, True], kind="mergesort"
    ).reset_index(drop=True)
    strengths.insert(0, "rank", np.arange(1, len(strengths) + 1))
    strengths.to_csv(TABLE_DIR / "F11_ale_interaction_strength.csv", index=False)
    pd.DataFrame(surface_rows).to_csv(DATA_DIR / "F11_ale_2d_surface_values.csv", index=False)
    strengths.head(5).to_csv(TABLE_DIR / "F11_ale_top_pairs.csv", index=False)
    lookup = {(item["feature_1"], item["feature_2"]): item for item in surfaces}
    top_two = [lookup[(row.feature_1, row.feature_2)] for row in strengths.head(2).itertuples(index=False)]
    parameters = {
        "split_design": split_metadata(),
        "method": "finite-grid model-agnostic second-order 2D accumulated local effects",
        "formal_shap_interaction": False,
        "interpretation": "Interaction effects are 2D ALE estimates, not SHAP interaction values.",
        "ranking_source": "top six features by Weighted-voting permutation SHAP mean absolute value",
        "top_six_features": top_features,
        "pair_n": len(strengths),
        "empirical_quantile_bins_per_axis": ALE_BINS,
        "empty_cell_policy": "excluded from centering/strength and masked in surfaces",
        "tie_policy": "non-increasing empirical edges advanced by one floating-point step",
        "centering": "empirical occupancy-weighted row/column double centering",
        "evaluation_dataset": "A_holdout_clean Stage-4 transformed inputs",
        "numpy_seed": ALE_SEED,
    }
    (DATA_DIR / "F11_ale_parameters.json").write_text(
        json.dumps(_safe(parameters), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"top_features": top_features, "strengths": strengths, "surfaces": surfaces, "top_two": top_two,
            "parameters": parameters}


def _display(feature: str) -> str:
    return DISPLAY_FEATURE.get(feature, feature)


def _plot_global(ax: plt.Axes, shap_result: dict[str, Any]) -> None:
    table = shap_result["global_table"].iloc[::-1].reset_index(drop=True)
    y = np.arange(len(table))
    colors = [FEATURE_GROUP_COLORS[group] for group in table["feature_group"]]
    ax.barh(y, table["mean_abs_shap"], color=colors, edgecolor="white", linewidth=0.35)
    ax.set_yticks(y, [_display(value) for value in table["feature"]])
    for tick, group in zip(ax.get_yticklabels(), table["feature_group"]):
        tick.set_color(FEATURE_GROUP_COLORS[group])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    handles = [Line2D([0], [0], marker="s", linestyle="", color=FEATURE_GROUP_COLORS[g], label=g, markersize=6)
               for g in ("I", "M", "N")]
    ax.legend(handles=handles, title="Feature group", loc="lower right", frameon=False, ncol=1,
              handletextpad=0.3, borderaxespad=0.2)


def _plot_beeswarm(ax: plt.Axes, shap_result: dict[str, Any], fig: plt.Figure) -> None:
    order = shap_result["global_table"]["feature"].tolist()
    values = shap_result["values"]
    x_data = shap_result["x_explain"]
    feature_index = {feature: idx for idx, feature in enumerate(EXPLAIN_FEATURES)}
    rng = np.random.default_rng(SHAP_SEED)
    cmap = mpl.colormaps["coolwarm"]
    for row, feature in enumerate(order):
        idx = feature_index[feature]
        raw = x_data[feature].to_numpy(dtype=float)
        low, high = np.quantile(raw, [0.05, 0.95])
        if high <= low:
            color_values = np.full(len(raw), 0.5)
        else:
            color_values = np.clip((raw - low) / (high - low), 0.0, 1.0)
        jitter = rng.normal(0.0, 0.095, size=len(raw))
        jitter = np.clip(jitter, -0.28, 0.28)
        ax.scatter(values[:, idx], row + jitter, c=color_values, cmap=cmap, vmin=0, vmax=1,
                   s=9, alpha=0.72, linewidths=0)
    ax.axvline(0, color="#6B6B6B", linewidth=0.8, linestyle="--")
    ax.set_yticks(np.arange(len(order)), [_display(feature) for feature in order])
    ax.invert_yaxis()
    for tick, feature in zip(ax.get_yticklabels(), order):
        tick.set_color(FEATURE_GROUP_COLORS[FEATURE_GROUPS[feature]])
    ax.set_xlabel("SHAP value (change in predicted probability)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.5, alpha=0.65)
    sm = mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    colorbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.015)
    if getattr(colorbar, "solids", None) is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.set_label("Transformed feature value", labelpad=-3)


def _plot_dependence(ax: plt.Axes, feature: str, shap_result: dict[str, Any]) -> None:
    idx = EXPLAIN_FEATURES.index(feature)
    x = shap_result["x_explain"][feature].to_numpy(dtype=float)
    y = shap_result["values"][:, idx]
    color = FEATURE_GROUP_COLORS[FEATURE_GROUPS[feature]]
    ax.scatter(x, y, color=color, s=15, alpha=0.62, edgecolor="white", linewidth=0.25)
    ax.axhline(0, color="#6B6B6B", linestyle="--", linewidth=0.75)
    ax.set_xlabel(f"{_display(feature)} (Stage-4 scale)")
    ax.set_ylabel("SHAP value")
    ax.grid(color="#E5E5E5", linewidth=0.5, alpha=0.65)
    ax.text(0.98, 0.97, f"Group {FEATURE_GROUPS[feature]}", transform=ax.transAxes,
            ha="right", va="top", color=color, fontsize=8.5, fontweight="bold")


def _plot_local(ax: plt.Axes, selected: pd.Series, shap_result: dict[str, Any]) -> None:
    sample_index = int(selected["sample_index"])
    contributions = pd.DataFrame({
        "feature": EXPLAIN_FEATURES,
        "value": shap_result["values"][sample_index, :],
    })
    contributions["abs"] = contributions["value"].abs()
    contributions = contributions.sort_values(["abs", "feature"], ascending=[False, True], kind="mergesort")
    top = contributions.head(8).copy()
    other_value = float(contributions.iloc[8:]["value"].sum())
    top = pd.concat([top, pd.DataFrame({"feature": ["Other features"], "value": [other_value], "abs": [abs(other_value)]})],
                    ignore_index=True)
    top = top.iloc[::-1].reset_index(drop=True)
    colors = np.where(top["value"].to_numpy(dtype=float) >= 0, "#B2182B", "#2166AC")
    ax.barh(np.arange(len(top)), top["value"], color=colors, alpha=0.9, edgecolor="white", linewidth=0.3)
    ax.set_yticks(np.arange(len(top)), [_display(feature) for feature in top["feature"]])
    ax.axvline(0, color="#4D4D4D", linewidth=0.75)
    ax.set_xlabel(
        "SHAP contribution\n"
        f"{selected['class_label']} representative\n"
        f"Base = {float(selected['base_value']):.3f}; prediction = {float(selected['weighted_probability']):.3f}",
        fontsize=7.8,
        linespacing=0.95,
    )
    ax.set_ylabel("Feature")
    ax.grid(axis="x", color="#E2E2E2", linewidth=0.5, alpha=0.7)


def _plot_interaction_matrix(ax: plt.Axes, interaction_result: dict[str, Any], fig: plt.Figure) -> None:
    features = interaction_result["top_features"]
    size = len(features)
    matrix = np.full((size, size), np.nan, dtype=float)
    lookup = {}
    for row in interaction_result["strengths"].itertuples(index=False):
        lookup[frozenset((row.feature_1, row.feature_2))] = float(row.interaction_strength)
    for i, first in enumerate(features):
        for j, second in enumerate(features):
            if i != j:
                matrix[i, j] = lookup[frozenset((first, second))]
    masked = np.ma.masked_invalid(matrix)
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    edges = np.arange(size + 1)
    mesh = ax.pcolormesh(edges, edges, masked, cmap="YlGnBu", vmin=0, vmax=max(vmax, 1e-8),
                         edgecolors="white", linewidth=0.7, shading="flat")
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_xticks(np.arange(size) + 0.5, [_display(item) for item in features], rotation=45, ha="right")
    ax.set_yticks(np.arange(size) + 0.5, [_display(item) for item in features])
    for i in range(size):
        for j in range(size):
            if i != j and np.isfinite(matrix[i, j]):
                color = "white" if matrix[i, j] > 0.58 * vmax else "black"
                ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=6.8, color=color)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.035)
    if getattr(colorbar, "solids", None) is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label("2D-ALE interaction strength")


def _plot_surface(ax: plt.Axes, surface: dict[str, Any], fig: plt.Figure) -> None:
    values = np.ma.masked_invalid(surface["interaction"].T)
    finite_values = values.compressed()
    vmax = max(float(np.max(np.abs(finite_values))) if finite_values.size else 0.0, 1e-7)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    mesh = ax.pcolormesh(surface["first_edges"], surface["second_edges"], values,
                         cmap="RdBu_r", norm=norm, edgecolors="white", linewidth=0.55, shading="flat")
    ax.set_xlabel(f"{_display(surface['feature_1'])} (Stage-4 scale)")
    ax.set_ylabel(f"{_display(surface['feature_2'])} (Stage-4 scale)")
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.025)
    if getattr(colorbar, "solids", None) is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label("Centered interaction ALE")
    ax.text(0.02, 0.98,
            f"{_display(surface['feature_1'])} × {_display(surface['feature_2'])}\nStrength = {surface['strength']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8})


def _make_figure(shap_result: dict[str, Any], interaction_result: dict[str, Any]) -> tuple[Path, Path]:
    apply_style()
    fig = plt.figure(figsize=(13.2, 16.7), layout="constrained")
    outer = fig.add_gridspec(4, 4, height_ratios=[1.62, 1.02, 1.18, 1.18])
    axes: list[plt.Axes] = []

    ax_a = fig.add_subplot(outer[0, 0:2]); axes.append(ax_a)
    ax_b = fig.add_subplot(outer[0, 2:4]); axes.append(ax_b)
    _plot_global(ax_a, shap_result)
    _plot_beeswarm(ax_b, shap_result, fig)

    top_four = shap_result["global_table"].head(4)["feature"].tolist()
    for column, feature in enumerate(top_four):
        ax = fig.add_subplot(outer[1, column]); axes.append(ax)
        _plot_dependence(ax, feature, shap_result)

    reps = shap_result["representatives"].sort_values("true_class", kind="mergesort").reset_index(drop=True)
    ax_g = fig.add_subplot(outer[2, 0]); axes.append(ax_g)
    ax_h = fig.add_subplot(outer[2, 1]); axes.append(ax_h)
    _plot_local(ax_g, reps.iloc[0], shap_result)
    _plot_local(ax_h, reps.iloc[1], shap_result)
    ax_i = fig.add_subplot(outer[2, 2:4]); axes.append(ax_i)
    _plot_interaction_matrix(ax_i, interaction_result, fig)

    ax_j = fig.add_subplot(outer[3, 0:2]); axes.append(ax_j)
    ax_k = fig.add_subplot(outer[3, 2:4]); axes.append(ax_k)
    _plot_surface(ax_j, interaction_result["top_two"][0], fig)
    _plot_surface(ax_k, interaction_result["top_two"][1], fig)

    for letter, ax in zip("ABCDEFGHIJK", axes):
        panel_label(ax, letter, x=-0.13 if letter not in "AB" else -0.10, y=1.035)
        ax.tick_params(direction="out")
    pdf, png = save_fig(fig, "unnumbered_ale_explainability_not_manuscript_fig7")
    plt.close(fig)
    return pdf, png


def _load_saved_results_for_render() -> tuple[dict[str, Any], dict[str, Any]]:
    long = pd.read_csv(DATA_DIR / "F11_permutation_shap_long.csv")
    base = pd.read_csv(DATA_DIR / "F11_shap_base_values.csv").sort_values("sample_index", kind="mergesort")
    global_table = pd.read_csv(TABLE_DIR / "F11_shap_global_importance.csv")
    representatives = pd.read_csv(TABLE_DIR / "F11_local_representatives.csv")
    sample_order = sorted(long["sample_index"].unique().tolist())
    values = long.pivot(index="sample_index", columns="feature", values="shap_value").reindex(
        index=sample_order, columns=EXPLAIN_FEATURES
    ).to_numpy(dtype=float)
    x_explain = long.pivot(index="sample_index", columns="feature", values="transformed_value").reindex(
        index=sample_order, columns=EXPLAIN_FEATURES
    )
    shap_result = {
        "x_explain": x_explain,
        "values": values,
        "base_values": base["base_value"].to_numpy(dtype=float),
        "predicted": base["weighted_probability"].to_numpy(dtype=float),
        "global_table": global_table,
        "representatives": representatives,
    }

    strengths = pd.read_csv(TABLE_DIR / "F11_ale_interaction_strength.csv")
    surface_table = pd.read_csv(DATA_DIR / "F11_ale_2d_surface_values.csv")
    parameters = json.loads((DATA_DIR / "F11_ale_parameters.json").read_text(encoding="utf-8"))
    surfaces = []
    for (first, second), part in surface_table.groupby(["feature_1", "feature_2"], sort=False):
        part = part.sort_values(["feature_1_bin", "feature_2_bin"], kind="mergesort")
        first_edges = np.r_[
            part.drop_duplicates("feature_1_bin").sort_values("feature_1_bin")["feature_1_lower"].to_numpy(dtype=float),
            part["feature_1_upper"].max(),
        ]
        second_edges = np.r_[
            part.drop_duplicates("feature_2_bin").sort_values("feature_2_bin")["feature_2_lower"].to_numpy(dtype=float),
            part["feature_2_upper"].max(),
        ]
        interaction = np.full((ALE_BINS, ALE_BINS), np.nan, dtype=float)
        counts = np.zeros((ALE_BINS, ALE_BINS), dtype=int)
        for row in part.itertuples(index=False):
            i, j = int(row.feature_1_bin) - 1, int(row.feature_2_bin) - 1
            interaction[i, j] = float(row.interaction_ale) if pd.notna(row.interaction_ale) else np.nan
            counts[i, j] = int(row.cell_n)
        strength = float(strengths.loc[
            (strengths["feature_1"] == first) & (strengths["feature_2"] == second), "interaction_strength"
        ].iloc[0])
        surfaces.append({
            "feature_1": first, "feature_2": second, "first_edges": first_edges,
            "second_edges": second_edges, "interaction": interaction, "counts": counts, "strength": strength,
        })
    lookup = {(item["feature_1"], item["feature_2"]): item for item in surfaces}
    top_two = [lookup[(row.feature_1, row.feature_2)] for row in strengths.head(2).itertuples(index=False)]
    interaction_result = {
        "top_features": parameters["top_six_features"],
        "strengths": strengths,
        "surfaces": surfaces,
        "top_two": top_two,
        "parameters": parameters,
    }
    return shap_result, interaction_result


def _refresh_figure_hashes(pdf: Path, png: Path) -> None:
    manifest_path = DATA_DIR / "F11_run_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("outputs", []):
        candidate = Path(item["path"])
        if candidate in {pdf, png}:
            item["sha256"] = _sha256(candidate)
            item["size_bytes"] = candidate.stat().st_size
    manifest["figure_render_refreshed"] = True
    manifest_path.write_text(json.dumps(_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


def _save_model_package(fit: dict[str, Any]) -> dict[str, Any]:
    package_path = MODEL_DIR / "F11_weighted_voting_model.joblib"
    package = {
        "split_design": split_metadata(),
        "model_name": "weighted_voting_cv_auc",
        "primary_model": True,
        "layer": "I_M_N",
        "member_order": NATIVE_MEMBERS,
        "weights": fit["weights"],
        "selected_direct": FROZEN_DIRECT,
        "fixed_composites": FIXED_COMPOSITES,
        "explain_features": EXPLAIN_FEATURES,
        "stage4_features": list(ALL_FEATURES),
        "preprocessor": fit["preprocessor"],
        "estimators": fit["estimators"],
        "training_scope": "A_development",
        "training_seed": MASTER_SEED,
        "model_seed_rule": "derive_seed(20260902, 'holdout_fit', 'model', model_name)",
        "scope_id": "common_scheme",
    }
    joblib.dump(package, package_path, compress=3)
    loaded = joblib.load(package_path)
    reloaded_predictor = WeightedPredictor(loaded["estimators"], loaded["weights"])
    x_check = fit["transformed"]["A_holdout_clean"].iloc[:25]
    before = fit["predictor"].predict_full_stage4(x_check)
    after = reloaded_predictor.predict_full_stage4(x_check)
    reload_max_abs = float(np.max(np.abs(before - after)))
    if reload_max_abs > 1e-12:
        raise RuntimeError(f"serialized model prediction check failed: {reload_max_abs}")
    manifest = {
        "split_design": split_metadata(),
        "artifact": str(package_path),
        "sha256": _sha256(package_path),
        "size_bytes": package_path.stat().st_size,
        "joblib_reload_max_abs_difference": reload_max_abs,
        "member_n": len(NATIVE_MEMBERS),
        "member_order": NATIVE_MEMBERS,
        "weights": fit["weights"],
        "selected_direct": FROZEN_DIRECT,
        "fixed_composites": FIXED_COMPOSITES,
        "stage4_preprocessor_fingerprint": fit["preprocessor"].fingerprint,
        "runtime_note": "Loading requires the archived project stage4_pipeline.py/model_factory.py dependencies.",
    }
    (MODEL_DIR / "F11_weighted_voting_model_manifest.json").write_text(
        json.dumps(_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _save_manifest(pdf: Path, png: Path, fit: dict[str, Any], shap_result: dict[str, Any],
                   interaction_result: dict[str, Any], audit: pd.DataFrame, model_manifest: dict[str, Any],
                   started: float) -> None:
    panel_rows = [
        ("A", "Global mean absolute permutation SHAP", "A evaluation set", "F11_shap_global_importance.csv"),
        ("B", "Permutation SHAP beeswarm", "A evaluation set", "F11_permutation_shap_long.csv"),
        ("C-F", "Top-four permutation SHAP dependence plots", "A evaluation set", "F11_permutation_shap_long.csv"),
        ("G-H", "Prespecified representative local explanations", "A evaluation set", "F11_local_representatives.csv"),
        ("I", "Model-agnostic 2D-ALE interaction-strength matrix; not SHAP interaction", "A evaluation set", "F11_ale_interaction_strength.csv"),
        ("J-K", "Top-two model-agnostic second-order 2D-ALE surfaces", "A evaluation set", "F11_ale_2d_surface_values.csv"),
    ]
    rows = []
    for panel, content, dataset, source in panel_rows:
        rows.append({
            "figure": "unnumbered ALE display; manuscript Figure 7 is the SHAP/SII figure",
            "panel": panel,
            "content": content,
            "model": "Weighted voting (Primary)",
            "layer": "I + M + N",
            "dataset": dataset,
            "method": "Permutation SHAP" if panel in {"A", "B", "C-F", "G-H"} else "second-order 2D ALE",
            "data_source": source,
            "output_png": str(png),
            "output_pdf": str(pdf),
        })
    pd.DataFrame(rows).to_csv(TABLE_DIR / "F11_manifest.csv", index=False)
    caption = (
        "Weighted-voting model explanation. (A) Global feature importance quantified by mean absolute model-agnostic "
        "permutation SHAP values. (B) SHAP beeswarm plot. (C-F) Dependence plots for the four highest-ranked "
        "features. (G-H) Local explanations for one representative patient from each observed class, selected "
        "a priori as the prediction closest to the within-class median. (I) Interaction-strength matrix among "
        "the six highest-ranked features. (J-K) Two-dimensional accumulated local effect surfaces for the two "
        "strongest pairs. Interaction panels are model-agnostic second-order 2D ALE estimates and are not formal "
        "SHAP interaction values. Empty empirical-quantile cells are masked. Feature groups: I, inflammation; "
        "M, immune; N, nutrition/metabolism."
    )
    (TABLE_DIR / "F11_figure_caption.md").write_text(caption + "\n", encoding="utf-8")
    outputs = sorted(
        [path for directory in (DATA_DIR, TABLE_DIR, MODEL_DIR, FIGURE_DIR) for path in directory.glob("F11*") if path.is_file()]
    )
    run_manifest = {
        "split_design": split_metadata(),
        "run_name": "F11_weighted_voting_explainability",
        "primary_model": "weighted_voting_cv_auc",
        "layer": "I_M_N",
        "existing_artifacts_modified": False,
        "refit_authorization": "User explicitly authorized refitting on 2026-09-11.",
        "training_scope": "A development only",
        "member_n": len(NATIVE_MEMBERS),
        "member_order": NATIVE_MEMBERS,
        "weights": fit["weights"],
        "prediction_reproducibility": audit.to_dict("records"),
        "shap": shap_result["parameters"],
        "interaction": interaction_result["parameters"],
        "model_package": model_manifest,
        "figure_caption": caption,
        "style": {
            "font": "Times New Roman",
            "language": "English",
            "figure_title": False,
            "panel_labels": "italic bold",
            "png_dpi": 600,
            "pdf_fonttype": 42,
            "vector_pdf": True,
        },
        "inputs": {str(path): _sha256(path) for path in (
            INPUT_WORKBOOK,
            A_REFERENCE,
        )},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "shap": shap.__version__,
            "joblib": joblib.__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
        "outputs": [{"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size} for path in outputs],
    }
    (DATA_DIR / "F11_run_manifest.json").write_text(
        json.dumps(_safe(run_manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(verify_only: bool = False, render_only: bool = False) -> None:
    started = time.perf_counter()
    if render_only:
        shap_result, interaction_result = _load_saved_results_for_render()
        pdf, png = _make_figure(shap_result, interaction_result)
        _refresh_figure_hashes(pdf, png)
        print(json.dumps({"status": "rendered", "pdf": str(pdf), "png": str(png)}, ensure_ascii=False, indent=2))
        return
    fit = _load_and_fit()
    _, audit = _prediction_audit(fit)
    if verify_only:
        print(audit.to_string(index=False))
        return
    model_manifest = _save_model_package(fit)
    shap_result = _run_shap(fit)
    interaction_result = _run_interactions(fit, shap_result)
    pdf, png = _make_figure(shap_result, interaction_result)
    _save_manifest(pdf, png, fit, shap_result, interaction_result, audit, model_manifest, started)
    print(json.dumps({
        "status": "complete",
        "pdf": str(pdf),
        "png": str(png),
        "prediction_max_abs_difference": float(audit["max_abs_difference"].max()),
        "shap_additivity_max_abs_difference": shap_result["parameters"]["additivity_max_abs_difference"],
        "top_four_features": shap_result["global_table"].head(4)["feature"].tolist(),
        "top_two_interactions": interaction_result["strengths"].head(2)[["feature_1", "feature_2"]].values.tolist(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only and args.render_only:
        parser.error("--verify-only and --render-only are mutually exclusive")
    run(verify_only=args.verify_only, render_only=args.render_only)
