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

import hashlib
import json
import math
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess


SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_ROOT = SCRIPT_DIR.parent
SUBMISSION_ROOT = REVISION_ROOT.parent
PROJECT_ROOT = data_root()


_ARCHIVE_MODEL_SRC = Path(__file__).resolve().parents[2] / "01_methods" / "models" / "src"
sys.path.insert(0, str(_ARCHIVE_MODEL_SRC))
from fold_selection import fit_scope_selection, selected_layer_specs, SELECTION_PROTOCOL, require_selection_protocol

import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents
                     if (p / "common" / "manuscript_source_paths.py").is_file())
if str(_archive_root) not in _archive_sys.path:
    _archive_sys.path.insert(0, str(_archive_root))
from common.manuscript_source_paths import activate_archive_sources
activate_archive_sources(__file__)

from model_factory import MODEL_NAMES, build_model, derive_seed as model_seed
from nature_style import (
    DATASET_COLORS,
    DATASET_LABELS,
    DATASET_ORDER,
    LAYER_LABELS,
    LAYER_ORDER,
    apply_style,
    panel_label,
    save_fig,
)
from run_a_center_fusions import (
    CV_SEED,
    MASTER_SEED,
    _clean_folds,
    _prepare_center,
    _subset,
)

def _add_audit_columns(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = frame.copy()
    out["scenario"] = scenario
    if "source_record_id" not in out.columns:
        out["source_record_id"] = out["record_id"].astype(str)
    if "duplicate_flag" not in out.columns:
        out["duplicate_flag"] = False
    return out
from stage4_pipeline import ALL_FEATURES, DIRECT_FEATURES, FrozenPreprocessor, load_base_workbook


INPUT_WORKBOOK = cohort_workbook()

EXISTING_A_CV = data_root() / "fusion_results" / "A_fusions_CV_holdout" / "fusion_predictions_A_CV.csv"
EXISTING_A_HOLDOUT = data_root() / "fusion_results" / "A_fusions_CV_holdout" / "fusion_predictions_A_holdout_clean.csv"
EXISTING_BC = data_root() / "fusion_results" / "three_fusions_BC" / "fusion_predictions_B_C.csv"

DATA_DIR = fusion_results() / "data"
TABLE_DIR = fusion_results() / "tables"
FIGURE_DIR = publication_results() / "figures" / "main"

EXPECTED_DIRECT = (
    "WBC",
    "PLT",
    "LDH",
    "PCT",
    "EO#",
    "EO%",
    "ALB",
    "TP",
    "GLB",
    "TC",
    "LDL-C",
    "MCH",
    "MCHC",
)
FIXED_COMPOSITES = ("SIRI", "LMR", "GAR", "PNI", "HALP")
NATIVE_MEMBERS = tuple(name for name in MODEL_NAMES if name != "rbf_svm")
WEIGHTED_NAME = "weighted_voting_cv_auc"

_BOOTSTRAP_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "manuscript_bootstrap.py").is_file())
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))
from common.manuscript_bootstrap import BOOTSTRAP_PROTOCOL, bootstrap_seed, canonical_rows, shared_bootstrap_indices, bootstrap_identity, threshold_metric_values, verify_manuscript_threshold


def _development_threshold() -> dict[str, Any]:
    frame = pd.read_csv(EXISTING_A_CV)
    primary = canonical_rows(frame.loc[frame["fusion"] == WEIGHTED_NAME])
    return verify_manuscript_threshold(
        primary["y_SCLC"].to_numpy(dtype=int),
        primary["score"].to_numpy(dtype=float),
        strict="--allow-threshold-mismatch" not in sys.argv[1:],
    )

BOOTSTRAP_N = 2000
BOOTSTRAP_NAMESPACE = "F6_F8_weighted_layers_v1"
LOWESS_FRAC = 0.75
LOWESS_IT = 0

LAYER_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "I",
            {
                "label": "I",
                "direct": ("WBC", "PLT", "LDH", "PCT"),
                "composites": ("SIRI",),
            },
        ),
        (
            "I_M",
            {
                "label": "I + M",
                "direct": ("WBC", "PLT", "LDH", "PCT", "EO#", "EO%"),
                "composites": ("SIRI", "LMR", "GAR"),
            },
        ),
        (
            "I_M_N",
            {
                "label": "I + M + N",
                "direct": EXPECTED_DIRECT,
                "composites": FIXED_COMPOSITES,
            },
        ),
    ]
)

COMPARISONS = (
    ("I", "I_M", "I → I + M"),
    ("I_M", "I_M_N", "I + M → I + M + N"),
    ("I", "I_M_N", "I → I + M + N"),
)


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
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


def _seed(*parts: object) -> int:
    return model_seed(MASTER_SEED, BOOTSTRAP_NAMESPACE, *parts)


class LayerSelector(BaseEstimator, TransformerMixin):

    def __init__(self, selected_direct: Sequence[str], selected_composites: Sequence[str]):
        self.selected_direct = selected_direct
        self.selected_composites = selected_composites

    def fit(self, X: Any, y: Any = None) -> "LayerSelector":
        columns = list(self.selected_direct) + list(self.selected_composites)
        if len(columns) != len(set(columns)):
            raise ValueError("duplicate feature in layer selector")
        unknown = [column for column in columns if column not in ALL_FEATURES]
        if unknown:
            raise ValueError(f"unknown Stage-4 features in layer selector: {unknown}")
        self.selected_features_ = tuple(columns)
        self.n_features_in_ = len(ALL_FEATURES)
        self.n_features_out_ = len(columns)
        self.feature_names_in_ = np.asarray(ALL_FEATURES, dtype=object)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("LayerSelector must be fitted before transform")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=ALL_FEATURES)
        return X.loc[:, list(self.selected_features_)].copy()

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("LayerSelector must be fitted before get_feature_names_out")
        return np.asarray(self.selected_features_, dtype=object)


def _layer_weights(auc_by_model: Mapping[str, float]) -> dict[str, float]:
    missing = [name for name in NATIVE_MEMBERS if name not in auc_by_model]
    if missing:
        raise RuntimeError(f"layer CV lacks native members: {missing}")
    values = np.asarray([float(auc_by_model[name]) for name in NATIVE_MEMBERS], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise RuntimeError("all fixed member AUC weights must be finite and positive")
    values /= values.sum()
    result = {name: float(value) for name, value in zip(NATIVE_MEMBERS, values)}
    if len(result) != 13 or not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("weighted-voting member contract is not exactly 13 normalized members")
    return result


def _check_stage4(frame: pd.DataFrame, name: str) -> None:
    if list(frame.columns) != list(ALL_FEATURES):
        raise RuntimeError(f"Stage-4 feature order mismatch for {name}")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise RuntimeError(f"non-finite Stage-4 values remain for {name}")


def _prepare_external(frame: pd.DataFrame, center: str) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    if set(out["center"].astype(str)) != {center}:
        raise ValueError(f"unexpected centre values in {center} external frame")
    if "missing_direct_n" in out.columns:
        out = out.loc[pd.to_numeric(out["missing_direct_n"], errors="coerce") < len(DIRECT_FEATURES)].copy()
    out["record_id"] = out["record_id"].astype(str)
    out["y_SCLC"] = pd.to_numeric(out["y_SCLC"], errors="raise").astype(int)
    if out["y_SCLC"].nunique() != 2:
        raise ValueError(f"{center} external frame lacks both outcome classes")
    return out.reset_index(drop=True)


def _load_cohorts() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    data = validate_base_cohorts(load_base_workbook(INPUT_WORKBOOK))
    a = _prepare_center(data["A"], "A")
    all_indices = np.arange(len(a), dtype=int)
    train_idx, holdout_idx = split_center_a_indices(a)
    train_idx = np.sort(train_idx)
    holdout_idx = np.sort(holdout_idx)
    train = _add_audit_columns(a.iloc[train_idx].reset_index(drop=True), scenario="A_dev_clean")
    clean_holdout = a.iloc[holdout_idx].reset_index(drop=True)

    b_external = _prepare_external(data["B"], "B")
    c_external = _prepare_external(data["C"], "C")
    if len(b_external) != 329 or len(c_external) != 313:
        raise RuntimeError("B/C external inputs do not match the frozen 329/313 cohorts")

    cohorts = {
        "A_holdout_clean": clean_holdout.reset_index(drop=True),
        "B_external": b_external,
        "C_external": c_external,
    }
    split_info = {
        "master_seed": MASTER_SEED, "split_design": split_metadata(),
        "train_n": len(train),
        "clean_holdout_n": len(clean_holdout),
        "B_external_n": len(b_external),
        "C_external_n": len(c_external),
        "train_record_id_sha256": hashlib.sha256("\n".join(train["record_id"].astype(str)).encode()).hexdigest(),
    }
    return train, cohorts, split_info


def _fit_weighted_cv(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    folds, _fold_assignment = _clean_folds(train, 10)
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    dataset = "A_dev_cv_clean"
    for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        fold_train = _subset(train, train_idx)
        fold_valid = _subset(train, valid_idx)

        preprocessor = FrozenPreprocessor.fit(fold_train)
        x_train, train_qc = preprocessor.transform(fold_train)
        x_valid, valid_qc = preprocessor.transform(fold_valid)
        _check_stage4(x_train, f"A CV fold {fold_number} training")
        _check_stage4(x_valid, f"A CV fold {fold_number} validation")
        y_train = fold_train["y_SCLC"].to_numpy(dtype=int)
        y_valid = fold_valid["y_SCLC"].to_numpy(dtype=int)
        scope_id = f"A_dev_{dataset}_fold_{fold_number}"

        selection = fit_scope_selection(x_train, y_train, fold_number)
        for layer_name, layer_spec in selected_layer_specs(selection).items():
            member_probabilities: list[np.ndarray] = []
            for model_name in NATIVE_MEMBERS:
                estimator = build_model(
                    model_name,
                    seed=model_seed(MASTER_SEED, "cv", fold_number, layer_name, "model", model_name),
                    selector=LayerSelector(layer_spec["direct"], layer_spec["composites"]),
                    scope_id=scope_id,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(x_train, y_train)
                if not hasattr(estimator, "predict_proba"):
                    raise RuntimeError(f"native member {model_name} lacks predict_proba")
                probability = np.asarray(estimator.predict_proba(x_valid), dtype=float)[:, 1]
                if not np.isfinite(probability).all():
                    raise RuntimeError(f"non-finite probability for {model_name}, {layer_name}, fold {fold_number}")
                member_probabilities.append(np.clip(probability, 0.0, 1.0))
            matrix = np.column_stack(member_probabilities)
            pending.append({
                "fold": fold_number,
                "layer": layer_name,
                "layer_label": layer_spec["label"],
                "y": y_valid,
                "rows": list(fold_valid.itertuples(index=False)),
                "matrix": matrix,
            })
    weight_maps: dict[str, dict[str, float]] = {}
    member_aucs: dict[str, dict[str, float]] = {}
    for layer_name in LAYER_SPECS:
        items = [item for item in pending if item["layer"] == layer_name]
        auc_by_model = {
            name: float(np.mean([roc_auc_score(item["y"], item["matrix"][:, index]) for item in items]))
            for index, name in enumerate(NATIVE_MEMBERS)
        }
        member_aucs[layer_name] = auc_by_model
        weight_maps[layer_name] = _layer_weights(auc_by_model)
    for item in pending:
        vector = np.asarray([weight_maps[item["layer"]][name] for name in NATIVE_MEMBERS], dtype=float)
        weighted = np.sum(item["matrix"] * vector.reshape(1, -1), axis=1)
        for row, y_value, probability in zip(item["rows"], item["y"], weighted):
            rows.append(
                {
                    "status": "manuscript",
                    "dataset": dataset,
                    "fold": item["fold"],
                    "layer": item["layer"],
                    "layer_label": item["layer_label"],
                    "model": WEIGHTED_NAME,
                    "record_id": str(getattr(row, "record_id")),
                    "source_record_id": str(getattr(row, "source_record_id", getattr(row, "record_id"))),
                    "duplicate_flag": bool(getattr(row, "duplicate_flag", False)),
                    "y_SCLC": int(y_value),
                    "probability": float(probability),
                }
            )
        provenance.append(
            {
                "fold": fold_number,
                "train_n": len(fold_train),
                "validation_n": len(fold_valid),
                "preprocessor_fingerprint": preprocessor.fingerprint,
                "train_qc": train_qc,
                "validation_qc": valid_qc,
                "selection": selection,
                "runtime_seconds": time.perf_counter() - fold_started,
            }
        )
    frame = pd.DataFrame(rows)
    expected_rows = len(train) * len(LAYER_SPECS)
    if len(frame) != expected_rows:
        raise RuntimeError(f"A-development prediction row count mismatch: {len(frame)} vs {expected_rows}")
    return frame, provenance, weight_maps, member_aucs


def _fit_weighted_full(
    train: pd.DataFrame,
    cohorts: Mapping[str, pd.DataFrame],
    weight_maps: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    preprocessor = FrozenPreprocessor.fit(train)
    x_train, train_qc = preprocessor.transform(train)
    _check_stage4(x_train, "full A-development training")
    transformed: dict[str, pd.DataFrame] = {}
    qc: dict[str, Any] = {}
    for dataset, cohort in cohorts.items():
        transformed[dataset], qc[dataset] = preprocessor.transform(cohort)
        _check_stage4(transformed[dataset], dataset)
    y_train = train["y_SCLC"].to_numpy(dtype=int)

    rows: list[dict[str, Any]] = []
    selection = fit_scope_selection(x_train, y_train)
    for layer_name, layer_spec in selected_layer_specs(selection).items():
        member_probabilities: dict[str, list[np.ndarray]] = {dataset: [] for dataset in cohorts}
        for model_name in NATIVE_MEMBERS:
            estimator = build_model(
                model_name,
                seed=model_seed(MASTER_SEED, "holdout_fit", layer_name, "model", model_name),
                selector=LayerSelector(layer_spec["direct"], layer_spec["composites"]),
                scope_id="common_scheme",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(x_train, y_train)
            if not hasattr(estimator, "predict_proba"):
                raise RuntimeError(f"native member {model_name} lacks predict_proba")
            for dataset, x_apply in transformed.items():
                probability = np.asarray(estimator.predict_proba(x_apply), dtype=float)[:, 1]
                if not np.isfinite(probability).all():
                    raise RuntimeError(f"non-finite probability for {model_name}, {layer_name}, {dataset}")
                member_probabilities[dataset].append(np.clip(probability, 0.0, 1.0))

        for dataset, cohort in cohorts.items():
            matrix = np.column_stack(member_probabilities[dataset])
            vector = np.asarray([weight_maps[layer_name][name] for name in NATIVE_MEMBERS], dtype=float)
            weighted = np.sum(matrix * vector.reshape(1, -1), axis=1)
            for row, probability in zip(cohort.itertuples(index=False), weighted):
                rows.append(
                    {
                        "status": "manuscript",
                        "dataset": dataset,
                        "fold": np.nan,
                        "layer": layer_name,
                        "layer_label": layer_spec["label"],
                        "model": WEIGHTED_NAME,
                        "record_id": str(getattr(row, "record_id")),
                        "source_record_id": str(getattr(row, "source_record_id", getattr(row, "record_id"))),
                        "duplicate_flag": bool(getattr(row, "duplicate_flag", False)),
                        "y_SCLC": int(getattr(row, "y_SCLC")),
                        "probability": float(probability),
                    }
                )
    frame = pd.DataFrame(rows)
    expected_rows = sum(len(cohort) for cohort in cohorts.values()) * len(LAYER_SPECS)
    if len(frame) != expected_rows:
        raise RuntimeError(f"full-fit prediction row count mismatch: {len(frame)} vs {expected_rows}")
    provenance = {
        "preprocessor_fingerprint": preprocessor.fingerprint,
        "train_qc": train_qc,
        "apply_qc": qc,
        "selection": selection,
    }
    return frame, provenance


def _existing_weighted(dataset: str) -> pd.DataFrame:
    if dataset == "A_dev_cv_clean":
        frame = pd.read_csv(EXISTING_A_CV)
        frame = frame.loc[(frame["dataset"] == dataset) & (frame["fusion"] == WEIGHTED_NAME)].copy()
        frame = frame.rename(columns={"score": "existing_probability"})
        keep = ["dataset", "fold", "record_id", "y_SCLC", "existing_probability"]
    elif dataset == "A_holdout_clean":
        frame = pd.read_csv(EXISTING_A_HOLDOUT)
        frame = frame.loc[(frame["dataset"] == dataset) & (frame["fusion"] == WEIGHTED_NAME)].copy()
        frame = frame.rename(columns={"score": "existing_probability"})
        keep = ["dataset", "record_id", "y_SCLC", "existing_probability"]
    else:
        frame = pd.read_csv(EXISTING_BC)
        frame = frame.loc[(frame["dataset"] == dataset) & (frame["model"] == WEIGHTED_NAME)].copy()
        frame = frame.rename(columns={"raw_score": "existing_probability"})
        keep = ["dataset", "record_id", "y_SCLC", "existing_probability"]
    require_selection_protocol(frame)
    return frame.loc[:, keep].reset_index(drop=True)


def _audit_i_m_n(predictions: pd.DataFrame) -> pd.DataFrame:
    audit_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        new = predictions.loc[
            (predictions["dataset"] == dataset) & (predictions["layer"] == "I_M_N")
        ].copy()
        old = _existing_weighted(dataset)
        keys = ["dataset", "record_id", "y_SCLC"]
        if dataset == "A_dev_cv_clean":
            keys.insert(1, "fold")
            new["fold"] = pd.to_numeric(new["fold"], errors="raise").astype(int)
            old["fold"] = pd.to_numeric(old["fold"], errors="raise").astype(int)
        if new.duplicated(keys).any() or old.duplicated(keys).any():
            raise RuntimeError(f"non-unique audit key for {dataset}")
        merged = new.merge(old, on=keys, how="outer", indicator=True, validate="one_to_one")
        if not (merged["_merge"] == "both").all():
            raise RuntimeError(f"new/existing record mismatch for {dataset}")
        difference = np.abs(merged["probability"].to_numpy(dtype=float) - merged["existing_probability"].to_numpy(dtype=float))
        max_difference = float(np.max(difference)) if len(difference) else float("nan")
        audit_rows.append(
            {
                "dataset": dataset,
                "n_compared": len(merged),
                "max_abs_difference": max_difference,
                "mean_abs_difference": float(np.mean(difference)),
                "tolerance": 1e-10,
                "within_tolerance": bool(max_difference <= 1e-10),
                "existing_source": str(
                    EXISTING_A_CV
                    if dataset == "A_dev_cv_clean"
                    else EXISTING_A_HOLDOUT
                    if dataset == "A_holdout_clean"
                    else EXISTING_BC
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)
    if not audit["within_tolerance"].all():
        failed = audit.loc[~audit["within_tolerance"], ["dataset", "max_abs_difference"]].to_dict("records")
        raise RuntimeError(f"I+M+N weighted-voting reproducibility audit failed: {failed}")
    return audit


def _bootstrap_indices(y: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    y = np.asarray(y, dtype=int)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("stratified bootstrap requires at least two records per class")
    rng = np.random.default_rng(int(seed))
    return [
        rng.permutation(
            np.concatenate(
                [
                    rng.choice(positive, size=len(positive), replace=True),
                    rng.choice(negative, size=len(negative), replace=True),
                ]
            )
        )
        for _ in range(n_boot)
    ]


def _metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def _percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def _metrics_with_ci(predictions: pd.DataFrame, threshold_info: dict[str, Any]) -> pd.DataFrame:
    threshold = float(threshold_info["threshold"])
    max_j = float(threshold_info["youden_j"])
    for dataset in DATASET_ORDER:
        reference = canonical_rows(predictions.loc[(predictions["dataset"] == dataset) & (predictions["layer"] == "I_M_N")])[["record_id", "y_SCLC"]]
        for layer in LAYER_ORDER:
            current = canonical_rows(predictions.loc[(predictions["dataset"] == dataset) & (predictions["layer"] == layer)])[["record_id", "y_SCLC"]]
            if not current.equals(reference):
                raise RuntimeError(f"Shared-bootstrap cohort mismatch: {dataset}/{layer}")
    rows: list[dict[str, Any]] = []
    for layer in LAYER_ORDER:
        for dataset in DATASET_ORDER:
            group = predictions.loc[
                (predictions["layer"] == layer) & (predictions["dataset"] == dataset)
            ].copy()
            group = canonical_rows(group)
            y = group["y_SCLC"].to_numpy(dtype=int)
            probability = group["probability"].to_numpy(dtype=float)
            points = {**_metric_values(y, probability), **threshold_metric_values(y, probability, threshold)}
            seed = bootstrap_seed(dataset)
            indices = shared_bootstrap_indices(y, dataset, group["record_id"].to_numpy(str))
            bootstrap = {name: [] for name in points}
            for index in indices:
                values = {**_metric_values(y[index], probability[index]), **threshold_metric_values(y[index], probability[index], threshold)}
                for name, value in values.items():
                    bootstrap[name].append(value)
            row: dict[str, Any] = {
                "layer": layer,
                "layer_label": LAYER_LABELS[layer],
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "model": WEIGHTED_NAME,
                "n": len(group),
                "positive_n": int(y.sum()),
                "negative_n": int(len(y) - y.sum()),
                "bootstrap_n": BOOTSTRAP_N,
                "bootstrap_seed": seed,
                "bootstrap_protocol": BOOTSTRAP_PROTOCOL,
                "bootstrap_case_sha256": bootstrap_identity(y, group["record_id"].to_numpy(str)),
                "threshold": threshold,
                "threshold_source": "development CV Youden maximum of the primary weighted-voting model",
                "development_youden_j": max_j,
                "ci_method": "stratified percentile bootstrap of per-record predictions",
            }
            for metric, point in points.items():
                low, high = _percentile_interval(bootstrap[metric])
                row[metric] = point
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            if dataset == "A_dev_cv_clean":
                fold_values = []
                for _, fold_group in group.groupby("fold", sort=True):
                    fold_values.append(_metric_values(fold_group["y_SCLC"].to_numpy(dtype=int), fold_group["probability"].to_numpy(dtype=float)))
                for metric in ("auc", "auprc", "brier"):
                    values = np.asarray([item[metric] for item in fold_values], dtype=float)
                    row[f"fold_{metric}_mean"] = float(values.mean())
                    row[f"fold_{metric}_sd"] = float(values.std(ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def _aligned_layers(predictions: pd.DataFrame, lower: str, upper: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = predictions.loc[predictions["dataset"] == "A_dev_cv_clean"].copy()
    keys = ["fold", "record_id", "y_SCLC"]
    low = subset.loc[subset["layer"] == lower, keys + ["probability"]].rename(columns={"probability": "lower_probability"})
    high = subset.loc[subset["layer"] == upper, keys + ["probability"]].rename(columns={"probability": "upper_probability"})
    low, high = canonical_rows(low), canonical_rows(high)
    if low.empty or set(low["record_id"]) != set(high["record_id"]):
        raise RuntimeError(f"paired layer comparison {lower}/{upper} has different or empty case sets")
    merged = low.merge(high, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise RuntimeError(f"paired layer comparison {lower}/{upper} has inconsistent folds or outcomes")
    merged = canonical_rows(merged.drop(columns="_merge"))
    return (
        merged["y_SCLC"].to_numpy(dtype=int),
        merged["lower_probability"].to_numpy(dtype=float),
        merged["upper_probability"].to_numpy(dtype=float),
    )


def _increment_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lower, upper, label in COMPARISONS:
        y, lower_probability, upper_probability = _aligned_layers(predictions, lower, upper)
        seed = bootstrap_seed("A_dev_cv_clean")
        case_rows = canonical_rows(predictions.loc[(predictions["dataset"] == "A_dev_cv_clean") & (predictions["layer"] == lower)])
        indices = shared_bootstrap_indices(y, "A_dev_cv_clean", case_rows["record_id"].to_numpy(str))
        lower_points = _metric_values(y, lower_probability)
        upper_points = _metric_values(y, upper_probability)
        for metric in ("auc", "auprc", "brier"):
            point = upper_points[metric] - lower_points[metric]
            bootstrap_delta: list[float] = []
            for index in indices:
                low_value = _metric_values(y[index], lower_probability[index])[metric]
                high_value = _metric_values(y[index], upper_probability[index])[metric]
                bootstrap_delta.append(high_value - low_value)
            ci_low, ci_high = _percentile_interval(bootstrap_delta)
            rows.append(
                {
                    "dataset": "A_dev_cv_clean",
                    "model": WEIGHTED_NAME,
                    "comparison": label,
                    "lower_layer": lower,
                    "upper_layer": upper,
                    "direction": "upper minus lower",
                    "metric": metric,
                    "lower_value": lower_points[metric],
                    "upper_value": upper_points[metric],
                    "increment": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_n": BOOTSTRAP_N,
                    "bootstrap_seed": seed,
                "bootstrap_protocol": BOOTSTRAP_PROTOCOL,
                    "ci_method": "paired stratified percentile bootstrap",
                }
            )
    return pd.DataFrame(rows)


def _midrank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    result = np.empty(len(values), dtype=float)
    result[order] = ranks
    return result


def _delong_covariance(predictions_sorted: np.ndarray, positive_n: int) -> tuple[np.ndarray, np.ndarray]:
    m = int(positive_n)
    n = predictions_sorted.shape[1] - m
    if m < 2 or n < 2:
        raise ValueError("DeLong covariance requires at least two observations per class")
    tx = np.array([_midrank(row) for row in predictions_sorted[:, :m]])
    ty = np.array([_midrank(row) for row in predictions_sorted[:, m:]])
    tz = np.array([_midrank(row) for row in predictions_sorted])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def _delong_pair(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    order = np.argsort(-np.asarray(y, dtype=int), kind="mergesort")
    aucs, covariance = _delong_covariance(np.vstack([lower, upper])[:, order], int(np.sum(y)))
    difference = float(aucs[1] - aucs[0])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    if not np.isfinite(variance) or variance <= 0:
        raise RuntimeError("non-positive paired DeLong variance")
    standard_error = math.sqrt(variance)
    z_value = difference / standard_error
    return {
        "lower_auc": float(aucs[0]),
        "upper_auc": float(aucs[1]),
        "delta_auc": difference,
        "standard_error": standard_error,
        "ci_low": difference - norm.ppf(0.975) * standard_error,
        "ci_high": difference + norm.ppf(0.975) * standard_error,
        "z": z_value,
        "p_raw": float(2.0 * norm.sf(abs(z_value))),
    }


def _holm_adjust(values: Sequence[float]) -> list[float]:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(len(values) - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def _delong_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lower, upper, label in COMPARISONS:
        y, lower_probability, upper_probability = _aligned_layers(predictions, lower, upper)
        result = _delong_pair(y, lower_probability, upper_probability)
        rows.append(
            {
                "comparison_family": "three_prespecified_weighted_voting_layer_comparisons",
                "family_n": 3,
                "dataset": "A_dev_cv_clean",
                "model": WEIGHTED_NAME,
                "comparison": label,
                "lower_layer": lower,
                "upper_layer": upper,
                "direction": "upper minus lower",
                **result,
            }
        )
    adjusted = _holm_adjust([row["p_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
    return pd.DataFrame(rows)


def _calibration_plot_data(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for layer in LAYER_ORDER:
        for dataset in DATASET_ORDER:
            group = predictions.loc[
                (predictions["layer"] == layer) & (predictions["dataset"] == dataset)
            ].copy()
            y = group["y_SCLC"].to_numpy(dtype=float)
            probability = group["probability"].to_numpy(dtype=float)
            lower, upper = np.quantile(probability, [0.025, 0.975])
            smooth = lowess(
                endog=y,
                exog=probability,
                frac=LOWESS_FRAC,
                it=LOWESS_IT,
                delta=0.0,
                is_sorted=False,
                return_sorted=True,
            )
            smooth = smooth[
                np.isfinite(smooth).all(axis=1)
                & (smooth[:, 0] >= lower)
                & (smooth[:, 0] <= upper)
            ]
            for x_value, y_value in smooth:
                rows.append(
                    {
                        "layer": layer,
                        "dataset": dataset,
                        "series_type": "LOWESS",
                        "x": float(x_value),
                        "y": float(y_value),
                        "n": len(group),
                        "bin": np.nan,
                        "prediction_percentile_low": float(lower),
                        "prediction_percentile_high": float(upper),
                        "lowess_frac": LOWESS_FRAC,
                        "lowess_it": LOWESS_IT,
                        "kernel": "tricube",
                        "local_polynomial_degree": 1,
                    }
                )
            quantile_bins = pd.qcut(pd.Series(probability), q=10, labels=False, duplicates="drop")
            bin_frame = pd.DataFrame({"probability": probability, "outcome": y, "bin": quantile_bins})
            for bin_number, bin_group in bin_frame.groupby("bin", observed=True, sort=True):
                rows.append(
                    {
                        "layer": layer,
                        "dataset": dataset,
                        "series_type": "equal_frequency_bin",
                        "x": float(bin_group["probability"].mean()),
                        "y": float(bin_group["outcome"].mean()),
                        "n": len(bin_group),
                        "bin": int(bin_number) + 1,
                        "prediction_percentile_low": float(lower),
                        "prediction_percentile_high": float(upper),
                        "lowess_frac": LOWESS_FRAC,
                        "lowess_it": LOWESS_IT,
                        "kernel": "tricube",
                        "local_polynomial_degree": 1,
                    }
                )
    return pd.DataFrame(rows)


def _plot_f6(calibration: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35), sharex=True, sharey=True)
    letters = "ABC"
    for ax, layer, letter in zip(axes, LAYER_ORDER, letters):
        ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.0, linestyle=":", zorder=0)
        for dataset in DATASET_ORDER:
            subset = calibration.loc[(calibration["layer"] == layer) & (calibration["dataset"] == dataset)]
            curve = subset.loc[subset["series_type"] == "LOWESS"].sort_values("x", kind="mergesort")
            bins = subset.loc[subset["series_type"] == "equal_frequency_bin"].sort_values("bin", kind="mergesort")
            color = DATASET_COLORS[dataset]
            ax.plot(curve["x"], curve["y"], color=color, linewidth=1.8, alpha=0.95)
            ax.scatter(
                bins["x"],
                bins["y"],
                s=22,
                facecolors="white",
                edgecolors=color,
                linewidths=0.9,
                zorder=3,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#E6E6E6", linewidth=0.55, alpha=0.8)
        ax.set_xlabel("Predicted probability")
        ax.text(
            0.035,
            0.955,
            LAYER_LABELS[layer],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            0.97,
            0.035,
            "LOWESS: local linear, tricube\nfrac = 0.75; it = 0; central 95%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.7,
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        )
        panel_label(ax, letter, x=-0.12, y=1.02)
    axes[0].set_ylabel("Observed probability")
    handles = [
        Line2D([0], [0], color=DATASET_COLORS[dataset], linewidth=1.8, marker="o", markerfacecolor="white", markersize=4.5, label=DATASET_LABELS[dataset])
        for dataset in DATASET_ORDER
    ]
    handles.append(Line2D([0], [0], color="#777777", linewidth=1.0, linestyle=":", label="Ideal"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=False, handlelength=2.8, columnspacing=1.5)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.95, bottom=0.20, wspace=0.16)
    paths = save_fig(fig, "FigS3_layer_calibration_weighted_voting")
    plt.close(fig)
    return paths


def _plot_f7(metrics: pd.DataFrame) -> tuple[Path, Path]:
    brier_low = float(metrics["brier_ci_low"].min())
    brier_high = float(metrics["brier_ci_high"].max())
    margin = max(0.008, 0.10 * (brier_high - brier_low))
    x_limits = (max(0.0, brier_low - margin), brier_high + margin)
    y_positions = np.arange(len(DATASET_ORDER), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), sharex=True, sharey=True)
    for ax, layer, letter in zip(axes, LAYER_ORDER, "ABC"):
        layer_data = metrics.loc[metrics["layer"] == layer].set_index("dataset").loc[list(DATASET_ORDER)]
        for position, dataset in zip(y_positions, DATASET_ORDER):
            row = layer_data.loc[dataset]
            ax.errorbar(
                float(row["brier"]),
                position,
                xerr=np.array([[float(row["brier"] - row["brier_ci_low"])], [float(row["brier_ci_high"] - row["brier"])]]),
                fmt="o",
                color=DATASET_COLORS[dataset],
                markerfacecolor="white",
                markeredgewidth=1.2,
                markersize=6.0,
                capsize=3.0,
                elinewidth=1.2,
                zorder=3,
            )
        ax.set_xlim(*x_limits)
        ax.set_ylim(-0.65, len(DATASET_ORDER) - 0.15)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        ax.set_xlabel("Brier score")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER])
        ax.text(0.035, 0.955, LAYER_LABELS[layer], transform=ax.transAxes, va="top", ha="left", fontsize=10, fontweight="bold")
        ax.annotate(
            "Lower is better",
            xy=(0.06, 0.06),
            xytext=(0.40, 0.06),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
            ha="center",
            va="center",
            fontsize=7.5,
            color="#444444",
        )
        panel_label(ax, letter, x=-0.12, y=1.02)
    fig.subplots_adjust(left=0.185, right=0.995, top=0.95, bottom=0.16, wspace=0.12)
    paths = save_fig(fig, "FigS2_layer_brier_weighted_voting")
    plt.close(fig)
    return paths


def _format_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _plot_f8(metrics: pd.DataFrame, increments: pd.DataFrame, delong: pd.DataFrame) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(13.4, 4.75))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.00, 1.55], wspace=0.28)
    ax_a = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.03)
    ax_b = fig.add_subplot(right[0, 0])
    ax_table = fig.add_subplot(right[0, 1])

    a_data = metrics.loc[metrics["dataset"] == "A_dev_cv_clean"].set_index("layer").loc[list(LAYER_ORDER)]
    x = np.arange(len(LAYER_ORDER), dtype=float)
    metric_styles = {
        "auc": ("ROC AUC", "#2F5597", "o", -0.035),
        "auprc": ("AUPRC", "#D55E00", "s", 0.035),
    }
    all_lows: list[float] = []
    all_highs: list[float] = []
    for metric, (label, color, marker, offset) in metric_styles.items():
        points = a_data[metric].to_numpy(dtype=float)
        lows = a_data[f"{metric}_ci_low"].to_numpy(dtype=float)
        highs = a_data[f"{metric}_ci_high"].to_numpy(dtype=float)
        all_lows.extend(lows.tolist())
        all_highs.extend(highs.tolist())
        ax_a.errorbar(
            x + offset,
            points,
            yerr=np.vstack([points - lows, highs - points]),
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=6.0,
            linewidth=1.5,
            capsize=3.0,
            label=label,
        )
    y_min = max(0.0, min(all_lows) - 0.15)
    y_max = min(1.0, max(all_highs) + 0.055)
    ax_a.set_ylim(y_min, y_max)
    ax_a.set_xlim(-0.28, 2.28)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([LAYER_LABELS[layer] for layer in LAYER_ORDER])
    ax_a.set_ylabel("Discrimination")
    ax_a.set_xlabel("Cumulative layer")
    ax_a.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax_a.legend(loc="upper left", frameon=False)

    adjacent = increments.loc[
        increments["metric"].isin(["auc", "auprc"])
        & increments["comparison"].isin(["I → I + M", "I + M → I + M + N"])
    ]
    annotation_y = y_min + 0.018 * (y_max - y_min)
    for midpoint, comparison in zip((0.50, 1.50), ("I → I + M", "I + M → I + M + N")):
        sub = adjacent.loc[adjacent["comparison"] == comparison].set_index("metric")
        text = f"ΔAUC {float(sub.loc['auc', 'increment']):+.3f}\nΔAUPRC {float(sub.loc['auprc', 'increment']):+.3f}"
        ax_a.text(midpoint, annotation_y, text, ha="center", va="bottom", fontsize=7.2, color="#333333")
    total = increments.loc[(increments["comparison"] == "I → I + M + N") & increments["metric"].isin(["auc", "auprc"])].set_index("metric")
    ax_a.text(
        0.98,
        0.96,
        f"Total ΔAUC {float(total.loc['auc', 'increment']):+.3f}\nTotal ΔAUPRC {float(total.loc['auprc', 'increment']):+.3f}",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "linewidth": 0.6, "pad": 2.0},
    )
    panel_label(ax_a, "A", x=-0.15, y=1.02)

    d = delong.set_index("comparison").loc[[item[2] for item in COMPARISONS]].reset_index()
    y = np.arange(len(d), dtype=float)
    point = d["delta_auc"].to_numpy(dtype=float)
    low = d["ci_low"].to_numpy(dtype=float)
    high = d["ci_high"].to_numpy(dtype=float)
    ax_b.axvline(0.0, color="#777777", linestyle=":", linewidth=1.0)
    ax_b.errorbar(
        point,
        y,
        xerr=np.vstack([point - low, high - point]),
        fmt="o",
        color="#2F5597",
        markerfacecolor="white",
        markeredgewidth=1.2,
        markersize=6.2,
        capsize=3.2,
        elinewidth=1.25,
    )
    range_low = min(0.0, float(low.min()))
    range_high = max(0.0, float(high.max()))
    span = max(0.02, range_high - range_low)
    ax_b.set_xlim(range_low - 0.13 * span, range_high + 0.13 * span)
    ax_b.set_ylim(-0.65, len(d) - 0.35)
    ax_b.invert_yaxis()
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(d["comparison"].tolist())
    ax_b.set_xlabel("ΔAUC (upper − lower)")
    ax_b.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    panel_label(ax_b, "B", x=-0.23, y=1.02)

    ax_table.set_axis_off()
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(-0.65, len(d) - 0.35)
    ax_table.invert_yaxis()
    headers = ((0.02, "ΔAUC (95% CI)"), (0.67, "P"), (0.84, "Holm P"))
    for x_header, header in headers:
        ax_table.text(x_header, -0.47, header, fontsize=8.0, fontweight="bold", ha="left", va="center")
    for row_number, row in d.iterrows():
        ax_table.text(
            0.02,
            row_number,
            f"{row['delta_auc']:+.3f} ({row['ci_low']:+.3f} to {row['ci_high']:+.3f})",
            fontsize=7.6,
            ha="left",
            va="center",
        )
        ax_table.text(0.67, row_number, _format_p(float(row["p_raw"])), fontsize=7.6, ha="left", va="center")
        ax_table.text(0.84, row_number, _format_p(float(row["p_holm"])), fontsize=7.6, ha="left", va="center")
    ax_table.text(
        0.02,
        1.04,
        "Paired DeLong tests; one pre-specified three-comparison Holm family",
        transform=ax_table.transAxes,
        fontsize=7.2,
        ha="left",
        va="bottom",
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.17)
    paths = save_fig(fig, "unnumbered_layer_increment_delong")
    plt.close(fig)
    return paths


def _manifest_rows(outputs: Mapping[str, tuple[Path, Path]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specifications = {
        "F6": (
            "FigS3_layer_calibration_weighted_voting",
            (("A", "I"), ("B", "I_M"), ("C", "I_M_N")),
            "LOWESS curves and ten equal-frequency calibration points",
            "data/weighted_voting_layer_calibration_plot_data.csv",
        ),
        "F7": (
            "FigS2_layer_brier_weighted_voting",
            (("A", "I"), ("B", "I_M"), ("C", "I_M_N")),
            "Brier score with stratified-bootstrap 95% CI",
            "tables/weighted_voting_layer_metrics_ci.csv",
        ),
        "F8": (
            "unnumbered_layer_increment_delong",
            (("A", "I|I_M|I_M_N"), ("B", "I|I_M|I_M_N")),
            "Layer discrimination, paired increments, and prespecified DeLong/Holm comparisons",
            "tables/weighted_voting_layer_metrics_ci.csv|tables/weighted_voting_layer_increment_ci.csv|tables/weighted_voting_layer_delong_holm.csv",
        ),
    }
    for figure, (basename, panels, description, source) in specifications.items():
        pdf, png = outputs[figure]
        for panel, layer in panels:
            rows.append(
                {
                    "figure": figure,
                    "panel": panel,
                    "description": description,
                    "model": WEIGHTED_NAME,
                    "layer": layer,
                    "datasets": "|".join(DATASET_ORDER) if figure in {"F6", "F7"} else "A_dev_cv_clean",
                    "data_source": source,
                    "code": "code/make_F6_F8_weighted_layers.py",
                    "output_png": str(png.relative_to(data_root())).replace("\\", "/"),
                    "output_pdf": str(pdf.relative_to(data_root())).replace("\\", "/"),
                    "png_sha256": _sha256_file(png),
                    "pdf_sha256": _sha256_file(pdf),
                    "deterministic_recalculation": True,
                }
            )
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    started = time.perf_counter()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    train, cohorts, split_info = _load_cohorts()
    cv_predictions, cv_provenance, weight_maps, member_aucs = _fit_weighted_cv(train)
    full_predictions, full_provenance = _fit_weighted_full(train, cohorts, weight_maps)
    predictions = pd.concat([cv_predictions, full_predictions], ignore_index=True)
    predictions["dataset"] = pd.Categorical(predictions["dataset"], categories=DATASET_ORDER, ordered=True)
    predictions["layer"] = pd.Categorical(predictions["layer"], categories=LAYER_ORDER, ordered=True)
    predictions = predictions.sort_values(["layer", "dataset", "fold", "record_id"], kind="mergesort").reset_index(drop=True)
    predictions["dataset"] = predictions["dataset"].astype(str)
    predictions["layer"] = predictions["layer"].astype(str)

    audit = _audit_i_m_n(predictions)
    metrics = _metrics_with_ci(predictions, _development_threshold())
    increments = _increment_table(predictions)
    delong = _delong_table(predictions)
    calibration = _calibration_plot_data(predictions)

    prediction_path = DATA_DIR / "weighted_voting_layer_predictions.csv"
    calibration_path = DATA_DIR / "weighted_voting_layer_calibration_plot_data.csv"
    weights_path = TABLE_DIR / "weighted_voting_fixed_weights.csv"
    metrics_path = TABLE_DIR / "weighted_voting_layer_metrics_ci.csv"
    increments_path = TABLE_DIR / "weighted_voting_layer_increment_ci.csv"
    delong_path = TABLE_DIR / "weighted_voting_layer_delong_holm.csv"
    audit_path = TABLE_DIR / "I_M_N_weighted_prediction_reproducibility_audit.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    calibration.to_csv(calibration_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "layer": layer,
                "model": model,
                "source_cv_auc": member_aucs[layer][model],
                "normalized_weight": weight,
                "weight_source": "same-layer development 10-fold CV ROC-AUC",
            }
            for layer, weights in weight_maps.items()
            for model, weight in weights.items()
        ]
    ).to_csv(weights_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    increments.to_csv(increments_path, index=False, encoding="utf-8-sig")
    delong.to_csv(delong_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    outputs = {
        "F6": _plot_f6(calibration),
        "F7": _plot_f7(metrics),
        "F8": _plot_f8(metrics, increments, delong),
    }
    figure_manifest = _manifest_rows(outputs)
    figure_manifest_path = TABLE_DIR / "F6_F8_manifest.csv"
    figure_manifest.to_csv(figure_manifest_path, index=False, encoding="utf-8-sig")

    produced = [
        prediction_path,
        calibration_path,
        weights_path,
        metrics_path,
        increments_path,
        delong_path,
        audit_path,
        figure_manifest_path,
        *[path for pair in outputs.values() for path in pair],
    ]
    manifest = {
        "split_design": split_metadata(),
        "run_name": "F6_F8_weighted_voting_primary_layer_revision_v1",
        "primary_model": WEIGHTED_NAME,
        "source_data_models_predictions_modified": False,
        "retraining_scope": "Authorized deterministic refit of the 13 native-probability members for each frozen cumulative layer only.",
        "input_workbook": str(INPUT_WORKBOOK),
        "input_workbook_sha256": _sha256_file(INPUT_WORKBOOK),
        "weight_source": "same-layer development 10-fold CV ROC-AUC",
        "native_members": list(NATIVE_MEMBERS),
        "weights": weight_maps,
        "layers": LAYER_SPECS,
        "dataset_order": DATASET_ORDER,
        "split": split_info,
        "A_development_cv": {
            "strategy": "StratifiedKFold",
            "n_splits": 10,
            "shuffle": True,
            "cv_seed": CV_SEED,
            "provenance": cv_provenance,
        },
        "full_fit": {
            "training_scope": "A development only",
            "seed_contract": "derive_seed(MASTER_SEED, holdout_fit, model, model_name)",
            "selector_scope_id": "common_scheme",
            "provenance": full_provenance,
        },
        "I_M_N_existing_prediction_audit": audit.to_dict("records"),
        "bootstrap": {
            "n": BOOTSTRAP_N,
            "method": "stratified percentile bootstrap; paired within layer comparisons",
            "seed_namespace": BOOTSTRAP_NAMESPACE,
        },
        "calibration": {
            "smoother": "LOWESS local-linear regression on individual binary outcomes",
            "kernel": "tricube",
            "frac": LOWESS_FRAC,
            "it": LOWESS_IT,
            "display_range": "dataset-specific probability percentiles 2.5 to 97.5",
            "raw_points": "10 equal-frequency bins",
            "ideal_reference": True,
        },
        "delong": {
            "comparisons": [item[2] for item in COMPARISONS],
            "direction": "upper minus lower",
            "ci": "paired DeLong normal-theory 95% CI",
            "multiplicity": "Holm adjustment across exactly the three pre-specified comparisons",
        },
        "graphics": {
            "font": "Times New Roman",
            "language": "English",
            "png_dpi": 600,
            "pdf_fonttype": 42,
            "vector_pdf": True,
            "figure_titles": False,
        },
        "outputs": [
            {
                "path": str(path.relative_to(data_root())).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in produced
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    manifest_path = DATA_DIR / "run_manifest_F6_F8.json"
    manifest_path.write_text(json.dumps(_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": "complete", "runtime_seconds": result["runtime_seconds"], "audit": result["I_M_N_existing_prediction_audit"]}, ensure_ascii=False, indent=2))
