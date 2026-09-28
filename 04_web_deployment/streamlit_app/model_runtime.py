from __future__ import annotations


import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
RUNTIME_DIR = APP_DIR / "runtime"
MODEL_PATH = ASSET_DIR / "weighted_voting_model.joblib"
MANIFEST_PATH = ASSET_DIR / "model_manifest.json"
FROZEN_DECISION_THRESHOLD = 0.2076


if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

import model_factory
import stage4_pipeline
from stage4_pipeline import DIRECT_FEATURES, compute_composites


INPUT_FIELDS: tuple[dict[str, str], ...] = (
    {"feature": "WBC", "group": "I", "label": "White blood cell count", "unit": "10^9/L", "kind": "selected"},
    {"feature": "PLT", "group": "I", "label": "Platelet count", "unit": "10^9/L", "kind": "selected"},
    {"feature": "LDH", "group": "I", "label": "Lactate dehydrogenase", "unit": "U/L", "kind": "selected"},
    {"feature": "PCT", "group": "I", "label": "Plateletcrit", "unit": "%", "kind": "selected"},
    {"feature": "NEUT#", "group": "I", "label": "Absolute neutrophil count", "unit": "10^9/L", "kind": "composite"},
    {"feature": "MONO#", "group": "I", "label": "Absolute monocyte count", "unit": "10^9/L", "kind": "composite"},
    {"feature": "EO#", "group": "M", "label": "Absolute eosinophil count", "unit": "10^9/L", "kind": "selected"},
    {"feature": "EO%", "group": "M", "label": "Eosinophil percentage", "unit": "%", "kind": "selected"},
    {"feature": "LYMPH#", "group": "M", "label": "Absolute lymphocyte count", "unit": "10^9/L", "kind": "composite"},
    {"feature": "ALB", "group": "N", "label": "Albumin", "unit": "g/L", "kind": "selected"},
    {"feature": "TP", "group": "N", "label": "Total protein", "unit": "g/L", "kind": "selected"},
    {"feature": "GLB", "group": "N", "label": "Globulin", "unit": "g/L", "kind": "selected"},
    {"feature": "TC", "group": "N", "label": "Total cholesterol", "unit": "mmol/L", "kind": "selected"},
    {"feature": "LDL-C", "group": "N", "label": "Low-density lipoprotein cholesterol", "unit": "mmol/L", "kind": "selected"},
    {"feature": "MCH", "group": "N", "label": "Mean corpuscular haemoglobin", "unit": "pg", "kind": "selected"},
    {"feature": "MCHC", "group": "N", "label": "Mean corpuscular haemoglobin concentration", "unit": "g/L", "kind": "selected"},
    {"feature": "GLU", "group": "N", "label": "Glucose", "unit": "mmol/L", "kind": "composite"},
    {"feature": "HGB", "group": "N", "label": "Haemoglobin", "unit": "g/L", "kind": "composite"},
)

GROUP_LABELS = {
    "I": "Inflammation",
    "M": "Immune",
    "N": "Nutrition / metabolism",
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "gam": "GAM",
    "knn": "KNN",
    "gaussian_nb": "Gaussian NB",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "gbdt": "GBDT",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "adaboost": "AdaBoost",
    "rotation_forest": "Rotation forest",
    "mlp": "MLP",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Model manifest not found: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    manifest = _load_manifest()
    expected_hash = str(manifest.get("sha256", ""))
    observed_hash = sha256_file(MODEL_PATH)
    if expected_hash and observed_hash != expected_hash:
        raise RuntimeError("The local model file does not match the frozen manifest hash.")

    model: dict[str, Any] = joblib.load(MODEL_PATH)
    required = {"preprocessor", "estimators", "member_order", "weights", "selected_direct", "fixed_composites"}
    missing = sorted(required.difference(model))
    if missing:
        raise RuntimeError(f"Frozen model lacks required members: {missing}")
    if tuple(model["member_order"]) != tuple(model["weights"].keys()):
        raise RuntimeError("The frozen model voting order and weights are inconsistent.")
    if not np.isclose(sum(float(x) for x in model["weights"].values()), 1.0):
        raise RuntimeError("Frozen voting weights do not sum to one.")
    return model


def input_defaults() -> dict[str, float]:
    model = load_model()
    medians: Mapping[str, float] = model["preprocessor"].medians
    return {item["feature"]: float(medians[item["feature"]]) for item in INPUT_FIELDS}


def input_template() -> pd.DataFrame:
    return pd.DataFrame(columns=["record_id"] + [item["feature"] for item in INPUT_FIELDS])


def validate_required_values(values: Mapping[str, float]) -> list[str]:
    messages: list[str] = []
    for feature in ("LYMPH#", "MONO#", "ALB", "PLT"):
        value = values.get(feature)
        if value is None or not np.isfinite(float(value)) or float(value) <= 0:
            messages.append(f"{feature} must be greater than zero to calculate the composite indices.")
    return messages


def _direct_frame(frame: pd.DataFrame) -> pd.DataFrame:
    direct = pd.DataFrame(index=frame.index)
    for feature in DIRECT_FEATURES:
        direct[feature] = pd.to_numeric(frame[feature], errors="coerce") if feature in frame.columns else np.nan
    return direct


def score_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    model = load_model()
    direct = _direct_frame(frame)


    imputed_direct = direct.fillna(pd.Series(model["preprocessor"].medians))
    input_composites = compute_composites(imputed_direct)
    transformed, audit = model["preprocessor"].transform(direct)

    member_order = list(model["member_order"])
    member_probability = pd.DataFrame(index=frame.index)
    for name in member_order:
        estimator = model["estimators"][name]
        member_probability[name] = estimator.predict_proba(transformed)[:, 1]

    weight_series = pd.Series(model["weights"], dtype=float).reindex(member_order)
    weighted_probability = member_probability.mul(weight_series, axis=1).sum(axis=1)

    result = pd.DataFrame(index=frame.index)
    if "record_id" in frame.columns:
        result["record_id"] = frame["record_id"].astype(str)
    result["weighted_voting_probability"] = weighted_probability
    result["frozen_decision_threshold"] = FROZEN_DECISION_THRESHOLD
    result["model_label"] = np.where(weighted_probability >= FROZEN_DECISION_THRESHOLD, "Predicted SCLC", "Predicted NSCLC")
    result = pd.concat([result, input_composites.add_prefix("calculated_")], axis=1)
    result = pd.concat([result, member_probability.add_prefix("member_")], axis=1)
    return result.reset_index(drop=True), transformed.reset_index(drop=True), audit


def score_one(values: Mapping[str, float]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, int]]:
    frame = pd.DataFrame([{key: values.get(key, np.nan) for key in [item["feature"] for item in INPUT_FIELDS]}])
    result, transformed, audit = score_frame(frame)
    row = result.iloc[0].to_dict()
    model = load_model()
    contribution = pd.DataFrame(
        {
            "Model": [DISPLAY_NAMES[name] for name in model["member_order"]],
            "Weight": [float(model["weights"][name]) for name in model["member_order"]],
            "Probability": [float(row[f"member_{name}"]) for name in model["member_order"]],
        }
    )
    contribution["Weighted contribution"] = contribution["Weight"] * contribution["Probability"]
    contribution = contribution.sort_values("Weighted contribution", ascending=False, kind="mergesort").reset_index(drop=True)
    return row, contribution, audit


def model_metadata() -> dict[str, Any]:
    model = load_model()
    manifest = _load_manifest()
    return {
        "artifact_sha256": str(manifest.get("sha256", "")),
        "model_name": str(model.get("model_name", "weighted_voting")),
        "training_scope": str(model.get("training_scope", "A_development")),
        "member_n": len(model["member_order"]),
        "selected_direct": list(model["selected_direct"]),
        "fixed_composites": list(model["fixed_composites"]),
        "threshold": FROZEN_DECISION_THRESHOLD,
    }
