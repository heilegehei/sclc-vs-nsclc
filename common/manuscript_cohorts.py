from __future__ import annotations
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_COHORTS = {"B": (329, 67), "C": (313, 70)}
MASTER_SEED = 20260902

def validate_cohort(frame, name):
    required = {"record_id", "y_SCLC"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name}: missing cohort fields {sorted(required - set(frame.columns))}")
    ids = frame["record_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.astype(str).duplicated().any():
        raise ValueError(f"{name}: patient record IDs must be present and unique")
    if "source_record_id" in frame and frame["source_record_id"].dropna().astype(str).duplicated().any():
        raise ValueError(f"{name}: repeated source_record_id cannot count as additional patients")
    y = pd.to_numeric(frame["y_SCLC"], errors="raise")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError(f"{name}: y_SCLC must contain only 0 or 1")
    actual = (len(frame), int(y.sum()))
    expected = EXPECTED_COHORTS.get(name)
    if y.nunique() != 2:
        raise ValueError(f"{name}: both outcome classes are required")
    if expected is not None and actual != expected:
        raise ValueError(f"Manuscript cohort mismatch for {name}: observed n/SCLC={actual}, required={expected}. "
                         "Supply the verified patient cohort and regenerate its frozen split/model outputs; "
                         "do not pad, duplicate, relabel, or change stored result counts.")
    return frame

def validate_base_cohorts(frames):
    for center in ("A", "B", "C"):
        validate_cohort(frames[center], center)
    ids = pd.concat([frames[c]["record_id"].astype(str) for c in ("A", "B", "C")])
    if ids.duplicated().any():
        raise ValueError("Patient record IDs overlap across centers")
    return frames

_LAST_SPLIT_METADATA = {}

SPLIT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "manuscript_split.json"

def split_design(master_seed=None):
    if not SPLIT_CONFIG_PATH.is_file():
        raise FileNotFoundError("Missing manuscript_split.json. Initialize and save the study split once before running any consumer; no automatic random draw is allowed.")
    raw = SPLIT_CONFIG_PATH.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    fraction = float(config["train_fraction"])
    test_fraction = float(config["test_fraction"])
    seed = int(config["split_seed"])
    if master_seed is not None and int(master_seed) != seed:
        raise ValueError("Requested seed differs from the study's frozen manuscript_split.json")
    if not .70 <= fraction <= .80 or not np.isclose(fraction + test_fraction, 1.0, rtol=0, atol=1e-14):
        raise ValueError("Invalid frozen train/test proportions")
    if config.get("strategy") != "frozen_once_random_fraction":
        raise ValueError("Unexpected study split strategy")
    design = {"master_seed": seed, "split_seed": seed, "train_fraction": fraction,
            "test_fraction": test_fraction, "strategy": config["strategy"],
            "split_config_sha256": hashlib.sha256(raw).hexdigest()}
    for key in ("manuscript_center_a_n", "manuscript_development_n", "manuscript_evaluation_n",
                "manuscript_development_sclc_n", "manuscript_evaluation_sclc_n"):
        if key in config:
            design[key] = int(config[key])
    if "manuscript_development_n" in design and "manuscript_evaluation_n" in design:
        if design["manuscript_development_n"] + design["manuscript_evaluation_n"] != design.get("manuscript_center_a_n", design["manuscript_development_n"] + design["manuscript_evaluation_n"]):
            raise ValueError("Manuscript development and evaluation counts do not sum to center A")
    return design

def split_metadata():
    return dict(_LAST_SPLIT_METADATA or split_design())

def _ids_hash(frame):
    return hashlib.sha256("\n".join(sorted(frame["record_id"].astype(str))).encode("utf-8")).hexdigest()

def split_center_a_indices(frame):
    global _LAST_SPLIT_METADATA
    validate_cohort(frame, "A")
    design = split_design()
    order = np.argsort(frame["record_id"].astype(str).to_numpy(), kind="stable")
    if design.get("manuscript_center_a_n") == len(frame):
        train_n = int(design["manuscript_development_n"])
    else:
        train_n = int(np.clip(round(design["train_fraction"] * len(frame)), np.ceil(0.70 * len(frame)), np.floor(0.80 * len(frame))))
    train, holdout = train_test_split(order, train_size=train_n,
        stratify=frame.iloc[order]["y_SCLC"].to_numpy(dtype=int), random_state=design["master_seed"])
    train, holdout = np.sort(train), np.sort(holdout)
    development, evaluation = frame.iloc[train], frame.iloc[holdout]
    validate_cohort(development, "A_development")
    validate_cohort(evaluation, "A_holdout_clean")
    if set(development["record_id"].astype(str)) & set(evaluation["record_id"].astype(str)):
        raise ValueError("A development and evaluation records overlap")
    _LAST_SPLIT_METADATA = {**design, "source_n": len(frame), "source_positive_n": int(frame.y_SCLC.sum()),
        "train_n": len(train), "test_n": len(holdout),
        "train_positive_n": int(development.y_SCLC.sum()), "test_positive_n": int(evaluation.y_SCLC.sum()),
        "actual_train_fraction": len(train)/len(frame), "actual_test_fraction": len(holdout)/len(frame),
        "train_record_id_sha256": _ids_hash(development), "test_record_id_sha256": _ids_hash(evaluation)}
    return train, holdout


def require_split_design(metadata):
    if not _LAST_SPLIT_METADATA or "train_record_id_sha256" not in _LAST_SPLIT_METADATA:
        raise RuntimeError("Reconstruct the current A split before validating a saved split_design")
    saved = metadata.get("split_design") if isinstance(metadata, dict) else None
    required = ("split_config_sha256", "master_seed", "train_fraction", "test_fraction", "actual_train_fraction", "actual_test_fraction",
                "train_n", "test_n", "train_positive_n", "test_positive_n",
                "train_record_id_sha256", "test_record_id_sha256")
    if not isinstance(saved, dict) or any(key not in saved for key in required):
        raise ValueError("Saved artifact lacks complete split_design provenance; regenerate the upstream training/weights/model for this split. Do not relabel existing artifacts.")
    for key in required:
        expected, observed = _LAST_SPLIT_METADATA[key], saved[key]
        same = np.isclose(observed, expected, rtol=0, atol=1e-14) if isinstance(expected, float) and isinstance(observed, (int, float)) else observed == expected
        if not same:
            raise ValueError(f"Saved artifact split_design mismatch for {key}; regenerate upstream training/weights/model using the same frozen manuscript_split.json and patient IDs.")
    return metadata
