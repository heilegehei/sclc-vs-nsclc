import hashlib
import warnings
import numpy as np
import pandas as pd

BOOTSTRAP_N = 2000
BOOTSTRAP_PROTOCOL = "manuscript-2000-stratified-v1"
MANUSCRIPT_DECISION_THRESHOLD = 0.2076
_DATASETS = {"A_dev_cv_clean": 0, "A_holdout_clean": 1, "B_external": 2, "C_external": 3}

def bootstrap_seed(dataset):
    if dataset not in _DATASETS:
        raise ValueError(f"Unknown bootstrap dataset: {dataset}")
    return 20260911 + 1009 * _DATASETS[dataset]

def canonical_rows(frame):
    if frame["record_id"].isna().any():
        raise ValueError("Missing record_id cannot identify bootstrap cases")
    frame = frame.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    if frame["record_id"].duplicated().any():
        raise ValueError("Bootstrap requires unique record_id values")
    return frame.sort_values("record_id", kind="mergesort").reset_index(drop=True)

def shared_bootstrap_indices(y, dataset, record_ids):
    y = np.asarray(y)
    ids = np.asarray(record_ids)
    if y.ndim != 1 or ids.ndim != 1 or len(y) != len(ids):
        raise ValueError("Outcome and record-ID arrays must be aligned vectors")
    if pd.isna(ids).any() or len(set(ids.astype(str))) != len(ids):
        raise ValueError("Bootstrap requires nonmissing unique record IDs")
    if not np.isin(y, [0, 1]).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("Bootstrap requires known binary outcomes and both classes")
    order = np.argsort(ids.astype(str), kind="stable")
    negative = order[y[order] == 0]
    positive = order[y[order] == 1]
    rng = np.random.default_rng(bootstrap_seed(dataset))
    return [np.concatenate((rng.choice(negative, len(negative), replace=True),
                            rng.choice(positive, len(positive), replace=True)))
            for _ in range(BOOTSTRAP_N)]

def bootstrap_identity(y, record_ids):
    frame = canonical_rows(pd.DataFrame({"record_id": record_ids, "y_SCLC": y}))
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()

from sklearn.metrics import confusion_matrix, roc_curve

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

def primary_youden_threshold(y, probability):
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=float)
    if set(y.tolist()) != {0, 1} or not np.isfinite(probability).all():
        raise ValueError("Primary OOF predictions must have two classes and finite probabilities")
    fpr, tpr, thresholds = roc_curve(y, probability)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    j = tpr[valid] - fpr[valid]
    maximum = float(np.max(j))
    return float(thresholds[valid][np.flatnonzero(np.isclose(j, maximum, rtol=0, atol=1e-15))[0]]), maximum

def verify_manuscript_threshold(y, probability, *, strict=True):
    derived, max_j = primary_youden_threshold(y, probability)
    rounded = round(derived, 4)
    matches = bool(np.isclose(rounded, MANUSCRIPT_DECISION_THRESHOLD, rtol=0, atol=1e-12))
    if not matches:
        message = (f"Development CV Youden threshold {derived:.6f} (rounded {rounded:.4f}) differs from the "
                   f"manuscript threshold {MANUSCRIPT_DECISION_THRESHOLD:.4f}")
        if strict:
            raise ValueError(message + "; rerun with --allow-threshold-mismatch for non-manuscript data")
        warnings.warn(message)
    return {"threshold": rounded, "derived_threshold": derived, "youden_j": max_j,
            "matches_manuscript": matches,
            "threshold_source": "maximum Youden index on pooled development 10-fold CV out-of-fold "
                                "weighted-voting predictions, rounded to four decimals"}
