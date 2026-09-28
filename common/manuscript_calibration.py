import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from common.manuscript_bootstrap import (BOOTSTRAP_N, BOOTSTRAP_PROTOCOL, canonical_rows,
                                  shared_bootstrap_indices, bootstrap_identity)


def calibration_stats(y, probability):
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=float)
    if y.shape != probability.shape or y.ndim != 1 or set(y.tolist()) != {0, 1}:
        raise ValueError("Calibration requires aligned binary outcomes and both classes")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Calibration requires finite probabilities in [0, 1]")
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    if np.ptp(z) <= np.finfo(float).eps:
        raise ValueError("Calibration slope is not identifiable for constant predictions")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        fit = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(z[:, None], y)
    ece = 0.0
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (probability >= lo) & ((probability < hi) if hi < 1 else (probability <= hi))
        if selected.any():
            ece += selected.mean() * abs(y[selected].mean() - probability[selected].mean())
    return {"calibration_intercept": float(fit.intercept_[0]),
            "calibration_slope": float(fit.coef_[0, 0]), "ece_10bin": float(ece)}


def calibration_summary(predictions):
    required = {"dataset", "model", "record_id", "y_SCLC", "probability"}
    if required - set(predictions.columns):
        raise ValueError(f"Missing calibration columns: {sorted(required-set(predictions.columns))}")
    rows = []
    for dataset, dataset_frame in predictions.groupby("dataset", sort=False):
        reference = None
        samples = None
        for model, group in dataset_frame.groupby("model", sort=False):
            frame = canonical_rows(group)
            identity = frame[["record_id", "y_SCLC"]]
            if reference is None:
                reference = identity
                samples = shared_bootstrap_indices(frame.y_SCLC, dataset, frame.record_id)
            elif not identity.equals(reference):
                raise ValueError(f"Calibration case/outcome mismatch: {dataset}/{model}")
            y = frame.y_SCLC.to_numpy()
            p = frame.probability.to_numpy(float)
            points = calibration_stats(y, p)
            draws = {key: [] for key in points}
            for index in samples:
                values = calibration_stats(y[index], p[index])
                for key, value in values.items():
                    draws[key].append(value)
            row = {"dataset": dataset, "model": model, "bootstrap_n": BOOTSTRAP_N,
                   "bootstrap_protocol": BOOTSTRAP_PROTOCOL,
                   "bootstrap_case_sha256": bootstrap_identity(y, frame.record_id)}
            for key, value in points.items():
                low, high = np.quantile(draws[key], [0.025, 0.975])
                row.update({key: value, f"{key}_ci_low": float(low), f"{key}_ci_high": float(high)})
            rows.append(row)
    return pd.DataFrame(rows)
