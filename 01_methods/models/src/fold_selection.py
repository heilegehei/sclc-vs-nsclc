from __future__ import annotations
import warnings
from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from model_factory import STAGE4_DIRECT_FEATURES as DIRECT_FEATURES, STAGE4_COMPOSITE_FEATURES as COMPOSITE_FEATURES, derive_seed as model_seed
SELECTION_PROTOCOL = "manuscript-fold-local-v1"
MASTER_SEED = 20260902
RFE_N_FEATURES = 10
DEFAULT_BORUTA_ESTIMATORS = 200
DEFAULT_BORUTA_MAX_ITER = 30
EXPECTED_DIRECT = ("WBC", "PLT", "LDH", "PCT", "EO#", "EO%", "ALB", "TP", "GLB", "TC", "LDL-C", "MCH", "MCHC")

def _scaled_direct(x: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    matrix = x.loc[:, list(DIRECT_FEATURES)].to_numpy(dtype=float)
    fitted = scaler if scaler is not None else StandardScaler().fit(matrix)
    return fitted.transform(matrix), fitted


def _ensure_nonempty(indices: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    return np.unique(np.asarray(indices, dtype=int).reshape(-1))


def select_lasso(x_train: pd.DataFrame, y: np.ndarray, seed: int, C: float = 1.0) -> dict[str, Any]:
    scaled, scaler = _scaled_direct(x_train)
    estimator = LogisticRegression(
        penalty="l1", solver="liblinear", C=float(C), class_weight="balanced",
        max_iter=2000, random_state=int(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(scaled, y)
    coef = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    raw = np.flatnonzero(np.abs(coef) > 1e-8).astype(int)
    selected = _ensure_nonempty(raw, np.abs(coef))
    return {
        "method": "lasso",
        "selected_indices": selected,
        "selected_features": [DIRECT_FEATURES[int(i)] for i in selected],
        "n_selected": int(len(selected)),
        "fallback_used": False,
        "parameters": {"penalty": "l1", "C": float(C), "solver": "liblinear", "class_weight": "balanced"},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def select_rfe(x_train: pd.DataFrame, y: np.ndarray, seed: int, n_features: int = RFE_N_FEATURES) -> dict[str, Any]:
    scaled, scaler = _scaled_direct(x_train)
    target_n = max(1, min(int(n_features), scaled.shape[1]))
    estimator = LogisticRegression(
        penalty="l2", solver="liblinear", C=1.0, class_weight="balanced",
        max_iter=2000, random_state=int(seed),
    )
    selector = RFE(estimator=estimator, n_features_to_select=target_n, step=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(scaled, y)
    indices = np.flatnonzero(np.asarray(selector.support_, dtype=bool)).astype(int)
    selected = _ensure_nonempty(indices, -np.asarray(selector.ranking_, dtype=float))
    return {
        "method": "rfe",
        "selected_indices": selected,
        "selected_features": [DIRECT_FEATURES[int(i)] for i in selected],
        "n_selected": int(len(selected)),
        "fallback_used": False,
        "parameters": {"estimator": "LogisticRegression(L2)", "C": 1.0, "n_features_to_select": target_n, "step": 1},
        "ranking": np.asarray(selector.ranking_, dtype=int).tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def select_boruta(
    x_train: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    *,
    n_estimators: int = DEFAULT_BORUTA_ESTIMATORS,
    max_iter: int = DEFAULT_BORUTA_MAX_ITER,
    perc: int = 100,
) -> dict[str, Any]:

    direct = x_train.loc[:, list(DIRECT_FEATURES)].to_numpy(dtype=float)
    fallback_used = False
    backend = "boruta_py"
    try:
        from boruta import BorutaPy

        estimator = RandomForestClassifier(
            n_estimators=int(n_estimators), max_features="sqrt",
            class_weight="balanced", random_state=int(seed), n_jobs=1,
        )
        selector = BorutaPy(
            estimator=estimator, n_estimators=int(n_estimators), perc=int(perc),
            alpha=0.05, two_step=True, max_iter=int(max_iter),
            random_state=int(seed), verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selector.fit(direct, y)
        raw = np.flatnonzero(np.asarray(selector.support_, dtype=bool)).astype(int)
        ranks = -np.asarray(getattr(selector, "ranking_", np.ones(direct.shape[1])), dtype=float)
        selected = _ensure_nonempty(raw, ranks)
        ranking = np.asarray(getattr(selector, "ranking_", np.ones(direct.shape[1])), dtype=int).tolist()
    except Exception as exc:
        raise RuntimeError("Required BorutaPy selection failed; no substitute is permitted") from exc
    result: dict[str, Any] = {
        "method": "boruta",
        "backend": backend,
        "selected_indices": selected,
        "selected_features": [DIRECT_FEATURES[int(i)] for i in selected],
        "n_selected": int(len(selected)),
        "fallback_used": bool(fallback_used),
        "parameters": {"n_estimators": int(n_estimators), "max_iter": int(max_iter), "perc": int(perc), "alpha": 0.05},
        "ranking": ranking,
    }
    return result


def _consensus(selection_records: Sequence[dict[str, Any]], min_support: int = 2) -> tuple[list[str], dict[str, int]]:
    counts = {name: 0 for name in DIRECT_FEATURES}
    for record in selection_records:
        for name in record["selected_features"]:
            counts[str(name)] += 1
    threshold = max(1, min(int(min_support), len(selection_records)))
    selected = [name for name in DIRECT_FEATURES if counts[name] >= threshold]
    return selected, counts


def fit_selectors(
    x_train: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    *,
    boruta_estimators: int,
    boruta_max_iter: int,
    lasso_C: float = 1.0,
    rfe_n_features: int = RFE_N_FEATURES,
    boruta_perc: int = 100,
    consensus_min_support: int = 2,
) -> dict[str, Any]:
    records = [
        select_lasso(x_train, y, model_seed(seed, "selector", "lasso"), C=float(lasso_C)),
        select_rfe(x_train, y, model_seed(seed, "selector", "rfe"), int(rfe_n_features)),
        select_boruta(
            x_train, y, model_seed(seed, "selector", "boruta"),
            n_estimators=boruta_estimators, max_iter=boruta_max_iter, perc=int(boruta_perc),
        ),
    ]
    selected, counts = _consensus(records, min_support=int(consensus_min_support))
    return {
        "methods": records,
        "consensus_rule": f"at_least_{int(consensus_min_support)}_of_3",
        "selected_direct": selected,
        "selected_composites": list(COMPOSITE_FEATURES),
        "selected_features": selected + list(COMPOSITE_FEATURES),
        "n_selected_direct": len(selected),
        "n_selected_total": len(selected) + len(COMPOSITE_FEATURES),
        "consensus_counts": counts,
    }


def fit_scope_selection(x_train, y_train, fold_number=None):
    seed = model_seed(MASTER_SEED, "holdout_fit") if fold_number is None else model_seed(MASTER_SEED, "cv", int(fold_number))
    result = fit_selectors(x_train, y_train, seed, boruta_estimators=200,
                           boruta_max_iter=30, lasso_C=1.0, rfe_n_features=10,
                           boruta_perc=100, consensus_min_support=2)
    if fold_number is None and list(result["selected_direct"]) != list(EXPECTED_DIRECT):
        raise RuntimeError("Full-development consensus differs from the manuscript's frozen 13 direct predictors; review before producing results")
    result["selection_protocol"] = SELECTION_PROTOCOL
    return result


def selected_layer_specs(selection):
    inflammation = set(DIRECT_FEATURES[:11])
    immune = set(DIRECT_FEATURES[11:17])
    chosen = selection["selected_direct"]
    if set(chosen) - set(DIRECT_FEATURES):
        raise ValueError("Unknown selected direct candidate")
    return {
        "I": {"label": "I", "direct": tuple(f for f in chosen if f in inflammation), "composites": ("SIRI",)},
        "I_M": {"label": "I + M", "direct": tuple(f for f in chosen if f in inflammation | immune), "composites": ("SIRI", "LMR", "GAR")},
        "I_M_N": {"label": "I + M + N", "direct": tuple(chosen), "composites": tuple(COMPOSITE_FEATURES)},
    }


def require_selection_protocol(source):
    if isinstance(source, dict):
        valid = source.get("selection_protocol") == SELECTION_PROTOCOL
    else:
        valid = ("selection_protocol" in source.columns and not source.empty
                 and source["selection_protocol"].eq(SELECTION_PROTOCOL).all())
    if not valid:
        raise RuntimeError("Stale or non-manuscript reference: first regenerate common-scheme weights and A/BC fusion predictions with selection_protocol=" + SELECTION_PROTOCOL)
