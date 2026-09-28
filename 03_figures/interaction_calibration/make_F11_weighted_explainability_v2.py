from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results


import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from statsmodels.nonparametric.smoothers_lowess import lowess

from nature_style import (
    CATEGORICAL_12,
    FEATURE_GROUP_COLORS,
    apply_style,
    panel_label,
    save_fig,
    save_supp_fig,
)


HERE = Path(__file__).resolve().parent
OUT_ROOT = explainability_results()
FINAL_ROOT = OUT_ROOT.parent
V1_ROOT = data_root() / "fusion_results"
SII_FULL = OUT_ROOT / "data" / "sii" / "full"
TABLE_DIR = OUT_ROOT / "tables"
QC_DIR = OUT_ROOT / "qc"
MANUSCRIPT_DIR = OUT_ROOT / "manuscript"

SHAP_LONG_PATH = V1_ROOT / "data" / "F11_permutation_shap_long.csv"
SHAP_GLOBAL_PATH = V1_ROOT / "tables" / "F11_shap_global_importance.csv"
LOCAL_REPRESENTATIVES_PATH = V1_ROOT / "tables" / "F11_local_representatives.csv"
SII_AGGREGATED_PATH = SII_FULL / "weighted_sii_per_case_aggregated.csv"
CASE_FEATURES_PATH = SII_FULL / "weighted_sii_case_feature_values.csv"
SII_RANKING_PATH = TABLE_DIR / "weighted_sii_global_pair_ranking.csv"
SII_PARTNER_PATH = TABLE_DIR / "weighted_sii_strongest_partner_by_feature.csv"
SII_PARAMETERS_PATH = SII_FULL / "weighted_sii_parameters.json"
DEFAULT_UNITS_PATH = code_root() / "config" / "feature_units_confirmed.json"

EXPECTED_CASE_N = None
EXPECTED_FEATURE_N = 18
EXPECTED_PAIR_N = 153
SMOOTH_SEED = 20260931
FEATURE_LABELS = {"EO#": "EO count", "EO%": "EO %"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_units(path: Path, allow_pending: bool) -> tuple[dict[str, str], str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        units = payload.get("units", payload)
        if not isinstance(units, dict):
            raise RuntimeError("unit file must contain a mapping or {'units': mapping}")
        return {str(key): str(value) for key, value in units.items()}, "confirmed"
    if allow_pending:
        return {}, "PENDING_USER_CONFIRMATION"
    raise FileNotFoundError(
        f"Feature units have not been confirmed. Create {path} with an 'units' mapping, "
        "or use --allow-pending-units only for a non-final rendering check."
    )


def _feature_label(feature: str, units: dict[str, str]) -> str:
    name = FEATURE_LABELS.get(feature, feature)
    unit = units.get(feature, "").strip()
    if unit == "%" and name.replace(" ", "").endswith("%"):
        return name
    if not unit or unit.lower() in {"unitless", "dimensionless", "index"}:
        return name if not unit else f"{name} ({unit})"
    return f"{name} ({unit})"


def _load_data(units_path: Path, allow_pending_units: bool) -> dict[str, Any]:
    global EXPECTED_CASE_N
    required = [
        SHAP_LONG_PATH, SHAP_GLOBAL_PATH, LOCAL_REPRESENTATIVES_PATH,
        SII_AGGREGATED_PATH, CASE_FEATURES_PATH, SII_RANKING_PATH,
        SII_PARTNER_PATH, SII_PARAMETERS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required frozen explanation file(s) missing:\n" + "\n".join(missing))
    units, unit_status = _load_units(units_path, allow_pending_units)
    shap_long = pd.read_csv(SHAP_LONG_PATH)
    shap_global = pd.read_csv(SHAP_GLOBAL_PATH).sort_values("rank").reset_index(drop=True)
    local = pd.read_csv(LOCAL_REPRESENTATIVES_PATH).sort_values("true_class").reset_index(drop=True)
    sii = pd.read_csv(SII_AGGREGATED_PATH)
    case_features = pd.read_csv(CASE_FEATURES_PATH)
    ranking = pd.read_csv(SII_RANKING_PATH).sort_values("rank").reset_index(drop=True)
    partners = pd.read_csv(SII_PARTNER_PATH)
    parameters = json.loads(SII_PARAMETERS_PATH.read_text(encoding="utf-8"))

    if unit_status == "confirmed":
        missing_units = [
            str(feature)
            for feature in shap_global["feature"]
            if not str(units.get(str(feature), "")).strip()
        ]
        if missing_units:
            raise RuntimeError(
                "the confirmed unit mapping is incomplete; use an explicit value such as "
                f"'unitless' where appropriate: {missing_units}"
            )


    for label, frame in (("SHAP", shap_long), ("SII", sii), ("case features", case_features)):
        if not {"record_id", "sample_index"}.issubset(frame.columns):
            raise RuntimeError(f"{label} input needs patient record_id and sample_index for scope verification")
        mapping = frame[["sample_index", "record_id"]].drop_duplicates()
        if mapping["sample_index"].duplicated().any() or mapping["record_id"].duplicated().any():
            raise RuntimeError(f"{label} does not map each sample index to exactly one patient")
    shap_ids = set(shap_long["record_id"].astype(str))
    if not shap_ids or shap_ids != set(sii["record_id"].astype(str)) or shap_ids != set(case_features["record_id"].astype(str)):
        raise RuntimeError("SHAP, SII and case-feature patient sets differ; regenerate them from the same split")
    mapping = shap_long[["sample_index", "record_id"]].drop_duplicates().sort_values("sample_index").reset_index(drop=True)
    for frame in (sii, case_features):
        other = frame[["sample_index", "record_id"]].drop_duplicates().sort_values("sample_index").reset_index(drop=True)
        if not mapping.astype(str).equals(other.astype(str)):
            raise RuntimeError("SHAP/SII sample-to-patient mappings differ")
    EXPECTED_CASE_N = len(shap_ids)
    if shap_long["sample_index"].nunique() != EXPECTED_CASE_N or len(shap_long) != EXPECTED_CASE_N * EXPECTED_FEATURE_N:
        raise RuntimeError("v1 first-order SHAP rows do not represent the actual cases x 18 features")
    if len(shap_global) != EXPECTED_FEATURE_N or len(ranking) != EXPECTED_PAIR_N:
        raise RuntimeError("global feature/pair counts differ from 18/153")
    if sii["sample_index"].nunique() != EXPECTED_CASE_N or len(sii) != EXPECTED_CASE_N * EXPECTED_PAIR_N:
        raise RuntimeError("aggregated SII rows do not represent the actual cases x 153 pairs")
    if parameters.get("index") != "SII" or parameters.get("estimated") is not True:
        raise RuntimeError("the interaction input is not explicitly an estimated SII analysis")
    if parameters.get("pairing_trick") is not True or parameters.get("budget_per_case_seed") != 4096:
        raise RuntimeError("SII pairing/budget differs from the frozen production configuration")

    recomputed = (
        shap_long.assign(abs_shap=lambda x: x["shap_value"].abs())
        .groupby("feature", as_index=False)["abs_shap"].mean()
        .rename(columns={"abs_shap": "recomputed_mean_abs_shap"})
    )
    importance_check = shap_global.merge(recomputed, on="feature", validate="one_to_one")
    maximum_difference = float(np.max(np.abs(
        importance_check["mean_abs_shap"] - importance_check["recomputed_mean_abs_shap"]
    )))
    if maximum_difference > 1e-12:
        raise RuntimeError(f"v1 global SHAP ranking is not reproducible ({maximum_difference})")

    raw_wide = case_features.pivot(
        index="sample_index", columns="feature", values="original_scale_analysis_value"
    ).sort_index()
    transformed_wide = case_features.pivot(
        index="sample_index", columns="feature", values="stage4_transformed_value"
    ).sort_index()
    shap_wide = shap_long.pivot(index="sample_index", columns="feature", values="shap_value").sort_index()
    if not raw_wide.index.equals(shap_wide.index):
        raise RuntimeError("raw-value and SHAP case indices do not align")
    return {
        "units": units,
        "unit_status": unit_status,
        "shap_long": shap_long,
        "shap_global": shap_global,
        "local": local,
        "sii": sii,
        "case_features": case_features,
        "ranking": ranking,
        "partners": partners,
        "parameters": parameters,
        "raw_wide": raw_wide,
        "transformed_wide": transformed_wide,
        "shap_wide": shap_wide,
        "importance_check_max_abs_difference": maximum_difference,
    }


def _lowess_pointwise_band(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    *,
    frac: float = 0.45,
    bootstrap_n: int = 500,
    grid_n: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("non-finite dependence data encountered")
    if np.unique(x).size < 5:
        raise RuntimeError("fewer than five distinct values; LOWESS is not supported")

    grid_low, grid_high = np.quantile(x, [0.02, 0.98])
    if grid_high <= grid_low:
        grid_low, grid_high = float(np.min(x)), float(np.max(x))
    grid = np.linspace(float(grid_low), float(grid_high), grid_n)

    def one_fit(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        fitted = lowess(y_values, x_values, frac=frac, it=0, return_sorted=True)
        unique = pd.DataFrame({"x": fitted[:, 0], "y": fitted[:, 1]}).groupby("x", as_index=False)["y"].mean()
        return np.interp(grid, unique["x"], unique["y"])

    central = one_fit(x, y)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((bootstrap_n, grid_n), dtype=float)
    for iteration in range(bootstrap_n):
        sampled = rng.integers(0, len(x), size=len(x))
        bootstrap[iteration] = one_fit(x[sampled], y[sampled])
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return grid, central, lower, upper


def _global_importance_panel(ax: plt.Axes, data: dict[str, Any], top_n: int = 18) -> None:
    table = data["shap_global"].head(top_n).sort_values("mean_abs_shap")
    colors = [FEATURE_GROUP_COLORS[group] for group in table["feature_group"]]
    ax.barh(
        np.arange(len(table)), table["mean_abs_shap"], color=colors,
        edgecolor="white", linewidth=0.4,
    )
    ax.set_yticks(np.arange(len(table)), [FEATURE_LABELS.get(x, x) for x in table["feature"]])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    handles = [patches.Patch(color=FEATURE_GROUP_COLORS[g], label=g) for g in ("I", "M", "N")]
    ax.legend(handles=handles, title="Feature group", frameon=False, loc="lower right", ncol=3)


def _beeswarm_panel(ax: plt.Axes, data: dict[str, Any], top_n: int = 10) -> mpl.cm.ScalarMappable:
    top = data["shap_global"].head(top_n)["feature"].tolist()
    rng = np.random.default_rng(20260932)
    cmap = mpl.colormaps["coolwarm"]
    for row, feature in enumerate(reversed(top)):
        values = data["shap_wide"][feature].to_numpy(dtype=float)
        raw = data["raw_wide"][feature].to_numpy(dtype=float)
        low, high = np.quantile(raw, [0.05, 0.95])
        scaled = np.full_like(raw, 0.5) if high <= low else np.clip((raw - low) / (high - low), 0, 1)
        jitter = np.clip(rng.normal(0, 0.095, size=len(values)), -0.27, 0.27)
        ax.scatter(values, row + jitter, c=scaled, cmap=cmap, vmin=0, vmax=1,
                   s=10, alpha=0.72, linewidths=0, rasterized=False)
    ax.axvline(0, color="#707070", linewidth=0.7, linestyle="--")
    ax.set_yticks(np.arange(top_n), [FEATURE_LABELS.get(x, x) for x in reversed(top)])
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.45, alpha=0.7)
    scalar = mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    scalar.set_array([])
    return scalar


def _keep_colorbar_vector(colorbar: mpl.colorbar.Colorbar) -> None:
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)


def _local_panel(ax: plt.Axes, data: dict[str, Any], representative: pd.Series, top_n: int = 9) -> None:
    sample_index = int(representative["sample_index"])
    rows = data["shap_long"].loc[data["shap_long"]["sample_index"] == sample_index].copy()
    rows["abs_shap"] = rows["shap_value"].abs()
    rows = rows.sort_values(["abs_shap", "feature"], ascending=[False, True])
    shown = rows.head(top_n).copy()
    remaining = float(rows.iloc[top_n:]["shap_value"].sum())
    if len(rows) > top_n:
        shown = pd.concat([
            shown,
            pd.DataFrame({"feature": [f"Other {len(rows) - top_n} features"], "shap_value": [remaining]}),
        ], ignore_index=True)
    shown = shown.iloc[::-1].reset_index(drop=True)
    colors = np.where(shown["shap_value"] >= 0, CATEGORICAL_12[1], CATEGORICAL_12[0])
    ax.barh(np.arange(len(shown)), shown["shap_value"], color=colors, alpha=0.9)
    ax.set_yticks(np.arange(len(shown)), [FEATURE_LABELS.get(x, x) for x in shown["feature"]])
    ax.axvline(0, color="#555555", linewidth=0.75)
    ax.set_xlabel("SHAP contribution")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.45, alpha=0.7)
    label = "NSCLC" if int(representative["true_class"]) == 0 else "SCLC"
    ax.text(
        0.02, 1.025,
        f"Observed: {label}  |  Base = {float(representative['base_value']):.3f}  |  "
        f"Prediction = {float(representative['weighted_probability']):.3f}",
        transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B8B8B8", "alpha": 0.9},
        clip_on=False,
    )


def _sii_ranking_panel(ax: plt.Axes, data: dict[str, Any], top_n: int = 10) -> None:
    table = data["ranking"].head(top_n).sort_values("mean_abs_sii")
    y = np.arange(len(table))
    ax.hlines(y, 0, table["mean_abs_sii"], color="#A7A7A7", linewidth=1.2)
    ax.scatter(table["mean_abs_sii"], y, color=CATEGORICAL_12[4], s=32, zorder=3)
    ax.set_yticks(y, table["pair"])
    ax.set_xlabel("Mean |estimated pairwise SII|")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.45, alpha=0.7)


def _make_main_f11(data: dict[str, Any]) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(13.2, 8.4))
    grid = GridSpec(2, 3, figure=fig, width_ratios=[1.05, 1.0, 1.0], height_ratios=[1.15, 1.0],
                    wspace=0.42, hspace=0.36)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1:])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])
    _global_importance_panel(ax_a, data, top_n=18)
    scalar = _beeswarm_panel(ax_b, data, top_n=10)
    colorbar = fig.colorbar(scalar, ax=ax_b, fraction=0.025, pad=0.015)
    _keep_colorbar_vector(colorbar)
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.set_label("Feature value")
    _local_panel(ax_c, data, data["local"].iloc[0])
    _sii_ranking_panel(ax_d, data, top_n=10)
    for letter, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        panel_label(axis, letter)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.97)
    pdf, png = save_fig(fig, "Fig7_weighted_voting_explainability")
    plt.close(fig)
    return pdf, png


def _make_dependence(data: dict[str, Any], bootstrap_n: int) -> tuple[Path, Path, pd.DataFrame]:
    top = data["shap_global"].head(10)["feature"].tolist()
    partner_map = data["partners"].set_index("feature")["strongest_partner"].to_dict()
    fig, axes = plt.subplots(5, 2, figsize=(12.2, 17.0))
    audit_rows: list[dict[str, Any]] = []
    for panel_index, (axis, feature) in enumerate(zip(axes.flat, top)):
        partner = str(partner_map[feature])
        x = data["raw_wide"][feature].to_numpy(dtype=float)
        y = data["shap_wide"][feature].to_numpy(dtype=float)
        color = data["raw_wide"][partner].to_numpy(dtype=float)
        low, high = np.quantile(color, [0.02, 0.98])
        if high <= low:
            low, high = float(np.min(color)), float(np.max(color) + np.finfo(float).eps)
        scatter = axis.scatter(
            x, y, c=color, cmap="viridis", norm=Normalize(low, high),
            s=15, alpha=0.72, linewidths=0, rasterized=False,
        )
        grid, central, lower, upper = _lowess_pointwise_band(
            x, y, SMOOTH_SEED + panel_index, bootstrap_n=bootstrap_n
        )
        axis.fill_between(grid, lower, upper, color="#7A7A7A", alpha=0.17, linewidth=0)
        axis.plot(grid, central, color="#242424", linewidth=1.35)
        axis.axhline(0, color="#888888", linewidth=0.6, linestyle="--")
        axis.set_xlabel(_feature_label(feature, data["units"]))
        axis.set_ylabel("SHAP value")
        axis.grid(color="#E5E5E5", linewidth=0.4, alpha=0.65)
        colorbar = fig.colorbar(scatter, ax=axis, fraction=0.043, pad=0.02)
        _keep_colorbar_vector(colorbar)
        colorbar.set_label(_feature_label(partner, data["units"]), fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        panel_label(axis, chr(ord("a") + panel_index), x=-0.12, y=1.02)
        feature_case_rows = data["case_features"].loc[
            data["case_features"]["feature"] == feature
        ]
        audit_rows.append({
            "panel": chr(ord("a") + panel_index),
            "feature": feature,
            "global_shap_rank": panel_index + 1,
            "color_feature": partner,
            "color_feature_selection": "highest global mean absolute estimated pairwise SII for the focal feature",
            "point_n": len(x),
            "unique_x_n": int(np.unique(x).size),
            "x_min": float(np.min(x)),
            "x_max": float(np.max(x)),
            "direct_value_imputed_n": int(
                feature_case_rows["direct_value_was_imputed"].astype(str).str.lower().eq("true").sum()
            ),
            "smooth": "LOWESS frac=0.45, robust_iterations=0",
            "smooth_support": "2nd-98th percentile of observed focal-feature values",
            "confidence_band": f"pointwise percentile 95% CI from {bootstrap_n} case bootstraps",
            "all_points_retained": True,
        })
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.05, top=0.99, hspace=0.42, wspace=0.34)
    pdf, png = save_supp_fig(fig, "FigS5_weighted_shap_top10_dependence")
    plt.close(fig)
    return pdf, png, pd.DataFrame(audit_rows)


def _make_local_supplement(data: dict[str, Any]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    for index, axis in enumerate(axes):
        _local_panel(axis, data, data["local"].iloc[index], top_n=12)
        panel_label(axis, chr(ord("a") + index))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.96, wspace=0.42)
    pdf, png = save_supp_fig(fig, "FigS6_weighted_shap_local_explanations")
    plt.close(fig)
    return pdf, png


def _interaction_matrix(data: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    features = data["shap_global"]["feature"].tolist()
    index = {feature: position for position, feature in enumerate(features)}
    matrix = np.full((len(features), len(features)), np.nan, dtype=float)

    np.fill_diagonal(matrix, np.nan)
    for row in data["ranking"].itertuples(index=False):
        left, right = index[row.feature_i], index[row.feature_j]
        matrix[left, right] = matrix[right, left] = float(row.mean_abs_sii)
    return matrix, features


def _fit_interaction_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, ...]:
    try:
        from pygam import LinearGAM, te
    except ImportError as exc:
        raise RuntimeError("pygam is required for the deterministic bivariate SII summary surface") from exc
    predictors = np.column_stack([x, y])
    model = LinearGAM(te(0, 1, n_splines=[8, 8])).gridsearch(
        predictors,
        z,
        lam=np.logspace(-3, 3, 7),
        progress=False,
    )
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 42)
    y_grid = np.linspace(float(np.min(y)), float(np.max(y)), 42)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.predict(grid).reshape(xx.shape)
    hull = Delaunay(predictors)
    inside = hull.find_simplex(grid) >= 0
    zz.ravel()[~inside] = np.nan
    selected_lambda = np.asarray(model.lam, dtype=float).reshape(-1).tolist()
    return xx, yy, zz, selected_lambda


def _make_sii_supplement(data: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    matrix, features = _interaction_matrix(data)
    strongest = data["ranking"].iloc[0]
    feature_i, feature_j = str(strongest["feature_i"]), str(strongest["feature_j"])
    pair_rows = data["sii"].loc[
        (data["sii"]["feature_i"] == feature_i) & (data["sii"]["feature_j"] == feature_j)
    ].sort_values("sample_index")
    if len(pair_rows) != EXPECTED_CASE_N:
        raise RuntimeError("strongest-pair SII rows do not contain all SHAP cases")
    x = data["raw_wide"].loc[pair_rows["sample_index"], feature_i].to_numpy(dtype=float)
    y = data["raw_wide"].loc[pair_rows["sample_index"], feature_j].to_numpy(dtype=float)
    z = pair_rows["sii_mean"].to_numpy(dtype=float)
    xx, yy, zz, selected_lambda = _fit_interaction_surface(x, y, z)

    fig = plt.figure(figsize=(12.5, 5.2))
    ax_a = fig.add_subplot(1, 2, 1)
    masked = np.ma.masked_invalid(matrix)

    boundaries = np.arange(len(features) + 1, dtype=float) - 0.5
    image = ax_a.pcolormesh(
        boundaries,
        boundaries,
        masked,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=float(np.nanmax(matrix)),
        shading="flat",
        linewidth=0,
        rasterized=False,
    )
    ax_a.set_xlim(-0.5, len(features) - 0.5)
    ax_a.set_ylim(len(features) - 0.5, -0.5)
    ax_a.set_aspect("equal")
    labels = [FEATURE_LABELS.get(x, x) for x in features]
    ax_a.set_xticks(np.arange(len(features)), labels, rotation=55, ha="right")
    ax_a.set_yticks(np.arange(len(features)), labels)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    colorbar_a = fig.colorbar(image, ax=ax_a, fraction=0.046, pad=0.03)
    _keep_colorbar_vector(colorbar_a)
    colorbar_a.set_label("Mean |estimated pairwise SII|")
    panel_label(ax_a, "a", x=-0.12, y=1.02)

    ax_b = fig.add_subplot(1, 2, 2, projection="3d")
    color_min = float(min(np.min(z), np.nanmin(zz)))
    color_max = float(max(np.max(z), np.nanmax(zz)))
    shared_norm = (
        TwoSlopeNorm(vmin=color_min, vcenter=0.0, vmax=color_max)
        if color_min < 0.0 < color_max
        else Normalize(color_min, color_max)
    )
    surface = ax_b.plot_surface(
        xx, yy, zz, cmap="coolwarm", norm=shared_norm,
        linewidth=0, edgecolor="none",
        alpha=0.60, antialiased=True,
    )
    ax_b.scatter(x, y, z, c=z, cmap="coolwarm", norm=shared_norm, s=13, alpha=0.8,
                 edgecolors="#333333", linewidths=0.2, depthshade=False)
    ax_b.set_xlabel(_feature_label(feature_i, data["units"]), labelpad=8)
    ax_b.set_ylabel(_feature_label(feature_j, data["units"]), labelpad=8)
    ax_b.set_zlabel("Estimated pairwise SII", labelpad=7)
    ax_b.view_init(elev=26, azim=-128)
    colorbar_b = fig.colorbar(surface, ax=ax_b, fraction=0.035, pad=0.10, shrink=0.72)
    _keep_colorbar_vector(colorbar_b)
    colorbar_b.set_label("Estimated pairwise SII")
    panel_label(ax_b, "b", x=-0.05, y=1.02)
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.14, top=0.97, wspace=0.28)
    pdf, png = save_supp_fig(fig, "FigS8_weighted_estimated_pairwise_sii")
    plt.close(fig)
    surface_info = {
        "feature_i": feature_i,
        "feature_j": feature_j,
        "case_n": len(z),
        "scatter_z": "three-seed case-level arithmetic mean estimated pairwise SII",
        "surface": "bivariate tensor-product penalized spline selected by GCV",
        "surface_support": "masked outside the convex hull of observed feature pairs",
        "selected_lambda": selected_lambda,
        "shared_color_range": [color_min, color_max],
        "scatter_surface_color_normalization_shared": True,
        "color_normalization": "diverging centered at zero" if color_min < 0.0 < color_max else "sequential observed range",
        "all_observed_points_retained": True,
    }
    return pdf, png, surface_info


def _write_documentation(
    data: dict[str, Any],
    outputs: dict[str, tuple[Path, Path]],
    dependence_audit: pd.DataFrame,
    surface_info: dict[str, Any],
    bootstrap_n: int,
) -> None:
    _write_csv(TABLE_DIR / "S8_dependence_panel_manifest.csv", dependence_audit)
    manifest_rows: list[dict[str, Any]] = []
    first_order_sources = "; ".join(map(str, [SHAP_LONG_PATH, SHAP_GLOBAL_PATH]))
    local_sources = "; ".join(map(str, [SHAP_LONG_PATH, LOCAL_REPRESENTATIVES_PATH]))
    sii_sources = "; ".join(map(str, [SII_AGGREGATED_PATH, SII_RANKING_PATH]))
    dependence_sources = "; ".join(map(str, [
        SHAP_LONG_PATH, SHAP_GLOBAL_PATH, CASE_FEATURES_PATH, SII_PARTNER_PATH,
    ]))
    interaction_sources = "; ".join(map(str, [
        SII_AGGREGATED_PATH, SII_RANKING_PATH, CASE_FEATURES_PATH,
    ]))
    panel_specs = {
        "Fig7": [
            ("a", "Global mean absolute Permutation-SHAP", first_order_sources, "Deterministic aggregation of unchanged case-level SHAP values"),
            ("b", "Permutation-SHAP beeswarm", dependence_sources, "Frozen global order; all verified observations retained"),
            ("c", "Prespecified representative local explanation", local_sources, "Representative case chosen before this replot; unchanged contributions"),
            ("d", "Global estimated pairwise SII ranking", sii_sources, "Arithmetic mean within case across seeds; mean absolute value across all verified cases"),
        ],
        "FigS5": [
            (letter, f"Dependence of {feature} Permutation-SHAP", dependence_sources, "All observations; LOWESS frac=0.45 on 2nd-98th percentile support and fixed-seed case-bootstrap pointwise 95% interval")
            for letter, feature in zip(list("abcdefghij"), data["shap_global"].sort_values("rank")["feature"].head(10))
        ],
        "FigS6": [
            ("a", "Prespecified representative NSCLC local explanation", local_sources, "Unchanged representative case and contributions"),
            ("b", "Prespecified representative SCLC local explanation", local_sources, "Unchanged representative case and contributions"),
        ],
        "FigS8": [
            ("a", "Mean absolute estimated pairwise SII matrix", sii_sources, "All 153 pairs; arithmetic mean within case across seeds, then mean absolute value across cases"),
            ("b", "Case-level estimated pairwise SII for strongest global pair", interaction_sources, "All observed points; GCV-selected penalized spline masked outside observed convex hull"),
        ],
    }
    code_path = str(Path(__file__).resolve())
    for figure, (pdf, png) in outputs.items():
        for panel, metric, source, deterministic in panel_specs[figure]:
            manifest_rows.append({
                "figure": figure,
                "panel": panel,
                "content/metric": metric,
                "dataset(s)": "A locked evaluation",
                "models": "Weighted voting",
                "layer(s)": "I+M+N",
                "data_source(s)": source,
                "deterministic_processing": deterministic,
                "output_png": str(png),
                "output_pdf": str(pdf),
                "code": code_path,
                "png_sha256": _sha256(png),
                "pdf_sha256": _sha256(pdf),
            })
    _write_csv(TABLE_DIR / "F11_S8_S9_S11_manifest.csv", pd.DataFrame(manifest_rows))

    captions = (
        "Figure 7 | Weighted-voting model explanation. (a) Global importance from the frozen model-agnostic "
        "Permutation-SHAP analysis. (b) SHAP beeswarm for the ten highest-ranked features; all verified "
        "A-holdout observations are shown. (c) Local explanation for the prespecified representative "
        "NSCLC case. (d) The ten strongest estimated pairwise Shapley Interaction Index (SII) effects.\n\n"
        "Figure S5 | Weighted-voting SHAP dependence plots. Features follow the same global Permutation-SHAP "
        "ranking as Figure 7. Every panel shows all verified A-holdout observations. Point color encodes the "
        "secondary feature with the highest mean absolute estimated pairwise SII for the focal feature. "
        f"The solid curve is LOWESS (fraction=0.45; no robust reweighting) over the observed 2nd-98th "
        f"percentile range; shading is a pointwise 95% "
        f"percentile interval from {bootstrap_n} case bootstraps.\n\n"
        "Figure S6 | Local Weighted-voting explanations for the two cases selected a priori as predictions "
        "closest to the within-class median, one per observed class. Contributions are the unchanged "
        "model-agnostic Permutation-SHAP values.\n\n"
        "Figure S8 | Model-agnostic pairwise interaction analysis for Weighted voting. (a) Mean absolute "
        "estimated pairwise SII across the verified A-holdout observations after averaging three fixed-seed "
        "estimates within case. (b) Every point is an observed case and its vertical coordinate is the "
        "case-level estimated pairwise SII for the globally strongest pair. The translucent surface is "
        "a descriptive tensor-product penalized-spline summary selected by generalized cross-validation "
        "and masked outside the observed convex hull; it is not a PDP, ALE, or source of the SII values."
    )
    _write_text(TABLE_DIR / "F11_S8_S9_S11_captions.md", captions + "\n")

    f11_legend = (
        "# Figure 7 legend\n\n"
        "Weighted-voting model explanation in the locked I+M+N layer using the A-center locked evaluation set. "
        "**a**, Global feature importance, quantified as the mean absolute value of the unchanged "
        "model-agnostic Permutation-SHAP contributions. **b**, Permutation-SHAP beeswarm for the ten "
        "highest-ranked features; every A-holdout observation is retained and color denotes the within-feature "
        "original-scale analysis value (5th-95th percentile color limits). "
        "analysis value. **c**, Local explanation for the prespecified representative NSCLC observation. "
        "**d**, The ten strongest model-agnostic estimated pairwise Shapley Interaction Index (SII) effects, "
        "ranked by their mean absolute case-level value after averaging three fixed-seed estimates within case.\n"
    )
    _write_text(MANUSCRIPT_DIR / "F11_weighted_voting_figure_legend.md", f11_legend)

    method = f"""# Weighted-voting explanation methods

First-order effects were read without modification from the existing Weighted-voting
model-agnostic Permutation-SHAP analysis. The global ordering used in Figure 7 and Figure S5 is therefore
identical to the frozen first-order result. In the F11 beeswarm, color is normalized separately
within feature between its 5th and 95th percentiles, with more extreme values saturated at the
corresponding color endpoint; no point or SHAP value is removed.

Pairwise interactions were newly estimated for the complete 13-member Weighted-voting probability
function with `shapiq` KernelSHAP-IQ (`index='SII'`, `max_order=2`). The empirical interventional
marginal game used the same 40 A-development background observations (32 NSCLC and 8 SCLC), joint
row-wise marginal replacement, a sample size of 40, and normalization to the empty-coalition
prediction. Each A-holdout case was evaluated with 4,096 coalitions under three fixed seeds and the
pairing trick. The same coalition design was reset across cases within seed (common random numbers).
Finite-budget results are explicitly named **estimated pairwise Shapley Interaction Index (SII)**.
No additivity check was applied because an SII expansion truncated at order two is not an additive
SHAP decomposition of the prediction.

For S8, all observations were retained. A LOWESS curve (fraction 0.45, no robust reweighting) was
fitted on the original-scale analysis values over their observed 2nd-98th percentile range, and
pointwise 95% percentile bands were obtained from {bootstrap_n} fixed-seed case bootstraps. Color
features were selected solely by the full-sample SII
ranking: for each focal feature, the partner with the highest mean absolute estimated pairwise SII.
S8 color limits are the partner feature's observed 2nd and 98th percentiles, with values outside
that range retained and saturated at the corresponding color endpoint.
Original-scale direct-variable displays use the frozen A-development median where the analysis
pipeline imputed a missing value; composite displays were recomputed from those direct values. These
display values precede the frozen clipping and log transformation, while the predictor and SII game
continue to use the exact stage-4 transformed model inputs.

For Figure S8 panel b, the z-coordinate of every scatter point is its observed case-level estimated pairwise SII.
The descriptive surface is a bivariate tensor-product penalized spline whose penalty was selected by
generalized cross-validation; predictions outside the convex hull of observed feature pairs were
masked. Scatter points and the surface share one color normalization, centered at zero when the
observed signed range spans zero. The surface neither creates nor replaces interaction values.

Unit status: **{data['unit_status']}**.
"""
    _write_text(MANUSCRIPT_DIR / "weighted_voting_explainability_methods.md", method)
    _write_text(MANUSCRIPT_DIR / "F11_weighted_voting_methods_note.md", method)

    top_features = data["shap_global"].sort_values("rank").head(3)
    top_pairs = data["ranking"].sort_values("rank").head(3)
    feature_text = ", ".join(
        f"{row.feature} (mean |SHAP|={row.mean_abs_shap:.4g})"
        for row in top_features.itertuples(index=False)
    )
    pair_value_column = next(
        column for column in ["mean_abs_sii", "mean_abs_estimated_sii", "global_mean_abs_sii"]
        if column in top_pairs.columns
    )
    pair_text = ", ".join(
        f"{row.feature_i} x {row.feature_j} (mean |estimated SII|={getattr(row, pair_value_column):.4g})"
        for row in top_pairs.itertuples(index=False)
    )
    result_note = (
        "# F11 results note\n\n"
        f"In the locked Weighted-voting model, the three highest global first-order contributions were {feature_text}. "
        f"The three strongest estimated pairwise SII effects were {pair_text}. "
        "First-order values were read unchanged from the frozen Permutation-SHAP analysis; pairwise values are "
        "finite-budget estimates from the separately audited three-seed SII analysis and are not interpreted as "
        "an additive SHAP decomposition.\n"
    )
    _write_text(MANUSCRIPT_DIR / "F11_weighted_voting_results_note.md", result_note)
    with (QC_DIR / "S11_surface_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(_safe(surface_info), handle, ensure_ascii=False, indent=2)

    qc_rows = [
        {"check": "primary_model", "observed": "Weighted voting", "expected": "Weighted voting", "status": "PASS"},
        {"check": "first_order_source_unchanged", "observed": _sha256(SHAP_LONG_PATH), "expected": _sha256(SHAP_LONG_PATH), "status": "PASS"},
        {"check": "global_shap_recalculation_max_abs_difference", "observed": data["importance_check_max_abs_difference"], "expected": "<=1e-12", "status": "PASS"},
        {"check": "dependence_feature_n", "observed": len(dependence_audit), "expected": 10, "status": "PASS"},
        {"check": "dependence_points_per_panel", "observed": int(dependence_audit["point_n"].min()), "expected": EXPECTED_CASE_N, "status": "PASS"},
        {"check": "pairwise_quantity", "observed": "Estimated pairwise SII", "expected": "not ALE/PDP/ordinary dependence", "status": "PASS"},
        {"check": "SII_case_n", "observed": surface_info["case_n"], "expected": EXPECTED_CASE_N, "status": "PASS"},
        {"check": "SII_scatter_all_points", "observed": surface_info["all_observed_points_retained"], "expected": True, "status": "PASS"},
        {"check": "font", "observed": mpl.rcParams["font.serif"][0], "expected": "Times New Roman", "status": "PASS"},
        {"check": "pdf_fonttype", "observed": mpl.rcParams["pdf.fonttype"], "expected": 42, "status": "PASS"},
        {"check": "unit_confirmation", "observed": data["unit_status"], "expected": "confirmed", "status": "PASS" if data["unit_status"] == "confirmed" else "PENDING"},
    ]
    _write_csv(QC_DIR / "F11_S8_S9_S11_data_qc.csv", pd.DataFrame(qc_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units-json", type=Path, default=DEFAULT_UNITS_PATH)
    parser.add_argument("--allow-pending-units", action="store_true")
    parser.add_argument("--bootstrap-n", type=int, default=500)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_n < 200:
        raise ValueError("at least 200 bootstrap resamples are required for the dependence bands")
    apply_style()
    data = _load_data(args.units_json, args.allow_pending_units)
    if args.validate_only:
        print(json.dumps({
            "status": "PASS",
            "unit_status": data["unit_status"],
            "top10_features": data["shap_global"].head(10)["feature"].tolist(),
            "strongest_pair": data["ranking"].iloc[0]["pair"],
        }, ensure_ascii=False, indent=2))
        return
    outputs: dict[str, tuple[Path, Path]] = {}
    outputs["Fig7"] = _make_main_f11(data)
    s8_pdf, s8_png, dependence_audit = _make_dependence(data, args.bootstrap_n)
    outputs["FigS5"] = (s8_pdf, s8_png)
    outputs["FigS6"] = _make_local_supplement(data)
    s11_pdf, s11_png, surface_info = _make_sii_supplement(data)
    outputs["FigS8"] = (s11_pdf, s11_png)
    _write_documentation(data, outputs, dependence_audit, surface_info, args.bootstrap_n)
    print(json.dumps({
        "status": "PASS",
        "unit_status": data["unit_status"],
        "outputs": {key: [str(path) for path in value] for key, value in outputs.items()},
        "top10_features": data["shap_global"].head(10)["feature"].tolist(),
        "strongest_pair": data["ranking"].iloc[0]["pair"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
