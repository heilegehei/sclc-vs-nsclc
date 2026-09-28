from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nature_style import apply_style


HERE = Path(__file__).resolve().parent
V4_ROOT = HERE.parent
FINAL_ROOT = V4_ROOT.parent
V2_ROOT = data_root() / "explainability_results"
V1_ROOT = data_root() / "fusion_results"
ARCHIVED_RENDERER = HERE.parent / "interaction_calibration" / "make_F11_weighted_explainability_v2.py"
MAIN_OUT = publication_results() / "figures" / "main"
SUPP_OUT = publication_results() / "figures" / "supplementary"


def _require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required frozen explanation source(s) missing:\n" + "\n".join(missing))


def _load_archived_renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("archived_weighted_explainability", ARCHIVED_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load archived renderer: {ARCHIVED_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.V1_ROOT = V1_ROOT
    module.SII_FULL = V2_ROOT / "data" / "sii" / "full"
    module.TABLE_DIR = V2_ROOT / "tables"
    module.DEFAULT_UNITS_PATH = code_root() / "config" / "feature_units_confirmed.json"
    return module


def _save_pair(fig: plt.Figure, directory: Path, stem: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / f"{stem}.pdf"
    png = directory / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    return pdf, png


def _global_coolwarm_panel(ax: plt.Axes, data: dict, top_n: int = 18) -> None:
    table = data["shap_global"].head(top_n).sort_values("mean_abs_shap").reset_index(drop=True)

    colors = mpl.colormaps["coolwarm"](np.linspace(0.10, 0.90, len(table)))
    ax.barh(np.arange(len(table)), table["mean_abs_shap"], color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(np.arange(len(table)), [data["feature_labels"].get(x, x) for x in table["feature"]])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.7)


def _local_coolwarm_panel(module: ModuleType, ax: plt.Axes, data: dict, representative, top_n: int = 9) -> None:
    sample_index = int(representative["sample_index"])
    rows = data["shap_long"].loc[data["shap_long"]["sample_index"] == sample_index].copy()
    rows["abs_shap"] = rows["shap_value"].abs()
    rows = rows.sort_values(["abs_shap", "feature"], ascending=[False, True])
    shown = rows.head(top_n).copy()
    remaining = float(rows.iloc[top_n:]["shap_value"].sum())
    if len(rows) > top_n:
        shown = pd.concat([
            rows.head(top_n).copy(),
            pd.DataFrame({"feature": [f"Other {len(rows) - top_n} features"], "shap_value": [remaining]}),
        ], ignore_index=True)
    else:
        shown = rows.head(top_n).copy()
    shown = shown.iloc[::-1].reset_index(drop=True)
    cmap = mpl.colormaps["coolwarm"]
    colours = [cmap(0.90) if value >= 0 else cmap(0.10) for value in shown["shap_value"].to_numpy(float)]
    ax.barh(np.arange(len(shown)), shown["shap_value"], color=colours, alpha=0.92)
    ax.set_yticks(np.arange(len(shown)), [module.FEATURE_LABELS.get(x, x) for x in shown["feature"]])
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


def _render_interactions(module: ModuleType, data: dict) -> tuple[Path, Path]:
    matrix, features = module._interaction_matrix(data)
    strongest = data["ranking"].iloc[0]
    feature_i, feature_j = str(strongest["feature_i"]), str(strongest["feature_j"])
    pair_rows = data["sii"].loc[
        (data["sii"]["feature_i"] == feature_i) & (data["sii"]["feature_j"] == feature_j)
    ].sort_values("sample_index")
    if len(pair_rows) != module.EXPECTED_CASE_N:
        raise RuntimeError("Frozen strongest-pair SII rows are incomplete.")
    x = data["raw_wide"].loc[pair_rows["sample_index"], feature_i].to_numpy(dtype=float)
    y = data["raw_wide"].loc[pair_rows["sample_index"], feature_j].to_numpy(dtype=float)
    z = pair_rows["sii_mean"].to_numpy(dtype=float)
    xx, yy, zz, _ = module._fit_interaction_surface(x, y, z)

    fig = plt.figure(figsize=(12.5, 5.2), facecolor="white")
    ax_a = fig.add_subplot(1, 2, 1)
    masked = np.ma.masked_invalid(matrix)
    boundaries = np.arange(len(features) + 1, dtype=float) - 0.5
    image = ax_a.pcolormesh(
        boundaries, boundaries, masked, cmap="coolwarm", vmin=0.0,
        vmax=float(np.nanmax(matrix)), shading="flat", linewidth=0, rasterized=False,
    )
    ax_a.set_xlim(-0.5, len(features) - 0.5)
    ax_a.set_ylim(len(features) - 0.5, -0.5)
    ax_a.set_aspect("equal")
    labels = [module.FEATURE_LABELS.get(item, item) for item in features]
    ax_a.set_xticks(np.arange(len(features)), labels, rotation=55, ha="right")
    ax_a.set_yticks(np.arange(len(features)), labels)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    colorbar_a = fig.colorbar(image, ax=ax_a, fraction=0.046, pad=0.03)
    module._keep_colorbar_vector(colorbar_a)
    colorbar_a.set_label("Mean |estimated pairwise SII|")
    module.panel_label(ax_a, "a", x=-0.12, y=1.02)

    ax_b = fig.add_subplot(1, 2, 2, projection="3d")
    color_min = float(min(np.min(z), np.nanmin(zz)))
    color_max = float(max(np.max(z), np.nanmax(zz)))
    shared_norm = (
        module.TwoSlopeNorm(vmin=color_min, vcenter=0.0, vmax=color_max)
        if color_min < 0.0 < color_max else module.Normalize(color_min, color_max)
    )
    surface = ax_b.plot_surface(
        xx, yy, zz, cmap="coolwarm", norm=shared_norm, linewidth=0,
        edgecolor="none", alpha=0.60, antialiased=True,
    )
    ax_b.scatter(x, y, z, c=z, cmap="coolwarm", norm=shared_norm, s=13, alpha=0.8,
                 edgecolors="#333333", linewidths=0.2, depthshade=False)
    ax_b.set_xlabel(module._feature_label(feature_i, data["units"]), labelpad=8)
    ax_b.set_ylabel(module._feature_label(feature_j, data["units"]), labelpad=8)
    ax_b.set_zlabel("Estimated pairwise SII", labelpad=7)
    ax_b.view_init(elev=26, azim=-128)
    colorbar_b = fig.colorbar(surface, ax=ax_b, fraction=0.035, pad=0.10, shrink=0.72)
    module._keep_colorbar_vector(colorbar_b)
    colorbar_b.set_label("Estimated pairwise SII")
    module.panel_label(ax_b, "b", x=-0.05, y=1.02)
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.14, top=0.97, wspace=0.28)
    outputs = _save_pair(fig, SUPP_OUT, "FigS8_weighted_pairwise_sii")
    plt.close(fig)
    return outputs


def main() -> None:
    _require([
        ARCHIVED_RENDERER,
        V1_ROOT / "data" / "F11_permutation_shap_long.csv",
        V1_ROOT / "tables" / "F11_shap_global_importance.csv",
        V1_ROOT / "tables" / "F11_local_representatives.csv",
        V2_ROOT / "data" / "sii" / "full" / "weighted_sii_per_case_aggregated.csv",
        V2_ROOT / "data" / "sii" / "full" / "weighted_sii_case_feature_values.csv",
        V2_ROOT / "tables" / "weighted_sii_global_pair_ranking.csv",
        V2_ROOT / "tables" / "weighted_sii_strongest_partner_by_feature.csv",
        V2_ROOT / "data" / "sii" / "full" / "weighted_sii_parameters.json",
        code_root() / "config" / "feature_units_confirmed.json",
    ])
    apply_style()
    module = _load_archived_renderer()
    data = module._load_data(module.DEFAULT_UNITS_PATH, allow_pending_units=False)

    data["feature_labels"] = module.FEATURE_LABELS
    module._global_importance_panel = _global_coolwarm_panel
    module._local_panel = lambda ax, values, representative, top_n=9: _local_coolwarm_panel(
        module, ax, values, representative, top_n=top_n
    )

    module.save_fig = lambda fig, _basename: _save_pair(fig, MAIN_OUT, "Fig7_weighted_voting_explainability")
    f16 = module._make_main_f11(data)
    module.save_supp_fig = lambda fig, _basename: _save_pair(fig, SUPP_OUT, "FigS6_weighted_shap_local_explanations")
    f18 = module._make_local_supplement(data)
    f20 = _render_interactions(module, data)
    for path in (*f16, *f18, *f20):
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
