from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from nature_style import FEATURE_GROUP_COLORS, apply_style, save_supp_fig


_BOOTSTRAP_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "manuscript_bootstrap.py").is_file())
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))
from common.manuscript_bootstrap import BOOTSTRAP_PROTOCOL, bootstrap_seed, canonical_rows, shared_bootstrap_indices, bootstrap_identity

HERE = Path(__file__).resolve().parent
FINAL_ROOT = HERE.parent.parent
WEIGHTED_SHAP = data_root() / "fusion_results" / "data" / "F11_permutation_shap_long.csv"


def main() -> None:
    if not WEIGHTED_SHAP.is_file():
        raise FileNotFoundError(f"Missing frozen SHAP input: {WEIGHTED_SHAP}")

    long = pd.read_csv(WEIGHTED_SHAP)
    required = {"sample_index", "record_id", "y_SCLC", "feature", "shap_value", "feature_group"}
    missing = sorted(required.difference(long.columns))
    if missing:
        raise RuntimeError(f"Frozen SHAP input is missing columns: {missing}")

    identities = long[["sample_index", "record_id", "y_SCLC"]].drop_duplicates()
    if identities["sample_index"].duplicated().any():
        raise RuntimeError("Conflicting SHAP case identities or outcomes")
    identities = canonical_rows(identities)
    matrix = long.pivot(index="sample_index", columns="feature", values="shap_value")
    matrix = matrix.reindex(identities["sample_index"].to_numpy())
    if matrix.isna().any().any():
        raise RuntimeError("Incomplete SHAP matrix; regenerate the case-level explanations")
    global_order = matrix.abs().mean().sort_values(ascending=False, kind="mergesort").index.tolist()
    values = matrix[global_order].to_numpy(float)
    samples = shared_bootstrap_indices(identities["y_SCLC"].to_numpy(), "A_holdout_clean", identities["record_id"].to_numpy())
    draws = np.asarray([np.mean(np.abs(values[selected, :]), axis=0) for selected in samples])

    estimate = np.mean(np.abs(values), axis=0)
    low_values = np.quantile(draws, 0.025, axis=0)
    high_values = np.quantile(draws, 0.975, axis=0)
    feature_groups = long.drop_duplicates("feature").set_index("feature")["feature_group"].to_dict()

    apply_style()
    fig, axis = plt.subplots(figsize=(8.3, 8.8), facecolor="white")
    y = np.arange(len(global_order))
    for yi, feature, value, low, high in zip(y, global_order, estimate, low_values, high_values, strict=True):
        color = FEATURE_GROUP_COLORS[feature_groups[feature]]
        axis.errorbar(
            value,
            yi,
            xerr=np.array([[value - low], [high - value]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=4.8,
            elinewidth=1.1,
            capsize=2.2,
        )
    axis.set_yticks(y, global_order)
    axis.invert_yaxis()
    axis.set_xlabel("Mean |SHAP value| (2,000 stratified-bootstrap 95% interval)")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.45)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=FEATURE_GROUP_COLORS[group], label=label)
        for group, label in [("I", "Inflammation"), ("M", "Immune"), ("N", "Nutrition/metabolism")]
    ]
    axis.legend(handles=handles, frameon=False, loc="lower right")
    outputs = save_supp_fig(fig, "FigS7_weighted_shap_stability")
    plt.close(fig)
    print(f"SOURCE={WEIGHTED_SHAP}")
    print(f"WROTE {outputs[0]}")
    print(f"WROTE {outputs[1]}")


if __name__ == "__main__":
    main()
