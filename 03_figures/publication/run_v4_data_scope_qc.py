from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results, publication_results

import csv
import subprocess
from pathlib import Path

import pandas as pd


ROOT = publication_results()
QC = ROOT / "qc"
QC.mkdir(exist_ok=True)

A_EVAL = "A_holdout_clean"
B_EXT = "B_external"
C_EXT = "C_external"
FUSIONS = {
    "weighted_voting_cv_auc",
    "soft_voting_native_probability",
    "stacking_in_sample_A_train",
}
LAYER_MODEL = "weighted_voting_cv_auc"
LAYER_DATASETS = [A_EVAL, B_EXT, C_EXT]
LAYER_SET = {"I", "I_M", "I_M_N"}
SINGLE = {
    "logistic_regression", "gam", "knn", "rbf_svm", "gaussian_nb",
    "decision_tree", "random_forest", "extra_trees", "gbdt", "xgboost",
    "lightgbm", "adaboost", "rotation_forest", "mlp",
}


def check(name: str, passed: bool, evidence: str) -> dict[str, object]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def pdf_text(path: Path) -> str:
    result = subprocess.run(["pdftotext", str(path), "-"], check=True, capture_output=True)
    return result.stdout.decode("utf-8", errors="replace")


def _case_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame[["record_id", "y_SCLC"]].copy()
    if labels.empty or labels.isna().any().any() or not labels["record_id"].astype(str).is_unique:
        raise RuntimeError("Locked prediction cases must have nonmissing unique IDs and outcomes")
    labels["record_id"] = labels["record_id"].astype(str)
    labels["y_SCLC"] = pd.to_numeric(labels["y_SCLC"], errors="raise")
    if not labels["y_SCLC"].isin([0, 1]).all():
        raise RuntimeError("Locked prediction outcomes must be binary")
    return labels.sort_values("record_id", kind="mergesort").reset_index(drop=True)


def _same_cases(frame: pd.DataFrame, reference: pd.DataFrame) -> bool:
    try:
        return _case_labels(frame).equals(reference)
    except (ValueError, RuntimeError):
        return False


def _locked_primary_cases() -> dict[str, pd.DataFrame]:
    support = data_root() / "fusion_results"
    a_fusions = pd.read_csv(support / "A_fusions_CV_holdout" / "fusion_predictions_A_holdout_clean.csv")
    bc_fusions = pd.read_csv(support / "three_fusions_BC" / "fusion_predictions_B_C.csv")
    result = {}
    for dataset, source, model_column in [(A_EVAL, a_fusions, "fusion"), (B_EXT, bc_fusions, "model"), (C_EXT, bc_fusions, "model")]:
        result[dataset] = _case_labels(source.loc[source["dataset"].eq(dataset) & source[model_column].eq(LAYER_MODEL)])
        for model, group in source.loc[source["dataset"].eq(dataset)].groupby(model_column):
            if not _same_cases(group, result[dataset]):
                raise RuntimeError(f"Frozen prediction model {dataset}/{model} differs from the locked primary case set")
    return result


def main() -> None:
    rows: list[dict[str, object]] = []
    reference_cases = _locked_primary_cases()
    expected_n = {dataset: len(frame) for dataset, frame in reference_cases.items()}
    expected_events = {dataset: int(frame["y_SCLC"].sum()) for dataset, frame in reference_cases.items()}
    native = pd.read_csv(data_root() / "fusion_results" / "native_predictions_all_eval.csv")
    a = native.loc[native["dataset"].eq(A_EVAL) & native["model"].isin(SINGLE)].copy()
    counts = a.groupby("model")["record_id"].nunique()
    rows.append(check("A-evaluation individual-model scope", set(counts.index) == SINGLE and all(_same_cases(group, reference_cases[A_EVAL]) for _, group in a.groupby("model")), f"models={len(counts)}; unique-record counts={sorted(counts.unique().tolist())}"))
    rbf = a.loc[a["model"].eq("rbf_svm")]
    rows.append(check("RBF score-scale boundary", rbf["score_kind"].eq("decision_function").all() and rbf["native_probability"].isna().all(), f"rows={len(rbf)}; score_kind={sorted(rbf['score_kind'].dropna().unique().tolist())}"))


    dca_models = SINGLE - {"rbf_svm"}
    individual_dca = native.loc[
        native["dataset"].eq(A_EVAL) & native["model"].isin(dca_models)
    ].copy()
    individual_dca_counts = individual_dca.groupby("model")["record_id"].nunique()
    dca_labels = individual_dca.loc[
        individual_dca["model"].eq("logistic_regression"), ["record_id", "y_SCLC"]
    ].drop_duplicates()
    individual_dca_ok = (
        set(individual_dca_counts.index) == dca_models
        and individual_dca_counts.eq(expected_n[A_EVAL]).all()
        and len(individual_dca) == len(dca_models) * expected_n[A_EVAL]
        and individual_dca["score_kind"].eq("probability").all()
        and individual_dca["native_probability"].notna().all()
        and len(dca_labels) == expected_n[A_EVAL]
        and all(_same_cases(group, reference_cases[A_EVAL]) for _, group in individual_dca.groupby("model"))
    )
    rows.append(check(
        "individual-model DCA source scope",
        individual_dca_ok,
        f"models={len(individual_dca_counts)}; rows={len(individual_dca)}; "
        f"records/model={sorted(individual_dca_counts.unique().tolist())}; "
        f"events={int(dca_labels['y_SCLC'].sum())}; score_kind={sorted(individual_dca['score_kind'].dropna().unique().tolist())}",
    ))


    layer_archive = data_root() / "fusion_results"
    layer_predictions = pd.read_csv(
        layer_archive / "data" / "weighted_voting_layer_predictions.csv",
        usecols=["dataset", "layer", "model", "record_id", "y_SCLC", "probability"],
    )
    layer_predictions = layer_predictions.loc[
        layer_predictions["dataset"].isin(LAYER_DATASETS)
        & layer_predictions["layer"].isin(LAYER_SET)
        & layer_predictions["model"].eq(LAYER_MODEL)
    ].copy()
    layer_counts = layer_predictions.groupby(["dataset", "layer"])["record_id"].nunique()
    layer_inventory_ok = len(layer_counts) == 9 and all(
        layer_counts.get((dataset, layer), 0) == expected_n[dataset]
        for dataset in LAYER_DATASETS for layer in LAYER_SET
    )
    layer_probability_ok = layer_predictions["probability"].notna().all() and pd.to_numeric(layer_predictions["probability"], errors="coerce").between(0, 1).all()
    label_events_ok = all(
        _same_cases(layer_predictions.loc[layer_predictions["dataset"].eq(dataset) & layer_predictions["layer"].eq(layer)], reference_cases[dataset])
        for dataset in LAYER_DATASETS for layer in LAYER_SET
    )
    layer_metrics = pd.read_csv(layer_archive / "tables" / "weighted_voting_layer_metrics_ci.csv")
    layer_metrics = layer_metrics.loc[
        layer_metrics["dataset"].isin(LAYER_DATASETS)
        & layer_metrics["layer"].isin(LAYER_SET)
        & layer_metrics["model"].eq(LAYER_MODEL)
    ].copy()
    metric_columns = ["auc", "auc_ci_low", "auc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high", "brier", "brier_ci_low", "brier_ci_high"]
    layer_metric_ok = len(layer_metrics) == 9 and layer_metrics[metric_columns].notna().all().all()
    layer_calibration = pd.read_csv(layer_archive / "data" / "weighted_voting_layer_calibration_plot_data.csv")
    layer_calibration = layer_calibration.loc[
        layer_calibration["dataset"].isin(LAYER_DATASETS) & layer_calibration["layer"].isin(LAYER_SET)
    ]
    calibration_types = layer_calibration.groupby(["dataset", "layer"])["series_type"].agg(set)
    layer_calibration_ok = len(calibration_types) == 9 and calibration_types.map(lambda values: {"LOWESS", "equal_frequency_bin"}.issubset(values)).all()
    rows.append(check(
        "ordered separate layer-comparison source scope",
        layer_inventory_ok and layer_probability_ok and label_events_ok and layer_metric_ok and layer_calibration_ok,
        f"prediction_rows={len(layer_predictions)}; metric_rows={len(layer_metrics)}; calibration_groups={len(calibration_types)}; "
        f"n={expected_n}; events={expected_events}",
    ))

    metrics = pd.read_csv(data_root() / "fusion_results" / "tables" / "model_metrics_with_ci.csv")
    fusion_metrics = metrics.loc[metrics["dataset"].isin([A_EVAL, B_EXT, C_EXT]) & metrics["model"].isin(FUSIONS)]
    by_dataset = fusion_metrics.groupby("dataset")["model"].agg(lambda x: set(x))
    rows.append(check("frozen fusion validation scope", len(fusion_metrics) == 9 and all(x == FUSIONS for x in by_dataset), f"rows={len(fusion_metrics)}; datasets={by_dataset.index.tolist()}"))

    dca = pd.read_csv(data_root() / "fusion_results" / "data" / "dca_curve_points.csv")
    dca_subset = dca.loc[dca["dataset"].isin([A_EVAL, B_EXT, C_EXT]) & dca["model"].isin(FUSIONS)]
    grid_counts = dca_subset.groupby(["dataset", "model"]).size()
    rows.append(check("DCA fixed dense grids", len(grid_counts) == 9 and grid_counts.eq(160).all(), f"curve sizes={sorted(grid_counts.unique().tolist())}; no Youden data series requested"))

    delong = pd.read_csv(data_root() / "fusion_results" / "tables" / "delong_holm_vs_weighted_voting.csv")
    ds = delong.loc[delong["dataset"].isin([A_EVAL, B_EXT, C_EXT])]
    rows.append(check("DeLong/Holm v4 scope", len(ds) == 15 and set(ds["dataset"]) == {A_EVAL, B_EXT, C_EXT}, f"rows={len(ds)}; datasets={sorted(ds['dataset'].unique().tolist())}"))

    s2 = pd.read_csv(ROOT / "tables" / "TableS8_development_cv_selection_evidence.csv")
    rows.append(check("development CV retained only as selection table", len(s2) == 3 and set(s2["model_code"]) == FUSIONS and s2["dataset"].eq("A development (10-fold CV)").all(), f"rows={len(s2)}; models={sorted(s2['model_code'].tolist())}"))

    layer_dir = data_root() / "fusion_results" / "tables"
    layer_metrics = pd.read_csv(layer_dir / "weighted_voting_layer_metrics_ci.csv")
    layer_increment = pd.read_csv(layer_dir / "weighted_voting_layer_increment_ci.csv")
    layer_delong = pd.read_csv(layer_dir / "weighted_voting_layer_delong_holm.csv")
    selected_layers = {"I", "I_M", "I_M_N"}
    selected_metric_rows = layer_metrics.loc[layer_metrics["dataset"].eq("A_dev_cv_clean") & layer_metrics["layer"].isin(selected_layers)]
    selected_increment_rows = layer_increment.loc[layer_increment["dataset"].eq("A_dev_cv_clean") & layer_increment["metric"].isin(["auc", "auprc"])]
    selected_delong_rows = layer_delong.loc[layer_delong["dataset"].eq("A_dev_cv_clean")]
    rows.append(check("layer-selection evidence source", len(selected_metric_rows) == 3 and len(selected_increment_rows) == 6 and len(selected_delong_rows) == 3, f"metric rows={len(selected_metric_rows)}; AUC/AUPRC increment rows={len(selected_increment_rows)}; DeLong rows={len(selected_delong_rows)}"))

    s3 = pd.read_csv(ROOT / "tables" / "TableS7_fusion_membership_and_weights.csv")
    weight_col = "weighted_voting_normalized_weight"
    members = s3.loc[s3["record_type"].eq("base_model_member")].copy()
    weights = pd.to_numeric(members[weight_col], errors="raise")
    rbf_rows = s3.loc[s3["model_code"].eq("rbf_svm")]
    rows.append(check("weighted-voting membership", len(members) == 13 and not members["model_code"].eq("rbf_svm").any() and len(rbf_rows) == 1 and rbf_rows["record_type"].eq("excluded_base_model").all() and abs(float(weights.sum()) - 1.0) < 1e-10, f"members={len(members)}; weight_sum={weights.sum():.12f}; RBF rows={len(rbf_rows)}"))

    forbidden = ("20%", "holdout", "top-x", "top x", "stacking (primary)")
    pdfs = sorted((ROOT / "figures").rglob("*.pdf"))
    bad = {}
    for path in pdfs:
        content = pdf_text(path).lower()
        hits = [token for token in forbidden if token in content]
        if hits:
            bad[str(path.relative_to(ROOT))] = hits
    rows.append(check("manuscript-facing wording", not bad, f"figures_checked={len(pdfs)}; forbidden_hits={bad}"))

    destination = QC / "data_scope_qc.csv"
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "evidence"])
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    report = ["# Data and scope QC", "", f"- Independent checks: {len(rows) - len(failures)}/{len(rows)} passed.", "- This audit reads frozen source records only; it does not rerun training or alter a metric.", "", "## Checks", ""]
    report += [f"- **{row['status']}** — {row['check']}: {row['evidence']}" for row in rows]
    (QC / "data_scope_qc.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"checks={len(rows)} failures={len(failures)}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
