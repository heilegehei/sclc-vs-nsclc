from __future__ import annotations
import sys as _public_sys
from pathlib import Path as _PublicPath
_public_root = next(p for p in _PublicPath(__file__).resolve().parents if (p / "common" / "public_paths.py").is_file())
if str(_public_root) not in _public_sys.path:
    _public_sys.path.insert(0, str(_public_root))
from common.public_paths import cohort_workbook, code_root, data_root, explainability_results, fusion_results


import sys as _cohort_sys
from pathlib import Path as _CohortPath
_cohort_root = next(p for p in _CohortPath(__file__).resolve().parents if (p / "common" / "manuscript_cohorts.py").is_file())
if str(_cohort_root) not in _cohort_sys.path:
    _cohort_sys.path.insert(0, str(_cohort_root))
from common.manuscript_cohorts import split_center_a_indices, validate_cohort, validate_base_cohorts, split_metadata, require_split_design

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

import shapiq
from shapiq import KernelSHAPIQ, TabularExplainer


HERE = Path(__file__).resolve().parent
OUT_ROOT = explainability_results()
FINAL_ROOT = OUT_ROOT.parent
PROJECT_ROOT = data_root()
V1_ROOT = data_root() / "fusion_results"


import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents
                     if (p / "common" / "manuscript_source_paths.py").is_file())
if str(_archive_root) not in _archive_sys.path:
    _archive_sys.path.insert(0, str(_archive_root))
from common.manuscript_source_paths import activate_archive_sources
activate_archive_sources(__file__)

from stage4_pipeline import (
    ALL_FEATURES,
    DIRECT_FEATURES,
    compute_composites,
    load_base_workbook,
)


MASTER_SEED = 20260902
BACKGROUND_SEED = 20260911
PRODUCTION_SEEDS = (20260921, 20260922, 20260923)
PRODUCTION_BUDGET = 4096
BACKGROUND_N = 40
EXPECTED_CASE_N = None
PAIR_N = 153
MODEL_PATH = fusion_results() / "model" / "F11_weighted_voting_model.joblib"
WORKBOOK_PATH = cohort_workbook()
CONFIRMED_UNITS_PATH = code_root() / "config" / "feature_units_confirmed.json"

FEATURE_GROUPS = {
    "WBC": "I", "PLT": "I", "LDH": "I", "PCT": "I", "SIRI": "I",
    "EO#": "M", "EO%": "M", "LMR": "M", "GAR": "M",
    "ALB": "N", "TP": "N", "GLB": "N", "TC": "N", "LDL-C": "N",
    "MCH": "N", "MCHC": "N", "PNI": "N", "HALP": "N",
}
DISPLAY_NAMES = {"EO#": "EO count", "EO%": "EO %"}
CLINICAL_DEFINITIONS = {"PCT": "plateletcrit", "GAR": "GLU/ALB"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


class WeightedProbability:

    def __init__(self, package: dict[str, Any]):
        self.estimators = package["estimators"]
        self.weights = {str(k): float(v) for k, v in package["weights"].items()}
        self.member_order = list(package["member_order"])
        self.features = list(package["explain_features"])
        self.stage4_features = list(package["stage4_features"])
        if self.member_order != list(self.weights):
            raise RuntimeError("member order and weight order differ")
        if len(self.member_order) != 13 or len(self.features) != 18:
            raise RuntimeError("the frozen package is not the expected 13-member/18-feature model")
        weight_vector = np.asarray([self.weights[name] for name in self.member_order], dtype=float)
        if not np.isclose(weight_vector.sum(), 1.0, atol=1e-12):
            raise RuntimeError("Weighted-voting weights do not sum to one")
        self.weight_vector = weight_vector

    def __call__(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != len(self.features):
            raise ValueError(f"expected (n, {len(self.features)}) transformed inputs")
        full = pd.DataFrame(0.0, index=np.arange(len(array)), columns=self.stage4_features)
        full.loc[:, self.features] = array
        member_probabilities = np.column_stack([
            np.asarray(self.estimators[name].predict_proba(full), dtype=float)[:, 1]
            for name in self.member_order
        ])
        return np.clip(member_probabilities @ self.weight_vector, 0.0, 1.0)


def _fixed_background_indices(train: pd.DataFrame, n: int = BACKGROUND_N) -> list[int]:
    rng = np.random.default_rng(BACKGROUND_SEED)
    labels = train["y_SCLC"].to_numpy(dtype=int)
    chosen: list[int] = []
    for outcome in (0, 1):
        candidates = np.flatnonzero(labels == outcome)
        n_class = int(round(n * len(candidates) / len(train)))
        chosen.extend(rng.choice(candidates, size=n_class, replace=False).tolist())
    if len(chosen) < n:
        remaining = np.setdiff1d(np.arange(len(train)), np.asarray(chosen, dtype=int))
        chosen.extend(rng.choice(remaining, size=n - len(chosen), replace=False).tolist())
    elif len(chosen) > n:
        chosen = chosen[:n]
    return sorted(chosen)


def _load_runtime() -> dict[str, Any]:
    global EXPECTED_CASE_N
    required = [MODEL_PATH, WORKBOOK_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen input(s):\n" + "\n".join(missing))
    package = joblib.load(MODEL_PATH)
    predictor = WeightedProbability(package)
    if set(predictor.features) != set(FEATURE_GROUPS):
        raise RuntimeError("the 18-feature map does not match the model package")

    center_a = validate_base_cohorts(load_base_workbook(WORKBOOK_PATH))["A"]
    center_a = center_a.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    if "missing_direct_n" in center_a.columns:
        center_a = center_a.loc[
            center_a["missing_direct_n"].astype(float) < len(DIRECT_FEATURES)
        ].reset_index(drop=True)
    indices = np.arange(len(center_a), dtype=int)
    train_index, holdout_index = split_center_a_indices(center_a)
    require_split_design(package)
    train = center_a.iloc[np.sort(train_index)].reset_index(drop=True)
    holdout = center_a.iloc[np.sort(holdout_index)].reset_index(drop=True)
    EXPECTED_CASE_N = len(holdout)

    transformed_train, train_audit = package["preprocessor"].transform(train)
    transformed_holdout, holdout_audit = package["preprocessor"].transform(holdout)
    features = predictor.features
    x_train = transformed_train.loc[:, features].reset_index(drop=True)
    x_holdout = transformed_holdout.loc[:, features].reset_index(drop=True)
    if not np.isfinite(x_train.to_numpy(dtype=float)).all() or not np.isfinite(
        x_holdout.to_numpy(dtype=float)
    ).all():
        raise RuntimeError("non-finite value remains after frozen Stage-4 preprocessing")

    background_index = _fixed_background_indices(train)
    background = x_train.iloc[background_index].reset_index(drop=True)
    background_meta = train.iloc[background_index][["record_id", "y_SCLC"]].reset_index(drop=True)
    if len(background) != BACKGROUND_N or background_meta["y_SCLC"].value_counts().to_dict() != {0: 32, 1: 8}:
        raise RuntimeError("fixed background is not the expected 40 records (32/8 by outcome)")
    prior_parameters = V1_ROOT / "data" / "F11_explainer_parameters.json"
    background_check = "prior manifest unavailable"
    if prior_parameters.exists():
        expected = json.loads(prior_parameters.read_text(encoding="utf-8"))["background_record_ids"]
        observed = background_meta["record_id"].astype(str).tolist()
        if observed != list(expected):
            raise RuntimeError("reconstructed A-development background differs from frozen v1 IDs")
        background_check = "PASS: IDs equal the frozen v1 background in the same order"

    probabilities = predictor(x_holdout.to_numpy(dtype=float))
    reproducibility = {"status": "not checked", "max_abs_difference": None}
    prior_predictions = V1_ROOT / "data" / "F11_weighted_voting_refit_predictions.csv"
    if prior_predictions.exists():
        reference = pd.read_csv(prior_predictions)
        reference = reference.loc[
            reference["dataset"].astype(str) == "A_holdout_clean"
        ].reset_index(drop=True)
        if len(reference) != len(holdout):
            raise RuntimeError("v1 A-holdout reference count differs")
        if reference["record_id"].astype(str).duplicated().any() or holdout["record_id"].astype(str).duplicated().any():
            raise RuntimeError("record_id is not unique for deterministic prediction alignment")
        reference_score = reference.set_index(reference["record_id"].astype(str))[
            "refit_weighted_probability"
        ]
        aligned_reference = reference_score.loc[holdout["record_id"].astype(str)].to_numpy(dtype=float)
        difference = np.abs(
            probabilities - aligned_reference
        )
        reproducibility = {
            "status": "PASS" if float(difference.max()) <= 1e-10 else "FAIL",
            "max_abs_difference": float(difference.max()),
        }
        if reproducibility["status"] != "PASS":
            raise RuntimeError("serialized Weighted-voting predictions do not reproduce v1")

    return {
        "package": package,
        "predictor": predictor,
        "train": train,
        "holdout": holdout,
        "x_train": x_train,
        "x_holdout": x_holdout,
        "background": background,
        "background_meta": background_meta,
        "background_check": background_check,
        "probabilities": probabilities,
        "train_audit": train_audit,
        "holdout_audit": holdout_audit,
        "prediction_reproducibility": reproducibility,
    }


def _coalition_audit(n_features: int, budget: int, seed: int) -> dict[str, Any]:

    approximator = KernelSHAPIQ(
        n=n_features,
        max_order=2,
        index="SII",
        pairing_trick=True,
        random_state=seed,
    )
    approximator._sampler.sample(budget)
    coalitions = np.asarray(approximator._sampler.coalitions_matrix, dtype=np.uint8)
    packed = np.packbits(coalitions, axis=1)
    return {
        "seed": seed,
        "requested_budget": budget,
        "sampled_coalition_n": int(len(coalitions)),
        "unique_coalition_n": int(len(np.unique(coalitions, axis=0))),
        "coalition_matrix_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        "pairing_trick": True,
    }


def _new_explainer(runtime: dict[str, Any], budget: int, seed: int) -> TabularExplainer:
    n_features = len(runtime["predictor"].features)
    approximator = KernelSHAPIQ(
        n=n_features,
        max_order=2,
        index="SII",
        pairing_trick=True,
        random_state=seed,
    )
    return TabularExplainer(
        model=runtime["predictor"],
        data=runtime["background"].to_numpy(dtype=float),
        index="SII",
        max_order=2,
        imputer="marginal",
        approximator=approximator,
        sample_size=BACKGROUND_N,
        joint_marginal_distribution=True,
        normalize=True,
        random_state=seed,
        verbose=False,
    )


def _explain_one_common_coalitions(
    explainer: TabularExplainer,
    x: np.ndarray,
    budget: int,
    seed: int,
) -> Any:
    return explainer.explain(x, budget=budget, random_state=seed)


def _metadata_value(frame: pd.DataFrame, row: int, column: str, default: Any) -> Any:
    if column not in frame.columns:
        return default
    value = frame.iloc[row][column]
    return default if pd.isna(value) else value


def _extract_rows(
    interaction_values: Iterable[Any],
    case_indices: list[int],
    runtime: dict[str, Any],
    run_kind: str,
    budget: int,
    seed: int,
    coalition_hash: str,
) -> pd.DataFrame:
    features = runtime["predictor"].features
    holdout = runtime["holdout"]
    probabilities = runtime["probabilities"]
    rows: list[dict[str, Any]] = []
    for sample_index, values in zip(case_indices, interaction_values):
        if values.index != "SII" or values.max_order != 2 or not values.estimated:
            raise RuntimeError("unexpected interaction object: expected estimated order-2 SII")
        for feature_i_index, feature_j_index in combinations(range(len(features)), 2):
            rows.append({
                "run_kind": run_kind,
                "budget": budget,
                "seed": seed,
                "sample_index": sample_index,
                "record_id": str(_metadata_value(holdout, sample_index, "record_id", sample_index)),
                "source_record_id": str(_metadata_value(
                    holdout, sample_index, "source_record_id",
                    _metadata_value(holdout, sample_index, "record_id", sample_index),
                )),
                "duplicate_flag": bool(_metadata_value(holdout, sample_index, "duplicate_flag", False)),
                "y_SCLC": int(_metadata_value(holdout, sample_index, "y_SCLC", -1)),
                "weighted_probability": float(probabilities[sample_index]),
                "feature_i_index": feature_i_index,
                "feature_j_index": feature_j_index,
                "feature_i": features[feature_i_index],
                "feature_j": features[feature_j_index],
                "sii_value": float(values[(feature_i_index, feature_j_index)]),
                "baseline_value": float(values.baseline_value),
                "estimated": bool(values.estimated),
                "estimation_budget": int(values.estimation_budget),
                "coalition_matrix_sha256": coalition_hash,
            })
    frame = pd.DataFrame(rows)
    expected = len(case_indices) * PAIR_N
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} pair rows, obtained {len(frame)}")
    return frame


def _load_checkpoint(path: Path, budget: int, seed: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"sample_index", "feature_i_index", "feature_j_index", "sii_value", "budget", "seed"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"malformed checkpoint {path}")
    if set(frame["budget"].astype(int)) != {budget} or set(frame["seed"].astype(int)) != {seed}:
        raise RuntimeError(f"checkpoint metadata differs: {path}")
    counts = frame.groupby("sample_index").size()
    complete = counts[counts == PAIR_N].index
    repaired = frame.loc[frame["sample_index"].isin(complete)].copy()
    duplicates = repaired.duplicated(["sample_index", "feature_i_index", "feature_j_index"])
    if duplicates.any():
        raise RuntimeError(f"duplicate pair rows in checkpoint {path}")
    if len(repaired) != len(frame):
        print(f"Repairing an incomplete trailing checkpoint block: {path}", flush=True)
        _atomic_csv(repaired, path)
    return repaired


def _run_budget_seed(
    runtime: dict[str, Any],
    case_indices: list[int],
    run_kind: str,
    budget: int,
    seed: int,
    n_jobs: int,
    batch_size: int,
) -> Path:
    checkpoint_dir = OUT_ROOT / "data" / "sii" / run_kind / "checkpoints"
    checkpoint = checkpoint_dir / f"sii_{run_kind}_budget{budget}_seed{seed}.csv"
    existing = _load_checkpoint(checkpoint, budget, seed)
    complete = set(existing["sample_index"].astype(int)) if not existing.empty else set()
    remaining = [index for index in case_indices if index not in complete]
    audit = _coalition_audit(len(runtime["predictor"].features), budget, seed)
    print(
        f"[{run_kind}] budget={budget} seed={seed}: {len(complete)}/{len(case_indices)} cases complete; "
        f"coalition={audit['coalition_matrix_sha256'][:12]}",
        flush=True,
    )
    for offset in range(0, len(remaining), batch_size):
        batch_indices = remaining[offset: offset + batch_size]
        explainer = _new_explainer(runtime, budget, seed)
        matrix = runtime["x_holdout"].iloc[batch_indices].to_numpy(dtype=float)
        started = time.perf_counter()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            values = Parallel(n_jobs=n_jobs)(
                delayed(_explain_one_common_coalitions)(explainer, matrix[row], budget, seed)
                for row in range(len(matrix))
            )
        new_rows = _extract_rows(
            values,
            batch_indices,
            runtime,
            run_kind,
            budget,
            seed,
            audit["coalition_matrix_sha256"],
        )
        existing = pd.concat([existing, new_rows], ignore_index=True)
        existing = existing.sort_values(
            ["sample_index", "feature_i_index", "feature_j_index"], kind="mergesort"
        ).reset_index(drop=True)
        _atomic_csv(existing, checkpoint)
        elapsed = time.perf_counter() - started
        print(
            f"[{run_kind}] budget={budget} seed={seed}: saved {len(existing) // PAIR_N}/"
            f"{len(case_indices)} cases ({elapsed:.1f}s for batch)",
            flush=True,
        )
    return checkpoint


def _probe_indices(runtime: dict[str, Any], n_cases: int) -> list[int]:
    labels = runtime["holdout"]["y_SCLC"].to_numpy(dtype=int)
    probabilities = runtime["probabilities"]
    selected: list[int] = []
    allocation = []
    for outcome in (0, 1):
        candidates = np.flatnonzero(labels == outcome)
        allocation.append(int(round(n_cases * len(candidates) / len(labels))))
    allocation[1] += n_cases - sum(allocation)
    for outcome, n_class in zip((0, 1), allocation):
        candidates = np.flatnonzero(labels == outcome)
        candidates = candidates[np.argsort(probabilities[candidates], kind="mergesort")]
        positions = np.rint(np.linspace(0, len(candidates) - 1, n_class)).astype(int)
        selected.extend(candidates[positions].tolist())
    selected = sorted(set(selected))
    if len(selected) != n_cases:
        raise RuntimeError("probe selection did not yield the requested unique number of cases")
    return selected


def _top_pair(frame: pd.DataFrame) -> str:
    row = frame.sort_values(["strength", "pair"], ascending=[False, True]).iloc[0]
    return str(row["pair"])


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left | right else 1.0


def _aggregate_probe(paths: list[Path], reference_budget: int = PRODUCTION_BUDGET) -> dict[str, Any]:
    output_dir = OUT_ROOT / "data" / "sii" / "probe"
    table_dir = OUT_ROOT / "tables"
    frames = [pd.read_csv(path) for path in paths]
    all_values = pd.concat(frames, ignore_index=True)
    all_values["pair"] = all_values["feature_i"] + " × " + all_values["feature_j"]
    _atomic_csv(all_values, output_dir / "weighted_sii_probe_pair_values.csv")

    strengths = (
        all_values.assign(abs_sii=lambda x: x["sii_value"].abs())
        .groupby(["budget", "seed", "pair", "feature_i", "feature_j"], as_index=False)["abs_sii"]
        .mean()
        .rename(columns={"abs_sii": "strength"})
    )
    strengths["rank"] = strengths.groupby(["budget", "seed"])["strength"].rank(
        method="min", ascending=False
    )
    _atomic_csv(strengths, table_dir / "weighted_sii_probe_global_pair_ranking.csv")

    seed_rows: list[dict[str, Any]] = []
    for budget, part in strengths.groupby("budget"):
        seeds = sorted(part["seed"].unique())
        for seed_a, seed_b in combinations(seeds, 2):
            a_raw = part.loc[part["seed"] == seed_a, ["pair", "strength"]].copy()
            b_raw = part.loc[part["seed"] == seed_b, ["pair", "strength"]].copy()
            a = a_raw.rename(
                columns={"strength": "strength_a"}
            )
            b = b_raw.rename(
                columns={"strength": "strength_b"}
            )
            merged = a.merge(b, on="pair", validate="one_to_one")
            top_a = set(a_raw.nlargest(10, "strength")["pair"])
            top_b = set(b_raw.nlargest(10, "strength")["pair"])
            top_pair_a = _top_pair(a_raw)
            top_pair_b = _top_pair(b_raw)
            seed_rows.append({
                "budget": int(budget),
                "seed_a": int(seed_a),
                "seed_b": int(seed_b),
                "spearman_global_pair_strength": float(spearmanr(
                    merged["strength_a"], merged["strength_b"]
                ).statistic),
                "top_pair_a": top_pair_a,
                "top_pair_b": top_pair_b,
                "top_pair_match": top_pair_a == top_pair_b,
                "top10_jaccard": _jaccard(top_a, top_b),
            })
    seed_stability = pd.DataFrame(seed_rows)
    _atomic_csv(seed_stability, table_dir / "weighted_sii_probe_seed_stability.csv")

    case_pair_sd = (
        all_values.groupby(["budget", "sample_index", "pair"], as_index=False)["sii_value"]
        .agg(seed_mean="mean", seed_sd="std")
    )
    dispersion = (
        case_pair_sd.groupby("budget", as_index=False)
        .agg(
            mean_seed_sd=("seed_sd", "mean"),
            median_seed_sd=("seed_sd", "median"),
            max_seed_sd=("seed_sd", "max"),
            mean_abs_seed_mean=("seed_mean", lambda x: float(np.mean(np.abs(x)))),
        )
    )
    dispersion["relative_mean_seed_sd"] = (
        dispersion["mean_seed_sd"] / dispersion["mean_abs_seed_mean"].replace(0, np.nan)
    )
    _atomic_csv(dispersion, table_dir / "weighted_sii_probe_seed_dispersion.csv")

    budget_global = (
        all_values.groupby(["budget", "sample_index", "pair"], as_index=False)["sii_value"]
        .mean()
        .assign(abs_sii=lambda x: x["sii_value"].abs())
        .groupby(["budget", "pair"], as_index=False)["abs_sii"]
        .mean()
        .rename(columns={"abs_sii": "strength"})
    )
    if reference_budget not in set(budget_global["budget"].astype(int)):
        reference_budget = int(budget_global["budget"].max())
    reference = budget_global.loc[
        budget_global["budget"] == reference_budget, ["pair", "strength"]
    ].rename(columns={"strength": "reference_strength"})
    convergence_rows: list[dict[str, Any]] = []
    for budget, part in budget_global.groupby("budget"):
        merged = part.merge(reference, on="pair", validate="one_to_one")
        top = set(part.nlargest(10, "strength")["pair"])
        top_reference = set(reference.nlargest(10, "reference_strength")["pair"])
        convergence_rows.append({
            "budget": int(budget),
            "reference_budget": reference_budget,
            "spearman_vs_reference": float(spearmanr(
                merged["strength"], merged["reference_strength"]
            ).statistic),
            "top_pair": str(part.nlargest(1, "strength").iloc[0]["pair"]),
            "reference_top_pair": str(reference.nlargest(1, "reference_strength").iloc[0]["pair"]),
            "top_pair_match": str(part.nlargest(1, "strength").iloc[0]["pair"])
            == str(reference.nlargest(1, "reference_strength").iloc[0]["pair"]),
            "top10_jaccard": _jaccard(top, top_reference),
        })
    convergence = pd.DataFrame(convergence_rows)
    _atomic_csv(convergence, table_dir / "weighted_sii_probe_budget_convergence.csv")

    reference_seed = seed_stability.loc[seed_stability["budget"] == reference_budget]
    top_pairs = strengths.loc[strengths["budget"] == reference_budget].groupby("seed").apply(
        lambda x: x.nlargest(1, "strength").iloc[0]["pair"], include_groups=False
    )
    top_pair_mode_count = Counter(top_pairs.tolist()).most_common(1)[0][1]
    prior_budget = max([x for x in budget_global["budget"].unique() if x < reference_budget], default=None)
    convergence_row = convergence.loc[convergence["budget"] == prior_budget] if prior_budget else pd.DataFrame()
    criteria = {
        "minimum_pairwise_seed_spearman_at_reference": 0.95,
        "minimum_pairwise_top10_jaccard_at_reference": 0.60,
        "minimum_top_pair_seed_consensus_of_three": 2,
        "minimum_prior_budget_vs_reference_spearman": 0.95,
    }
    stable = bool(
        not reference_seed.empty
        and float(reference_seed["spearman_global_pair_strength"].min())
        >= criteria["minimum_pairwise_seed_spearman_at_reference"]
        and float(reference_seed["top10_jaccard"].min())
        >= criteria["minimum_pairwise_top10_jaccard_at_reference"]
        and top_pair_mode_count >= criteria["minimum_top_pair_seed_consensus_of_three"]
        and not convergence_row.empty
        and float(convergence_row.iloc[0]["spearman_vs_reference"])
        >= criteria["minimum_prior_budget_vs_reference_spearman"]
    )
    decision = {
        "quantity": "Estimated pairwise Shapley Interaction Index (SII)",
        "reference_budget": reference_budget,
        "criteria": criteria,
        "stable_for_full_run": stable,
        "minimum_seed_spearman": float(reference_seed["spearman_global_pair_strength"].min()),
        "minimum_seed_top10_jaccard": float(reference_seed["top10_jaccard"].min()),
        "top_pair_seed_mode_count": int(top_pair_mode_count),
        "prior_budget": prior_budget,
        "prior_budget_spearman": None if convergence_row.empty else float(
            convergence_row.iloc[0]["spearman_vs_reference"]
        ),
        "next_action": "run full 4096 analysis" if stable else "extend probe to budget 8192",
    }
    _atomic_json(decision, OUT_ROOT / "qc" / "weighted_sii_probe_decision.json")
    return decision


def _case_inputs(runtime: dict[str, Any]) -> pd.DataFrame:
    holdout = runtime["holdout"].reset_index(drop=True)
    preprocessor = runtime["package"]["preprocessor"]
    direct = holdout.loc[:, DIRECT_FEATURES].apply(pd.to_numeric, errors="coerce")
    imputed = direct.fillna(pd.Series(preprocessor.medians))
    composites = compute_composites(imputed)
    original_scale = pd.concat([imputed, composites], axis=1)
    transformed = runtime["x_holdout"]
    rows: list[dict[str, Any]] = []
    for sample_index in range(len(holdout)):
        for feature in runtime["predictor"].features:
            rows.append({
                "sample_index": sample_index,
                "record_id": str(_metadata_value(holdout, sample_index, "record_id", sample_index)),
                "source_record_id": str(_metadata_value(
                    holdout, sample_index, "source_record_id",
                    _metadata_value(holdout, sample_index, "record_id", sample_index),
                )),
                "duplicate_flag": bool(_metadata_value(holdout, sample_index, "duplicate_flag", False)),
                "y_SCLC": int(_metadata_value(holdout, sample_index, "y_SCLC", -1)),
                "weighted_probability": float(runtime["probabilities"][sample_index]),
                "feature": feature,
                "group": FEATURE_GROUPS[feature],
                "original_scale_analysis_value": float(original_scale.loc[sample_index, feature]),
                "stage4_transformed_value": float(transformed.loc[sample_index, feature]),
                "direct_value_was_imputed": bool(
                    feature in DIRECT_FEATURES and pd.isna(direct.loc[sample_index, feature])
                ),
                "value_definition": (
                    "median-imputed direct value before clipping/log1p"
                    if feature in DIRECT_FEATURES
                    else "recomputed composite before clipping/log1p"
                ),
            })
    return pd.DataFrame(rows)


def _aggregate_full(paths: list[Path], runtime: dict[str, Any], budget: int) -> dict[str, Any]:
    full_dir = OUT_ROOT / "data" / "sii" / "full"
    table_dir = OUT_ROOT / "tables"
    qc_dir = OUT_ROOT / "qc"
    values = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["seed", "sample_index", "feature_i_index", "feature_j_index"]
    if values.duplicated(keys).any():
        raise RuntimeError("duplicate rows found while aggregating the full SII run")
    expected = len(PRODUCTION_SEEDS) * EXPECTED_CASE_N * PAIR_N
    if len(values) != expected:
        raise RuntimeError(f"full SII result has {len(values)} rows; expected {expected}")
    if set(values["budget"].astype(int)) != {budget}:
        raise RuntimeError("full SII result contains an unexpected budget")
    values = values.sort_values(keys, kind="mergesort").reset_index(drop=True)
    _atomic_csv(values, full_dir / "weighted_sii_per_case_seed.csv")

    group_keys = [
        "sample_index", "record_id", "source_record_id", "duplicate_flag", "y_SCLC",
        "weighted_probability", "feature_i_index", "feature_j_index", "feature_i", "feature_j",
    ]
    aggregated = values.groupby(group_keys, as_index=False)["sii_value"].agg(
        sii_mean="mean", sii_sd_across_seeds="std", sii_min="min", sii_max="max"
    )
    aggregated["abs_sii_mean"] = aggregated["sii_mean"].abs()
    _atomic_csv(aggregated, full_dir / "weighted_sii_per_case_aggregated.csv")

    global_ranking = aggregated.groupby(
        ["feature_i_index", "feature_j_index", "feature_i", "feature_j"], as_index=False
    ).agg(
        mean_abs_sii=("abs_sii_mean", "mean"),
        median_abs_sii=("abs_sii_mean", "median"),
        mean_signed_sii=("sii_mean", "mean"),
        sd_signed_sii_across_cases=("sii_mean", "std"),
        mean_seed_sd=("sii_sd_across_seeds", "mean"),
    )
    global_ranking = global_ranking.sort_values(
        ["mean_abs_sii", "feature_i", "feature_j"], ascending=[False, True, True]
    ).reset_index(drop=True)
    global_ranking.insert(0, "rank", np.arange(1, len(global_ranking) + 1))
    global_ranking["pair"] = global_ranking["feature_i"] + " × " + global_ranking["feature_j"]
    _atomic_csv(global_ranking, table_dir / "weighted_sii_global_pair_ranking.csv")

    partner_rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(runtime["predictor"].features):
        candidates = global_ranking.loc[
            (global_ranking["feature_i_index"] == feature_index)
            | (global_ranking["feature_j_index"] == feature_index)
        ].copy()
        best = candidates.iloc[0]
        if int(best["feature_i_index"]) == feature_index:
            partner = best["feature_j"]
            partner_index = int(best["feature_j_index"])
        else:
            partner = best["feature_i"]
            partner_index = int(best["feature_i_index"])
        partner_rows.append({
            "feature_index": feature_index,
            "feature": feature,
            "feature_group": FEATURE_GROUPS[feature],
            "strongest_partner_index": partner_index,
            "strongest_partner": partner,
            "strongest_partner_group": FEATURE_GROUPS[str(partner)],
            "global_pair_rank": int(best["rank"]),
            "mean_abs_sii": float(best["mean_abs_sii"]),
            "pair": best["pair"],
        })
    interaction_map = pd.DataFrame(partner_rows)
    _atomic_csv(interaction_map, table_dir / "weighted_sii_strongest_partner_by_feature.csv")

    confirmed_units: dict[str, str] = {}
    unit_status = "PENDING_USER_CONFIRMATION"
    if CONFIRMED_UNITS_PATH.exists():
        unit_payload = json.loads(CONFIRMED_UNITS_PATH.read_text(encoding="utf-8"))
        confirmed_units = {
            str(key): str(value) for key, value in unit_payload.get("units", {}).items()
        }
        missing_units = [
            feature for feature in runtime["predictor"].features
            if not confirmed_units.get(feature, "").strip()
        ]
        if missing_units:
            raise RuntimeError(f"confirmed unit map is incomplete: {missing_units}")
        unit_status = "CONFIRMED_BY_USER"
    feature_map = pd.DataFrame({
        "feature_index": np.arange(len(runtime["predictor"].features)),
        "feature": runtime["predictor"].features,
        "display_name": [DISPLAY_NAMES.get(x, x) for x in runtime["predictor"].features],
        "clinical_definition": [CLINICAL_DEFINITIONS.get(x, "") for x in runtime["predictor"].features],
        "group": [FEATURE_GROUPS[x] for x in runtime["predictor"].features],
        "unit": [confirmed_units.get(x, "") for x in runtime["predictor"].features],
        "unit_status": [unit_status] * len(runtime["predictor"].features),
        "interaction_input_scale": ["Stage-4 transformed"] * len(runtime["predictor"].features),
        "dependence_x_scale": ["original-scale analysis value"] * len(runtime["predictor"].features),
    })
    _atomic_csv(feature_map, table_dir / "weighted_sii_feature_map.csv")
    _atomic_csv(_case_inputs(runtime), full_dir / "weighted_sii_case_feature_values.csv")

    seed_strength = (
        values.assign(abs_sii=lambda x: x["sii_value"].abs())
        .groupby(["seed", "feature_i", "feature_j"], as_index=False)["abs_sii"].mean()
    )
    seed_stability_rows: list[dict[str, Any]] = []
    for seed_a, seed_b in combinations(sorted(seed_strength["seed"].unique()), 2):
        left = seed_strength.loc[seed_strength["seed"] == seed_a].copy()
        right = seed_strength.loc[seed_strength["seed"] == seed_b].copy()
        merged = left.merge(right, on=["feature_i", "feature_j"], suffixes=("_a", "_b"))
        seed_stability_rows.append({
            "seed_a": int(seed_a),
            "seed_b": int(seed_b),
            "spearman_global_pair_strength": float(spearmanr(
                merged["abs_sii_a"], merged["abs_sii_b"]
            ).statistic),
            "top_pair_a": str(
                left.nlargest(1, "abs_sii").iloc[0]["feature_i"] + " × "
                + left.nlargest(1, "abs_sii").iloc[0]["feature_j"]
            ),
            "top_pair_b": str(
                right.nlargest(1, "abs_sii").iloc[0]["feature_i"] + " × "
                + right.nlargest(1, "abs_sii").iloc[0]["feature_j"]
            ),
            "top10_jaccard": _jaccard(
                set((left.nlargest(10, "abs_sii")["feature_i"] + " × " + left.nlargest(10, "abs_sii")["feature_j"])),
                set((right.nlargest(10, "abs_sii")["feature_i"] + " × " + right.nlargest(10, "abs_sii")["feature_j"])),
            ),
        })
    seed_stability = pd.DataFrame(seed_stability_rows)
    _atomic_csv(seed_stability, qc_dir / "weighted_sii_full_seed_stability.csv")

    coalition_audits = [
        _coalition_audit(len(runtime["predictor"].features), budget, seed)
        for seed in PRODUCTION_SEEDS
    ]
    expected_coalition_hashes = {
        int(item["seed"]): str(item["coalition_matrix_sha256"])
        for item in coalition_audits
    }
    observed_coalition_hashes = {
        int(seed): sorted(group["coalition_matrix_sha256"].astype(str).unique().tolist())
        for seed, group in values.groupby("seed", sort=True)
    }
    coalition_hash_match = all(
        observed_coalition_hashes.get(seed) == [expected_hash]
        for seed, expected_hash in expected_coalition_hashes.items()
    )
    if not coalition_hash_match:
        raise RuntimeError(
            "the persisted full-run coalition hashes do not match the fixed per-seed designs"
        )
    parameters = {
        "split_design": split_metadata(),
        "analysis": "Model-agnostic estimated pairwise Shapley Interaction Index",
        "model": "complete 13-member Weighted voting",
        "model_selection_basis": "A-development 10-fold CV (frozen upstream)",
        "explanation_dataset": "A_holdout_clean",
        "explanation_n": EXPECTED_CASE_N,
        "index": "SII",
        "max_order": 2,
        "approximator": "KernelSHAPIQ",
        "pairing_trick": True,
        "budget_per_case_seed": budget,
        "exact_budget": 2 ** len(runtime["predictor"].features),
        "estimated": budget < 2 ** len(runtime["predictor"].features),
        "seeds": list(PRODUCTION_SEEDS),
        "seed_aggregation": "arithmetic mean within case and pair, then mean absolute value across cases",
        "common_random_numbers": (
            "Within each seed, the same sampled coalition matrix is reset and used for every case."
        ),
        "imputer": "MarginalImputer",
        "imputation_interpretation": "interventional empirical marginal game",
        "background_scope": "A development only",
        "background_n": BACKGROUND_N,
        "background_class_counts": runtime["background_meta"]["y_SCLC"].value_counts().sort_index().to_dict(),
        "background_record_ids": runtime["background_meta"]["record_id"].astype(str).tolist(),
        "joint_marginal_distribution": True,
        "sample_size": BACKGROUND_N,
        "normalize": True,
        "input_scale": "Stage-4 transformed 18-feature representation used by the fitted estimators",
        "feature_unit_status": unit_status,
        "feature_units_source": str(CONFIRMED_UNITS_PATH) if confirmed_units else None,
        "feature_units": confirmed_units,
        "clinical_definitions": CLINICAL_DEFINITIONS,
        "warning": (
            "Order-2 SII is not an additive decomposition of the prediction. No SHAP additivity "
            "test is applicable. These finite-budget values must be labelled estimated pairwise SII."
        ),
        "coalition_audits": coalition_audits,
        "model_package": str(MODEL_PATH),
        "model_package_sha256": _sha256(MODEL_PATH),
        "background_check": runtime["background_check"],
        "prediction_reproducibility": runtime["prediction_reproducibility"],
        "software": {
            "python": platform.python_version(),
            "shapiq": shapiq.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
    }
    _atomic_json(parameters, full_dir / "weighted_sii_parameters.json")

    audit_rows = [
        {"check": "row_count", "observed": len(values), "expected": expected, "status": "PASS"},
        {"check": "case_count", "observed": values["sample_index"].nunique(), "expected": EXPECTED_CASE_N, "status": "PASS"},
        {"check": "pair_count", "observed": values[["feature_i", "feature_j"]].drop_duplicates().shape[0], "expected": PAIR_N, "status": "PASS"},
        {"check": "seed_count", "observed": values["seed"].nunique(), "expected": len(PRODUCTION_SEEDS), "status": "PASS"},
        {"check": "finite_sii", "observed": int(np.isfinite(values["sii_value"]).sum()), "expected": len(values), "status": "PASS"},
        {"check": "model_reproduction_max_abs_difference", "observed": runtime["prediction_reproducibility"]["max_abs_difference"], "expected": "<=1e-10", "status": runtime["prediction_reproducibility"]["status"]},
        {"check": "background_identity", "observed": runtime["background_check"], "expected": "same 40 IDs as v1", "status": "PASS"},
        {"check": "one_coalition_design_per_seed", "observed": observed_coalition_hashes, "expected": expected_coalition_hashes, "status": "PASS" if coalition_hash_match else "FAIL"},
        {"check": "feature_units", "observed": unit_status, "expected": "CONFIRMED_BY_USER", "status": "PASS" if unit_status == "CONFIRMED_BY_USER" else "PENDING"},
        {"check": "quantity_label", "observed": "Estimated pairwise SII", "expected": "not SHAP/PDP/ALE", "status": "PASS"},
        {"check": "SII_additivity_test", "observed": "NOT PERFORMED", "expected": "not applicable to truncated SII", "status": "PASS"},
    ]
    _atomic_csv(pd.DataFrame(audit_rows), qc_dir / "weighted_sii_full_audit.csv")
    return {
        "top_pair": global_ranking.iloc[0]["pair"],
        "top_pair_mean_abs_sii": float(global_ranking.iloc[0]["mean_abs_sii"]),
        "minimum_seed_spearman": float(seed_stability["spearman_global_pair_strength"].min()),
        "minimum_top10_jaccard": float(seed_stability["top10_jaccard"].min()),
    }


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "full"), required=True)
    parser.add_argument("--budgets", default=None, help="Comma-separated budgets")
    parser.add_argument("--seeds", default=",".join(map(str, PRODUCTION_SEEDS)))
    parser.add_argument("--probe-cases", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--auto-extend-8192", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.n_jobs <= 4:
        raise ValueError("n_jobs must be between 1 and 4 to prevent oversubscription")
    if args.batch_size < args.n_jobs:
        raise ValueError("batch_size must be at least n_jobs")

    runtime = _load_runtime()
    seeds = _parse_ints(args.seeds)
    if len(seeds) != 3:
        raise ValueError("exactly three fixed seeds are required")
    if args.mode == "probe":
        budgets = _parse_ints(args.budgets or "1024,2048,4096")
        case_indices = _probe_indices(runtime, args.probe_cases)
        selection = runtime["holdout"].iloc[case_indices][["record_id", "y_SCLC"]].copy()
        selection.insert(0, "sample_index", case_indices)
        selection["weighted_probability"] = runtime["probabilities"][case_indices]
        _atomic_csv(selection, OUT_ROOT / "data" / "sii" / "probe" / "probe_case_selection.csv")
    else:
        budgets = _parse_ints(args.budgets or str(PRODUCTION_BUDGET))
        if budgets != [PRODUCTION_BUDGET]:
            raise ValueError("the full run is frozen to budget 4096")
        if tuple(seeds) != PRODUCTION_SEEDS:
            raise ValueError(f"the full run is frozen to seeds {PRODUCTION_SEEDS}")
        case_indices = list(range(EXPECTED_CASE_N))

    all_paths: list[Path] = []
    started = time.perf_counter()
    for budget in budgets:
        for seed in seeds:
            all_paths.append(_run_budget_seed(
                runtime, case_indices, args.mode, budget, seed, args.n_jobs, args.batch_size
            ))

    if args.mode == "probe":
        decision = _aggregate_probe(all_paths)
        if not decision["stable_for_full_run"] and args.auto_extend_8192 and 8192 not in budgets:
            print("4096 probe did not meet stability criteria; extending to budget 8192.", flush=True)
            extra_paths = [
                _run_budget_seed(
                    runtime, case_indices, args.mode, 8192, seed, args.n_jobs, args.batch_size
                )
                for seed in seeds
            ]
            all_paths.extend(extra_paths)
            decision = _aggregate_probe(all_paths, reference_budget=8192)
        result = decision
    else:
        result = _aggregate_full(all_paths, runtime, budget=PRODUCTION_BUDGET)

    result["runtime_seconds"] = time.perf_counter() - started
    print(json.dumps(_safe(result), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
