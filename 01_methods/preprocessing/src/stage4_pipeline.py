from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DECISION_VERSION = "E1-E8_P1-P3_SEED_20260902_v2_23design"
MASTER_SEED = 20260902
SEED_NAMESPACE = "SCLC_NSCLC|seed_v2_23design"
SEED_MODULUS = 2_147_483_646

DIRECT_FEATURES = [
    "WBC", "NEUT#", "NEUT%", "MONO#", "MONO%", "PLT", "RDW-CV", "CRP", "FIB", "LDH", "PCT",
    "LYMPH#", "LYMPH%", "EO#", "EO%", "BASO#", "BASO%",
    "HGB", "ALB", "TP", "GLB", "PA", "GLU", "TG", "TC", "HDL-C", "LDL-C", "UA",
    "MCH", "MCHC", "Cl", "Na",
]
COMPOSITE_FEATURES = ["SIRI", "LMR", "GAR", "PNI", "HALP"]
ALL_FEATURES = DIRECT_FEATURES + COMPOSITE_FEATURES
LOG_FEATURES = [
    "WBC", "NEUT#", "MONO#", "PLT", "CRP", "FIB", "LDH", "SIRI",
    "LYMPH#", "EO#", "BASO#", "LMR", "GLU", "TG", "GAR", "HALP",
]
META_COLUMNS = ["record_id", "center", "source_row", "label", "y_SCLC", "missing_direct_n"]
COMPOSITE_FORMULA_VERSION = "composites_v2_32direct_E5"
QUANTILE_METHOD = "linear"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def record_id_hash(ids: Iterable[str]) -> str:
    canonical = "\n".join(sorted(str(x) for x in ids)) + "\n"
    return sha256_bytes(canonical.encode("utf-8"))


def dataframe_hash(df: pd.DataFrame, columns: Sequence[str]) -> str:
    text = df.loc[:, list(columns)].to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
        float_format="%.17g",
    )
    return sha256_bytes(text.encode("utf-8"))


def derive_seed(*parts: object) -> int:
    suffix = "|".join(str(x) for x in parts)
    payload = f"{SEED_NAMESPACE}|{MASTER_SEED}|{suffix}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return 1 + (int(digest[:16], 16) % SEED_MODULUS)


def load_base_workbook(path: Path) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}
    for center in ("A", "B", "C"):
        df = pd.read_excel(path, sheet_name=f"{center}_base", engine="openpyxl")
        missing = [x for x in META_COLUMNS + DIRECT_FEATURES if x not in df.columns]
        if missing:
            raise ValueError(f"{center}_base missing required columns: {missing}")
        df = df.loc[:, META_COLUMNS + DIRECT_FEATURES + COMPOSITE_FEATURES].copy()
        df["record_id"] = df["record_id"].astype(str)
        df = df.sort_values("record_id", kind="mergesort").reset_index(drop=True)
        if df["record_id"].isna().any() or df["record_id"].duplicated().any():
            raise ValueError(f"{center}_base record_id is missing or duplicated")
        if set(df["center"].astype(str)) != {center}:
            raise ValueError(f"{center}_base contains an unexpected center code")
        expected_y = (df["label"].astype(str) == "SCLC").astype(int)
        if not np.array_equal(expected_y.to_numpy(), df["y_SCLC"].astype(int).to_numpy()):
            raise ValueError(f"{center}_base label/y_SCLC mapping mismatch")
        for feature in DIRECT_FEATURES:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
        sheets[center] = df
    all_ids = pd.concat([x[["record_id"]] for x in sheets.values()], ignore_index=True)
    if all_ids["record_id"].duplicated().any():
        raise ValueError("record_id overlaps across centers")
    return sheets


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = denominator.notna() & numerator.notna() & (denominator > 0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def compute_composites(imputed_direct: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=imputed_direct.index)
    out["SIRI"] = _safe_ratio(imputed_direct["NEUT#"] * imputed_direct["MONO#"], imputed_direct["LYMPH#"])
    out["LMR"] = _safe_ratio(imputed_direct["LYMPH#"], imputed_direct["MONO#"])
    out["GAR"] = _safe_ratio(imputed_direct["GLU"], imputed_direct["ALB"])
    out["PNI"] = imputed_direct["ALB"] + 5.0 * imputed_direct["LYMPH#"]
    out["HALP"] = _safe_ratio(
        imputed_direct["HGB"] * imputed_direct["ALB"] * imputed_direct["LYMPH#"],
        imputed_direct["PLT"],
    )
    return out.loc[:, COMPOSITE_FEATURES]


@dataclass
class FrozenPreprocessor:
    medians: dict[str, float]
    q01: dict[str, float]
    q99: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "FrozenPreprocessor":
        direct = frame.loc[:, DIRECT_FEATURES].apply(pd.to_numeric, errors="coerce")
        medians_series = direct.median(axis=0, skipna=True)
        if medians_series.isna().any():
            bad = medians_series.index[medians_series.isna()].tolist()
            raise ValueError(f"all-missing direct features in training scope: {bad}")
        imputed = direct.fillna(medians_series)
        features = pd.concat([imputed, compute_composites(imputed)], axis=1).loc[:, ALL_FEATURES]
        if features.isna().any().any():
            bad = features.columns[features.isna().any()].tolist()
            raise ValueError(f"non-computable composite(s) in training scope: {bad}")
        q01: dict[str, float] = {}
        q99: dict[str, float] = {}
        for feature in ALL_FEATURES:
            values = features[feature].to_numpy(dtype=float)
            q01[feature] = float(np.quantile(values, 0.01, method=QUANTILE_METHOD))
            q99[feature] = float(np.quantile(values, 0.99, method=QUANTILE_METHOD))
            if q01[feature] > q99[feature]:
                raise ValueError(f"invalid quantile bounds for {feature}")
        return cls(
            medians={k: float(medians_series[k]) for k in DIRECT_FEATURES},
            q01=q01,
            q99=q99,
        )

    def parameter_dict(self) -> dict[str, Any]:
        return {
            "decision_version": DECISION_VERSION,
            "direct_features": DIRECT_FEATURES,
            "composite_features": COMPOSITE_FEATURES,
            "all_features": ALL_FEATURES,
            "log1p_features": LOG_FEATURES,
            "composite_formula_version": COMPOSITE_FORMULA_VERSION,
            "quantile_method": QUANTILE_METHOD,
            "medians": self.medians,
            "q01": self.q01,
            "q99": self.q99,
        }

    @classmethod
    def from_parameter_dict(cls, obj: dict[str, Any]) -> "FrozenPreprocessor":
        if obj["direct_features"] != DIRECT_FEATURES or obj["all_features"] != ALL_FEATURES:
            raise ValueError("feature order mismatch in serialized preprocessor")
        if obj["log1p_features"] != LOG_FEATURES or obj["quantile_method"] != QUANTILE_METHOD:
            raise ValueError("frozen transform rule mismatch in serialized preprocessor")
        return cls(
            medians={k: float(v) for k, v in obj["medians"].items()},
            q01={k: float(v) for k, v in obj["q01"].items()},
            q99={k: float(v) for k, v in obj["q99"].items()},
        )

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.parameter_dict()))

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        missing_cols = [x for x in DIRECT_FEATURES if x not in frame.columns]
        if missing_cols:
            raise ValueError(f"transform input missing direct features: {missing_cols}")
        direct = frame.loc[:, DIRECT_FEATURES].apply(pd.to_numeric, errors="coerce")
        imputed = direct.fillna(pd.Series(self.medians))
        composites = compute_composites(imputed)
        features = pd.concat([imputed, composites], axis=1).loc[:, ALL_FEATURES]
        invalid_composite_n = int(composites.isna().sum().sum())
        clipped_low_n = 0
        clipped_high_n = 0
        transformed = features.copy()
        for feature in ALL_FEATURES:
            raw = transformed[feature]
            clipped_low_n += int((raw < self.q01[feature]).fillna(False).sum())
            clipped_high_n += int((raw > self.q99[feature]).fillna(False).sum())
            transformed[feature] = raw.clip(lower=self.q01[feature], upper=self.q99[feature])
        log_domain_error_n = 0
        for feature in LOG_FEATURES:
            bad = (transformed[feature] <= -1).fillna(False)
            log_domain_error_n += int(bad.sum())
            transformed.loc[bad, feature] = np.nan
            transformed[feature] = np.log1p(transformed[feature])
        audit = {
            "input_missing_direct_n": int(direct.isna().sum().sum()),
            "invalid_composite_n": invalid_composite_n,
            "clipped_low_n": clipped_low_n,
            "clipped_high_n": clipped_high_n,
            "log_domain_error_n": log_domain_error_n,
            "remaining_missing_n": int(transformed.isna().sum().sum()),
        }
        return transformed.loc[:, ALL_FEATURES], audit


def environment_info() -> dict[str, Any]:
    import openpyxl
    import sklearn

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "openpyxl": openpyxl.__version__,
    }
