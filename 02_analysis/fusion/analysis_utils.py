from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
from fold_selection import EXPECTED_DIRECT as _EXPECTED_DIRECT, COMPOSITE_FEATURES, DIRECT_FEATURES


EXPECTED_DIRECT = list(_EXPECTED_DIRECT)
FIXED_COMPOSITES = list(COMPOSITE_FEATURES)

def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_safe(x) for x in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _prepare_center(frame: pd.DataFrame, center: str) -> pd.DataFrame:
    out = frame.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    if set(out["center"].astype(str)) != {center}:
        raise ValueError(f"unexpected center for {center}")
    if "missing_direct_n" in out.columns:
        out = out.loc[out["missing_direct_n"].astype(float) < len(DIRECT_FEATURES)].copy()
    if len(out) < 20 or out["y_SCLC"].nunique() != 2:
        raise ValueError(f"unexpected {center} cohort size")
    return out.reset_index(drop=True)
