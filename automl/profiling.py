"""Data profiling: infers column roles and produces the structured profile that is returned."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

ID_NAME_RE = re.compile(
    r"(^|[_\s\-.])(id|uuid|guid|udi|key|idx|index|code|number|no)$|^(id|uuid|guid|udi|row|rowid|"
    r"row_id|index|unnamed:?\s*\d+|sample|record)$",
    re.I,
)
DATE_HINT_RE = re.compile(r"\d{1,4}[-/.:]\d{1,2}")


def to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.strip(), errors="coerce").astype(float)


def parse_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.to_datetime(s, errors="coerce")
        if out.notna().mean() < 0.9 * s.notna().mean():
            try:
                out = pd.to_datetime(s, errors="coerce", format="mixed")
            except (TypeError, ValueError):
                pass
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def infer_type(s: pd.Series, name: str, n_rows: int) -> str:
    nn = s.dropna()
    if len(nn) == 0:
        return "constant"
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    # object / string column
    sample = nn.sample(min(len(nn), 2000), random_state=0) if len(nn) > 2000 else nn
    num = pd.to_numeric(sample.astype(str).str.strip(), errors="coerce")
    if num.notna().mean() >= 0.95:
        return "numeric"
    sstr = sample.astype(str)
    if sstr.str.contains(DATE_HINT_RE).mean() > 0.9:
        dt = parse_datetime(sstr)
        if dt.notna().mean() >= 0.9:
            return "datetime"
    avg_words = sstr.str.split().str.len().mean()
    uniq_ratio = nn.nunique() / max(len(nn), 1)
    if avg_words >= 4 and uniq_ratio > 0.3:
        return "text"
    return "categorical"


def is_id_like(s: pd.Series, name: str, inferred: str, n_rows: int) -> bool:
    nn = s.dropna()
    if len(nn) < 20:
        return False
    uniq_ratio = nn.nunique() / len(nn)
    named = bool(ID_NAME_RE.search(str(name).strip()))
    if inferred in ("categorical", "text") and uniq_ratio > 0.98 and named:
        return True
    if inferred == "categorical" and uniq_ratio > 0.98:
        # all-unique short tokens (e.g. "7590-VHVEG", "M14860") carry no generalisable signal
        avg_len = nn.astype(str).str.len().mean()
        return avg_len < 40
    if inferred == "numeric" and uniq_ratio > 0.98:
        v = to_numeric(nn).values
        if not np.allclose(v, np.round(v)):
            return False
        if named:
            return True
        d = np.diff(v)
        # a running counter 0..n-1 / 1..n
        return len(d) > 0 and (np.all(d == 1) or np.all(d == -1))
    return False


def _canonical(s: pd.Series) -> pd.Series:
    """Value representation used to find duplicate columns: numbers compare as numbers (1 == 1.0)."""
    v = to_numeric(s)
    if v.notna().sum() == s.notna().sum():
        return v.round(9)          # numbers hash and compare as numbers: fast, and 1 == 1.0
    return s.astype(str)


def _hash_series(s: pd.Series) -> int:
    return int(pd.util.hash_pandas_object(_canonical(s), index=False).sum())


def profile_dataset(train: pd.DataFrame, test: pd.DataFrame | None, task: str, target: str | None,
                    declared: dict[str, str], id_column: str | None, n_clusters: int | None) -> dict:
    """Return (profile dict, roles dict). roles maps feature column -> numeric/categorical/datetime/text."""
    n = len(train)
    feature_cols = [c for c in train.columns if c != target]
    if test is not None:
        # never use columns the test table does not have (prevents label leakage)
        feature_cols = [c for c in feature_cols if c in test.columns]

    cols_info = []
    roles: dict[str, str] = {}
    dropped: dict[str, str] = {}
    hashes: dict[int, str] = {}

    for c in feature_cols:
        s = train[c]
        inferred = infer_type(s, c, n)
        decl = declared.get(str(c))
        role = decl if decl in ("numeric", "categorical", "datetime", "text", "id") else inferred
        if role == "numeric" and inferred not in ("numeric", "constant"):
            role = inferred  # declared numeric but values are not parseable
        info = {
            "name": str(c),
            "dtype": str(s.dtype),
            "inferred_type": inferred if inferred != "constant" else (decl or "numeric"),
            "declared_type": decl,
            "missing_count": int(s.isna().sum()),
            "missing_fraction": float(s.isna().mean()) if n else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }
        info["unique_fraction"] = info["n_unique"] / max(n - info["missing_count"], 1)
        info["is_constant"] = info["n_unique"] <= 1
        info["is_id_like"] = False
        info["duplicate_of"] = None

        if c == id_column or role == "id":
            info["is_id_like"] = True
            dropped[str(c)] = "id column"
        elif info["is_constant"]:
            dropped[str(c)] = "constant"
        elif is_id_like(s, c, role, n) and decl not in ("numeric", "categorical"):
            info["is_id_like"] = True
            dropped[str(c)] = "id-like (unique per row)"
        else:
            h = _hash_series(s)
            if h in hashes and _canonical(train[hashes[h]]).equals(_canonical(s)):
                info["duplicate_of"] = hashes[h]
                dropped[str(c)] = f"duplicate of {hashes[h]}"
            else:
                hashes.setdefault(h, c)

        if role == "numeric" or (inferred == "numeric" and role != "categorical"):
            v = to_numeric(s)
            if v.notna().any():
                info.update({
                    "mean": float(v.mean()), "std": float(v.std()) if v.notna().sum() > 1 else 0.0,
                    "min": float(v.min()), "max": float(v.max()), "median": float(v.median()),
                    "skew": float(v.skew()) if v.notna().sum() > 2 else 0.0,
                })
        if role in ("categorical", "text"):
            vc = s.astype(str).where(s.notna()).value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in vc.items()}
        info["role"] = role if str(c) not in dropped else "dropped"
        cols_info.append(info)
        if str(c) not in dropped:
            roles[c] = role if role != "constant" else "numeric"

    profile: dict = {
        "task": task,
        "target": target,
        "n_rows": int(n),
        "n_rows_train": int(n),
        "n_rows_test": int(len(test)) if test is not None else 0,
        "n_columns": int(train.shape[1]),
        "n_features": int(len(feature_cols)),
        "n_features_used": int(len(roles)),
        "numeric_columns": [str(c) for c, r in roles.items() if r == "numeric"],
        "categorical_columns": [str(c) for c, r in roles.items() if r == "categorical"],
        "datetime_columns": [str(c) for c, r in roles.items() if r == "datetime"],
        "text_columns": [str(c) for c, r in roles.items() if r == "text"],
        "constant_columns": [i["name"] for i in cols_info if i["is_constant"]],
        "duplicate_columns": [i["name"] for i in cols_info if i["duplicate_of"] is not None],
        "id_like_columns": [i["name"] for i in cols_info if i["is_id_like"]],
        "dropped_columns": dropped,
        "feature_types": {i["name"]: (i["role"] if i["role"] != "dropped" else
                                      f"dropped ({dropped.get(i['name'], '')})") for i in cols_info},
        "missing_by_column": {i["name"]: i["missing_count"] for i in cols_info if i["missing_count"] > 0},
        "columns_with_missing": [i["name"] for i in cols_info if i["missing_count"] > 0],
        "missing_by_column_test": ({str(c): int(test[c].isna().sum()) for c in feature_cols
                                    if test[c].isna().any()} if test is not None else {}),
        "missing_cells": int(sum(i["missing_count"] for i in cols_info)),
        "missing_fraction": float(sum(i["missing_count"] for i in cols_info) / max(n * max(len(feature_cols), 1), 1)),
        "has_missing_values": any(i["missing_count"] > 0 for i in cols_info) or bool(
            test is not None and feature_cols and test[feature_cols].isna().any().any()),
        "duplicate_rows": int(train[feature_cols].duplicated().sum()) if feature_cols else 0,
        "columns": cols_info,
    }

    # scale differences between numeric features
    stds = [i.get("std") for i in cols_info if i.get("std") not in (None, 0.0) and i["role"] == "numeric"]
    if len(stds) >= 2:
        profile["numeric_scale_ratio"] = float(max(stds) / max(min(stds), 1e-12))
        profile["features_differently_scaled"] = bool(profile["numeric_scale_ratio"] > 10)

    if task == "classification" and target in train.columns:
        y = train[target].dropna()
        vc = y.value_counts()
        frac = vc / vc.sum()
        profile["target_profile"] = {
            "n_classes": int(len(vc)),
            "class_counts": {str(k): int(v) for k, v in vc.items()},
            "class_fractions": {str(k): float(v) for k, v in frac.items()},
            "majority_class": str(vc.index[0]),
            "minority_class": str(vc.index[-1]),
            "imbalance_ratio": float(vc.iloc[0] / max(vc.iloc[-1], 1)),
            "is_imbalanced": bool(frac.iloc[-1] < 0.5 / len(vc) or vc.iloc[0] / max(vc.iloc[-1], 1) > 3),
            "missing_target": int(train[target].isna().sum()),
        }
        profile["n_classes"] = profile["target_profile"]["n_classes"]
        profile["is_imbalanced"] = profile["target_profile"]["is_imbalanced"]
    elif task == "regression" and target in train.columns:
        y = to_numeric(train[target]).dropna()
        profile["target_profile"] = {
            "mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()),
            "max": float(y.max()), "median": float(y.median()), "skew": float(y.skew()),
            "is_skewed": bool(abs(y.skew()) > 1), "missing_target": int(train[target].isna().sum()),
        }
    elif task == "clustering":
        profile["n_clusters"] = int(n_clusters) if n_clusters else None
    return profile, roles
