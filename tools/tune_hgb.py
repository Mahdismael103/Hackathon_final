"""Systematic tuning of the HistGradientBoosting defaults (n >= 1000 regime), by cross-validation.

    python tools/tune_hgb.py

One-at-a-time variation around the current default on every supervised practice dataset with at least
1,000 training rows. Each configuration is scored by 3-fold CV on the *training* rows only (the hidden
test labels are never used), and summarised as the mean change versus the default.
Metrics: macro-F1 (classification), R2 (regression), so all datasets are on a comparable 0..1 scale.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import f1_score, r2_score  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold  # noqa: E402

from automl.api import _clean_class_target  # noqa: E402
from automl.io_contract import load_task  # noqa: E402
from automl.preprocess import Preparer  # noqa: E402
from automl.profiling import profile_dataset, to_numeric  # noqa: E402
from automl.supervised import _hgb_params  # noqa: E402

VARIANTS = {
    "default": {},
    "lr=0.04": {"learning_rate": 0.04}, "lr=0.12": {"learning_rate": 0.12},
    "leaves=15": {"max_leaf_nodes": 15}, "leaves=63": {"max_leaf_nodes": 63},
    "min_leaf=10": {"min_samples_leaf": 10}, "min_leaf=40": {"min_samples_leaf": 40},
    "l2=0": {"l2_regularization": 0.0}, "l2=3": {"l2_regularization": 3.0},
    "max_features=0.7": {"max_features": 0.7}, "max_features=0.5": {"max_features": 0.5},
}


def prepared(d: Path):
    spec = load_task(data=str(d))
    train = spec.train[spec.train[spec.target].notna()].reset_index(drop=True)
    _, roles = profile_dataset(train, spec.test, spec.task, spec.target, spec.feature_types, spec.id_column, None)
    roles.pop(spec.id_column, None)
    y = _clean_class_target(train[spec.target]) if spec.task == "classification" else to_numeric(train[spec.target])
    prep = Preparer(roles).fit(train[list(roles)], y=y.values, task=spec.task)
    X, mask = prep.transform_tree(train[list(roles)], train=True)
    return spec.task, X, mask, np.asarray(y)


def cv_score(task, X, mask, y, extra, n):
    params = dict(_hgb_params(n, mask, n_classes=len(np.unique(y)) if task == "classification" else 2), **extra)
    if task == "classification":
        y = np.unique(y.astype(str), return_inverse=True)[1]
        splits = StratifiedKFold(3, shuffle=True, random_state=0).split(X, y)
        Model = HistGradientBoostingClassifier
    else:
        splits = KFold(3, shuffle=True, random_state=0).split(X)
        Model = HistGradientBoostingRegressor
    pred = np.zeros(len(y))
    for tr, va in splits:
        pred[va] = Model(**params).fit(X[tr], y[tr]).predict(X[va])
    return f1_score(y, pred, average="macro") if task == "classification" else r2_score(y, pred)


def main():
    rows = []
    for d in sorted((ROOT / "practice").iterdir()):
        meta = json.loads((d / "meta.json").read_text())
        if meta["task"] == "clustering":
            continue
        task, X, mask, y = prepared(d)
        if len(y) < 1000:
            continue
        t = time.time()
        row = {"dataset": d.name, "task": task}
        for name, extra in VARIANTS.items():
            row[name] = cv_score(task, X, mask, y, extra, len(y))
        print(f"{d.name:26s} {time.time() - t:5.1f}s  " +
              " ".join(f"{k}={row[k] - row['default']:+.4f}" for k in VARIANTS if k != "default"), flush=True)
        rows.append(row)
    df = pd.DataFrame(rows)
    delta = df[list(VARIANTS)].sub(df["default"], axis=0)
    summary = pd.DataFrame({"mean_change": delta.mean(), "datasets_better": (delta > 1e-4).sum(),
                            "datasets_worse": (delta < -1e-4).sum(), "worst_change": delta.min()})
    print("\nchange versus the current default (CV on training rows, mean over datasets):")
    print(summary.drop(index="default").sort_values("mean_change", ascending=False).round(4).to_string())
    (ROOT / "outputs").mkdir(exist_ok=True)
    df.to_csv(ROOT / "outputs" / "tune_hgb.csv", index=False)


if __name__ == "__main__":
    main()
