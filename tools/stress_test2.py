"""Second robustness sweep: messy real-world encodings.

    python tools/stress_test2.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(tempfile.gettempdir()) / "automl_stress2"
rng = np.random.default_rng(1)


def write(name, train, test=None, meta=None):
    d = BASE / "data" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    train.to_csv(d / "train.csv", index=False)
    if test is not None:
        test.to_csv(d / "test.csv", index=False)
    if meta:
        (d / "meta.json").write_text(json.dumps(meta))
    return d, (test if test is not None else train)


def cases():
    n = 800
    x = rng.normal(size=(n, 3))
    df = pd.DataFrame(x, columns=["a", "b", "c"]).round(3).astype(object)
    df.loc[rng.random(n) < 0.15, "a"] = "?"            # UCI-style missing marker (15% of cells)
    df["y"] = (x[:, 0] + x[:, 1] > 0)                    # boolean target
    yield ("question_marks_bool_target", *write("question_marks_bool_target", df.iloc[:600],
           df.iloc[600:].drop(columns="y"), {"task": "classification", "target": "y"}), lambda p: True)

    df = pd.DataFrame({"x": rng.normal(size=n), "empty": np.nan,
                       "when": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC").astype(str)})
    df["price"] = (df["x"] * 100 + 500).round(2).astype(str)   # numeric target stored as text
    yield ("text_target_tz_dates_empty_col", *write("text_target_tz_dates_empty_col", df.iloc[:600],
           df.iloc[600:].drop(columns="price"), {"task": "regression", "target": "price"}), lambda p: True)

    df = pd.DataFrame({"x": rng.normal(size=300), "y": 7.0})
    yield ("constant_regression_target", *write("constant_regression_target", df.iloc[:200],
           df.iloc[200:].drop(columns="y"), {"task": "regression", "target": "y"}),
           lambda p: np.allclose(p, 7.0))

    df = pd.DataFrame({"Weird name (%)": rng.normal(size=n), "col.with.dots": rng.normal(size=n),
                       "city": rng.choice(["a", "b", "c"], n)})
    df["label"] = np.where(df["Weird name (%)"] > 0, "hi", "lo")
    test = df.iloc[600:].drop(columns="label")[["city", "col.with.dots", "Weird name (%)"]].copy()
    test.loc[test.index[:20], "city"] = "never_seen"      # unseen category + reordered columns
    yield ("odd_names_reordered_unseen_cats", *write("odd_names_reordered_unseen_cats", df.iloc[:600], test,
           {"task": "classification", "target": "label"}), lambda p: True)

    from sklearn.datasets import make_blobs
    X, _ = make_blobs(600, 3, centers=3, random_state=0)
    yield ("k_smaller_than_structure", *write("k_smaller_than_structure", pd.DataFrame(X, columns=list("pqr")),
           None, {"task": "clustering", "n_clusters": 2}), lambda p: len(np.unique(p)) == 2)

    df = pd.DataFrame({"x1": rng.normal(size=500)})
    df["cls"] = rng.choice([3, 7, 9], 500)               # integer class labels that are not 0..k-1
    yield ("integer_labels_not_contiguous", *write("integer_labels_not_contiguous", df.iloc[:400],
           df.iloc[400:].drop(columns="cls"), {"task": "classification", "target": "cls"}),
           lambda p: set(np.unique(p)) <= {3, 7, 9})


def main():
    bad = 0
    for name, d, frame, check in cases():
        out = BASE / "out" / name
        t0 = time.time()
        p = subprocess.run([sys.executable, str(ROOT / "main.py"), "--data", str(d), "--output", str(out)],
                           capture_output=True, text=True, timeout=180)
        dt = time.time() - t0
        try:
            pred = pd.read_csv(out / "predictions.csv")
            prof = json.loads((out / "profile.json").read_text())
            status = prof.get("model", {}).get("status")
            good = (len(pred) == len(frame) and not pred["prediction"].isna().any() and status == "ok"
                    and check(pred["prediction"].values) and dt < 60 and p.returncode == 0)
            msg = f"status={status} types={dict(list(prof.get('feature_types', {}).items())[:4])}"
        except Exception as e:
            good, msg = False, f"{e} | {p.stderr[-1500:]}"
        bad += not good
        print(f"{'OK' if good else 'FAIL':5s} {name:34s} {dt:5.1f}s  {msg}")
        if not good:
            print(p.stderr[-2500:])
    print("all good" if bad == 0 else f"{bad} problem case(s)")


if __name__ == "__main__":
    main()
