"""Third sweep: size and shape limits, scored against hidden labels.

    python tools/stress_test3.py              # all cases
    python tools/stress_test3.py wide_1000    # cases whose name contains "wide_1000"

Each case must finish in < 60 s with model status "ok"; quality is printed for comparison between
versions (it is not a pass/fail criterion except where a minimum is given).
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
from sklearn import datasets as skd
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(tempfile.gettempdir()) / "automl_stress3"
rng = np.random.default_rng(3)


def save(name, train, test, meta, truth):
    d = BASE / "data" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    train.to_csv(d / "train.csv", index=False)
    if test is not None:
        test.to_csv(d / "test.csv", index=False)
    (d / "meta.json").write_text(json.dumps(meta))
    return d, truth


def split(X, y, frac=0.75):
    idx = rng.permutation(len(X))
    cut = int(frac * len(X))
    tr, te = idx[:cut], idx[cut:]
    return X.iloc[tr].assign(target=np.asarray(y)[tr]), X.iloc[te].reset_index(drop=True), np.asarray(y)[te]


def wide_1000():
    X, y = skd.make_classification(3000, 1000, n_informative=15, n_redundant=5, random_state=1)
    tr, te, yt = split(pd.DataFrame(X).add_prefix("g"), y)
    return save("wide_1000_cols", tr, te, {"task": "classification", "target": "target"}, yt), "classification", 0.70


def many_classes_20():
    X, y = skd.make_classification(6000, 30, n_informative=20, n_classes=20, n_clusters_per_class=1,
                                   class_sep=1.5, random_state=2)
    tr, te, yt = split(pd.DataFrame(X).add_prefix("m"), y)
    return save("many_classes_20", tr, te, {"task": "classification", "target": "target"}, yt), "classification", 0.5


def clustering_mostly_noise():
    X, y = skd.make_blobs(2000, 3, centers=4, cluster_std=1.0, random_state=4)
    noise = rng.normal(0, 1, (2000, 30)) * X.std(0).mean()
    df = pd.DataFrame(np.hstack([X, noise])).add_prefix("f")
    return save("clustering_3inf_30noise", df, None, {"task": "clustering", "n_clusters": 4}, y), "clustering", None


def house_80_columns():
    n = 1460
    cats = {f"cat{i}": rng.choice([f"v{j}" for j in range(rng.integers(2, 12))], n) for i in range(40)}
    nums = {f"num{i}": rng.normal(rng.uniform(0, 1000), rng.uniform(1, 300), n) for i in range(38)}
    df = pd.DataFrame({**cats, **nums})
    for c in list(df.columns)[::5]:                 # NaN meaning "none" (PoolQC, Alley...) in many columns
        df.loc[rng.random(n) < rng.uniform(0.05, 0.95), c] = np.nan
    eff = {v: rng.normal(0, 0.15) for v in df["cat0"].dropna().unique()}
    logp = (12 + 0.4 * (df["num0"].fillna(df["num0"].mean()) - df["num0"].mean()) / df["num0"].std()
            + df["cat0"].map(eff).fillna(0) + 0.2 * df["num1"].notna() + rng.normal(0, 0.1, n))
    df.insert(0, "Id", np.arange(1, n + 1))
    tr, te, yt = split(df, np.exp(logp).round())
    tr = tr.rename(columns={"target": "SalePrice"})
    return save("house_80_columns", tr, te, {"task": "regression", "target": "SalePrice"}, yt), "regression", 0.5


def rows_300k():
    n = 300_000
    X = pd.DataFrame(rng.normal(size=(n, 20))).add_prefix("x")
    X["seg"] = rng.choice(list("abcdefghij"), n)
    y = ((X["x0"] * X["x1"] + np.sin(2 * X["x2"]) + (X["seg"] == "c") * 0.8 + rng.normal(0, .6, n)) > .4).astype(int)
    tr, te, yt = split(X, y, 0.8)
    return save("rows_300k", tr, te, {"task": "classification", "target": "target"}, yt), "classification", 0.7


CASES = [wide_1000, many_classes_20, clustering_mostly_noise, house_80_columns, rows_300k]


def main():
    pick = sys.argv[1:]
    bad = 0
    for make in CASES:
        if pick and not any(p in make.__name__ for p in pick):
            continue
        global rng
        rng = np.random.default_rng(abs(hash(make.__name__)) % 2**32 if False else sum(map(ord, make.__name__)))
        (d, truth), task, minimum = make()   # each case has its own seed: same data whichever cases run
        out = BASE / "out" / d.name
        t0 = time.time()
        p = subprocess.run([sys.executable, str(ROOT / "main.py"), "--data", str(d), "--output", str(out)],
                           capture_output=True, text=True, timeout=240)
        dt = time.time() - t0
        try:
            pred = pd.read_csv(out / "predictions.csv")["prediction"].values
            prof = json.loads((out / "profile.json").read_text())
            if task == "classification":
                q, name = skm.f1_score(truth, pred, average="macro"), "F1"
            elif task == "regression":
                q, name = skm.r2_score(truth, pred), "R2"
            else:
                q, name = skm.adjusted_rand_score(truth, pred), "ARI"
            status = prof["model"]["status"]
            good = status == "ok" and dt < 60 and (minimum is None or q >= minimum)
            timing = prof.get("timings", {})
            msg = f"{name}={q:.4f} peak={prof.get('peak_memory_mb')}MB status={status} chosen={prof['model'].get('chosen') or prof['model'].get('ensemble_weights')}"
            if timing:
                msg += f" total={timing.get('total')}s"
        except Exception as e:
            good, msg = False, f"{e} | {p.stderr[-1500:]}"
        bad += not good
        print(f"{'OK' if good else 'FAIL':5s} {d.name:26s} {dt:5.1f}s  {msg}", flush=True)
    print("all good" if bad == 0 else f"{bad} problem case(s)")


if __name__ == "__main__":
    main()
