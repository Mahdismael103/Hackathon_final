"""One-minute self-test: one small dataset per task through the real entry point.

    python tools/smoke_test.py

Builds the datasets in a temporary folder (no downloads), runs main.py on each, and checks that the
outputs are complete, that quality is sensible, and that each run stays under 60 s.
"""
from __future__ import annotations

import json
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


def build(base: Path):
    rng = np.random.default_rng(0)
    bc = skd.load_breast_cancer(as_frame=True)
    idx = rng.permutation(len(bc.data))
    tr, te = idx[:400], idx[400:]
    d = base / "classification"; d.mkdir()
    bc.data.iloc[tr].assign(target=bc.target.iloc[tr].values).to_csv(d / "train.csv", index=False)
    bc.data.iloc[te].to_csv(d / "test.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"task": "classification", "target": "target"}))
    yield d, bc.target.values[te], 0.90

    db = skd.load_diabetes(as_frame=True, scaled=False)
    idx = rng.permutation(len(db.data))
    tr, te = idx[:330], idx[330:]
    d = base / "regression"; d.mkdir()
    db.data.iloc[tr].assign(target=db.target.iloc[tr].values).to_csv(d / "train.csv", index=False)
    db.data.iloc[te].to_csv(d / "test.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"task": "regression", "target": "target"}))
    yield d, db.target.values[te], 0.30

    ir = skd.load_iris(as_frame=True)
    d = base / "clustering"; d.mkdir()
    ir.data.to_csv(d / "train.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"task": "clustering", "n_clusters": 3}))
    yield d, ir.target.values, 0.55


def main():
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for d, truth, minimum in build(base):
            out = base / f"out_{d.name}"
            t0 = time.time()
            p = subprocess.run([sys.executable, str(ROOT / "main.py"), "--data", str(d), "--output", str(out)],
                               capture_output=True, text=True, timeout=120)
            secs = time.time() - t0
            try:
                pred = pd.read_csv(out / "predictions.csv")["prediction"].values
                prof = json.loads((out / "profile.json").read_text())
                if d.name == "classification":
                    q, name = skm.f1_score(truth, pred, average="macro"), "macro-F1"
                elif d.name == "regression":
                    q, name = skm.r2_score(truth, pred), "R2"
                else:
                    q, name = skm.adjusted_rand_score(truth, pred), "ARI"
                good = p.returncode == 0 and prof["model"]["status"] == "ok" and q >= minimum and secs < 60
            except Exception as e:  # missing / broken output
                good, q, name = False, float("nan"), f"error {e}"
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  {d.name:15s} {name} {q:.3f}  {secs:5.1f}s")
            if not good:
                print(p.stderr[-2000:])
    print("smoke test passed" if ok else "smoke test FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
