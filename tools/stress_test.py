"""Edge-case / robustness tests: every case must finish < 60 s and produce valid outputs.

    python tools/stress_test.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# short paths: Windows has a 260-character path limit
BASE = Path(tempfile.gettempdir()) / "automl_stress" / "data"
OUTS = Path(tempfile.gettempdir()) / "automl_stress" / "out"
rng = np.random.default_rng(0)


def case(name, train, test=None, meta=None, sep=",", extra_args=()):
    d = BASE / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    train.to_csv(d / "train.csv", index=False, sep=sep)
    if test is not None:
        test.to_csv(d / "test.csv", index=False, sep=sep)
    if meta is not None:
        (d / "meta.json").write_text(json.dumps(meta))
    return d, test if test is not None else train, list(extra_args)


def cases():
    n = 100_000
    X = pd.DataFrame(rng.normal(size=(n, 40)), columns=[f"c{i}" for i in range(40)])
    X["cat"] = rng.choice(list("abcdefgh"), n)
    y = (X["c0"] * X["c1"] + np.sin(X["c2"] * 3) + (X["cat"] == "c") + rng.normal(0, .5, n) > .5).astype(int)
    tr, te = X.iloc[:80000].copy(), X.iloc[80000:].copy()
    tr["y"] = y[:80000]
    yield case("big_classification_100k", tr, te, {"task": "classification", "target": "y"})

    yr = X["c0"] * 3 + X["c1"] ** 2 + rng.normal(0, 1, n)
    tr2 = X.iloc[:80000].copy(); tr2["y"] = yr[:80000]
    yield case("big_regression_100k", tr2, te, {"task": "regression", "target": "y"})
    yield case("big_clustering_100k", X.iloc[:, :10], None, {"task": "clustering", "n_clusters": 6})

    m = 3000
    hc = pd.DataFrame({"zip": rng.integers(0, 2000, m).astype(str), "city": rng.choice([f"city{i}" for i in range(400)], m),
                       "desc": [" ".join(rng.choice(["good", "bad", "fast", "slow", "red", "big"], 6)) for _ in range(m)],
                       "num": rng.normal(size=m)})
    hc["label"] = np.where(hc["num"] + hc["desc"].str.count("good") * .5 > .5, "pos", "neg")
    hc.loc[rng.random(m) < .05, "label"] = np.nan
    yield case("high_card_and_text", hc.iloc[:2400], hc.iloc[2400:].drop(columns="label"),
               {"task": "classification", "target": "label"})

    cat = pd.DataFrame({f"q{i}": rng.choice(["yes", "no", "maybe"], 1500) for i in range(8)})
    cat["out"] = (cat["q0"] == "yes").astype(int) * 2 + (cat["q1"] == "no").astype(int)
    yield case("all_categorical_multiclass", cat.iloc[:1200], cat.iloc[1200:].drop(columns="out"),
               {"task": "classification", "target": "out"})
    yield case("categorical_clustering", cat.drop(columns="out"), None, {"task": "clustering", "n_clusters": 3})

    rare = pd.DataFrame(rng.normal(size=(600, 5)), columns=list("abcde"))
    rare["t"] = np.where(rare["a"] > 0, "x", "y")
    rare.loc[0, "t"] = "z"  # a class with a single example
    yield case("rare_single_class", rare.iloc[:500], rare.iloc[500:].drop(columns="t"),
               {"problem_type": "multiclass", "label": "t"})

    semi = pd.DataFrame(rng.normal(size=(800, 4)), columns=["w", "x", "y_", "z"])
    semi["target"] = semi["w"] * 2 - 5 + rng.normal(0, .1, 800)
    yield case("semicolon_listtypes_neg_target", semi.iloc[:600], semi.iloc[600:].drop(columns="target"),
               {"task_type": "regression", "target_column": "target",
                "features": [{"name": c, "type": "float"} for c in ["w", "x", "y_", "z"]]}, sep=";")

    nometa = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    nometa["label"] = (nometa["a"] > 0).astype(int)
    yield case("no_meta_no_test_cli_args", nometa, None, None,
               extra_args=["--task", "classification", "--target", "label"])

    wide = pd.DataFrame(rng.normal(size=(400, 300)), columns=[f"g{i}" for i in range(300)])
    wide["y"] = wide["g0"] + wide["g1"] + rng.normal(0, .3, 400)
    yield case("wide_300_features_small_n", wide.iloc[:300], wide.iloc[300:].drop(columns="y"),
               {"task": "regression", "target": "y"})

    tiny = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3, "b": list("ababababab") * 3})
    tiny["y"] = [0, 1] * 15
    yield case("tiny_30_rows", tiny.iloc[:24], tiny.iloc[24:].drop(columns="y"), {"task": "classification", "target": "y"})


def main():
    bad = 0
    for d, pred_frame, extra in cases():
        out = OUTS / d.name
        t0 = time.time()
        p = subprocess.run([sys.executable, str(ROOT / "main.py"), "--data", str(d), "--output", str(out), *extra],
                           capture_output=True, text=True, timeout=180)
        dt = time.time() - t0
        status, msg = "OK", ""
        try:
            pred = pd.read_csv(out / "predictions.csv")
            prof = json.loads((out / "profile.json").read_text())
            if len(pred) != len(pred_frame):
                status, msg = "FAIL", f"rows {len(pred)} != {len(pred_frame)}"
            elif pred["prediction"].isna().any():
                status, msg = "FAIL", "NaN predictions"
            elif prof.get("model", {}).get("status") != "ok":
                status, msg = "FALLBACK", str(prof.get("model"))[:300]
            else:
                msg = str(prof["model"].get("ensemble_weights") or prof["model"].get("chosen"))
        except Exception as e:
            status, msg = "FAIL", f"{e} | {p.stderr[-1500:]}"
        if dt > 60:
            status = "SLOW"
        bad += status != "OK"
        print(f"{status:8s} {d.name:34s} {dt:6.1f}s  {msg}")
    print("all good" if bad == 0 else f"{bad} problem case(s)")


if __name__ == "__main__":
    main()
