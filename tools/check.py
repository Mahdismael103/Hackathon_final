"""Local checker: runs main.py on every practice dataset like the assessment agent would.

    python tools/check.py                       # all folders in practice/
    python tools/check.py practice/reg_bike_like

Reports wall time (limit 60 s), output validity and quality against practice_solutions/.
Classification: macro-F1 + accuracy | Regression: R2 + RMSE | Clustering: ARI + NMI.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parents[1]


def score(task, y, p):
    if task == "classification":
        y, p = y.astype(str), p.astype(str)
        return {"f1_macro": skm.f1_score(y, p, average="macro"), "accuracy": skm.accuracy_score(y, p)}
    if task == "regression":
        y, p = y.astype(float), p.astype(float)
        return {"r2": skm.r2_score(y, p), "rmse": float(np.sqrt(skm.mean_squared_error(y, p)))}
    return {"ari": skm.adjusted_rand_score(y, p), "nmi": skm.normalized_mutual_info_score(y, p)}


MAIN = ROOT / "main.py"


# Run outputs go to the temp folder: a synced folder (OneDrive, Dropbox) would try to upload every file
# while the timing is being measured. Only the summary CSV is written into the project's outputs/.
RUN_DIR = Path(tempfile.gettempdir()) / "automl_check"


def run_one(d: Path):
    out = RUN_DIR / ("main" if MAIN == ROOT / "main.py" else "other_version") / d.name
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(MAIN), "--data", str(d), "--output", str(out)],
                          capture_output=True, text=True, timeout=120)
    wall = time.time() - t0
    meta = json.loads((d / "meta.json").read_text())
    task = meta["task"]
    row = {"dataset": d.name, "task": task, "seconds": round(wall, 2), "ok": proc.returncode == 0}
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        return row
    try:
        pred = pd.read_csv(out / "predictions.csv")
        prof = json.loads((out / "profile.json").read_text())
    except (FileNotFoundError, ValueError) as e:   # missing / broken output counts as a failure
        row.update(ok=False, error=f"{type(e).__name__}: {e}"[:160])
        return row
    row["model_status"] = prof.get("model", {}).get("status")
    row["peak_mb"] = prof.get("peak_memory_mb")
    sol_path = ROOT / "practice_solutions" / f"{d.name}.csv"
    if sol_path.exists():
        sol = pd.read_csv(sol_path)
        if len(sol) != len(pred):
            row["error"] = f"row count {len(pred)} != {len(sol)}"
            return row
        merged = sol.merge(pred, left_on="id", right_on=pred.columns[0], how="left")
        row.update({k: round(v, 4) for k, v in score(task, merged["label"].values, merged["prediction"].values).items()})
    m = prof.get("model", {})
    row["chosen"] = m.get("chosen") or ",".join(f"{k}:{v:.2f}" for k, v in m.get("ensemble_weights", {}).items())
    return row


def main():
    args = sys.argv[1:]
    tag, only = None, None
    if "--tag" in args:
        i = args.index("--tag"); tag = args[i + 1]; del args[i:i + 2]
    if "--main" in args:  # compare another version of the program on the same datasets
        global MAIN
        i = args.index("--main"); MAIN = Path(args[i + 1]).resolve(); del args[i:i + 2]
    if "--only" in args:  # classification | regression | clustering
        i = args.index("--only"); only = args[i + 1]; del args[i:i + 2]
    targets = [Path(a) for a in args] or sorted(p for p in (ROOT / "practice").iterdir() if p.is_dir())
    if only:
        targets = [d for d in targets if json.loads((d / "meta.json").read_text())["task"] == only]
    rows = []
    for d in targets:
        r = run_one(d)
        rows.append(r)
        print(json.dumps(r))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print()
    print(df.to_string(index=False))
    print("\nmean quality per task:")
    for task, col in (("classification", "f1_macro"), ("regression", "r2"), ("clustering", "ari")):
        sub = df[df["task"] == task]
        if len(sub) and col in sub:
            print(f"  {task:15s} {col:9s} {sub[col].mean():.4f}   (n={len(sub)}, mean time {sub['seconds'].mean():.1f}s)")
    if tag:
        (ROOT / "outputs").mkdir(exist_ok=True)
        df.to_csv(ROOT / "outputs" / f"results_{tag}.csv", index=False)
    slow = df[df["seconds"] > 60]
    print(f"\nmax time {df['seconds'].max():.1f}s; over-limit datasets: {len(slow)}; failures: {int((~df['ok']).sum())}")


if __name__ == "__main__":
    main()
