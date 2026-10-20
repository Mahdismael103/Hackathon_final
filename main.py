"""Standard entry point.

    python main.py --data path/to/dataset_folder --output path/to/output_folder

The dataset folder is expected to hold a training table, an optional test table and a JSON
metadata file (task, target, feature types, n_clusters). Every value can be given explicitly:

    python main.py --train train.csv --test test.csv --task classification --target Churn --output out
    python main.py --train mall.csv --task clustering --n-clusters 5 --output out

Outputs (in --output): predictions.csv, profile.json and, for classification, probabilities.csv.
"""
import time

T0 = time.time()

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

# OpenMP (HistGradientBoosting, KMeans) and BLAS get slower beyond ~4 threads on small/medium
# tables because of thread start-up and oversubscription; tree forests use joblib threads instead.
def _cpus():
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


_THREADS = str(max(1, min(4, _cpus())))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

from automl import config  # noqa: E402  (light: no heavy imports)


def _early_time_limit() -> float:
    """--time-limit read straight from argv, so the watchdog can start before the heavy imports."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        try:
            if a == "--time-limit" and i + 1 < len(argv):
                return float(argv[i + 1])
            if a.startswith("--time-limit="):
                return float(a.split("=", 1)[1])
        except ValueError:
            break
    return config.TIME_LIMIT_SECONDS


def _start_watchdog(limit: float):
    """Hard stop shortly before the time limit.

    Started before pandas / numpy / scikit-learn are imported, because on a slow or swapping machine
    the imports alone can take many seconds. Outputs are written atomically (fallback first,
    improved versions later), so ending the process here always leaves a valid submission on disk.
    """
    def fire():
        print(f"[automl] watchdog: stopping at {time.time() - T0:.1f}s, keeping the outputs written so far",
              file=sys.stderr, flush=True)
        os._exit(0)

    timer = threading.Timer(max(1.0, limit - config.WATCHDOG_MARGIN_SECONDS - (time.time() - T0)), fire)
    timer.daemon = True
    timer.start()


if __name__ == "__main__":
    _start_watchdog(_early_time_limit())

from automl.api import solve  # noqa: E402
from automl.io_contract import load_task  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unknown-dataset AutoML (classification / regression / clustering)")
    p.add_argument("data", nargs="?", help="dataset folder (or a single table / metadata file)")
    p.add_argument("output_pos", nargs="?", help="output folder (positional alternative to --output)")
    p.add_argument("--data", dest="data_opt")
    p.add_argument("--train")
    p.add_argument("--test")
    p.add_argument("--meta", help="metadata JSON file")
    p.add_argument("--task", choices=["classification", "regression", "clustering"])
    p.add_argument("--target")
    p.add_argument("--n-clusters", "--k", dest="n_clusters", type=int)
    p.add_argument("--id-column")
    p.add_argument("--metric", help="selection metric, e.g. f1_macro, accuracy, rmse, r2")
    p.add_argument("--output", "-o")
    p.add_argument("--time-limit", type=float, default=config.TIME_LIMIT_SECONDS)
    # output-contract overrides, so the official format can be matched without code changes
    p.add_argument("--predictions-file", help=f"default {config.PREDICTIONS_FILE}")
    p.add_argument("--profile-file", help=f"default {config.PROFILE_FILE}")
    p.add_argument("--prediction-column", help=f"default {config.PREDICTION_COLUMN}")
    p.add_argument("--output-id-column", help="force the id column name in outputs")
    return p.parse_args(argv)


def _apply_overrides(a):
    for arg, attr in (("predictions_file", "PREDICTIONS_FILE"), ("profile_file", "PROFILE_FILE"),
                      ("prediction_column", "PREDICTION_COLUMN"), ("output_id_column", "OUTPUT_ID_COLUMN")):
        if getattr(a, arg):
            setattr(config, attr, getattr(a, arg))


def main(argv=None):
    a = parse_args(argv)
    if __name__ != "__main__":      # called from Python instead of the command line
        _start_watchdog(a.time_limit)
    _apply_overrides(a)
    data = a.data_opt or a.data
    try:
        spec = load_task(data=data, train=a.train, test=a.test, meta=a.meta, task=a.task, target=a.target,
                         n_clusters=a.n_clusters, id_column=a.id_column, metric=a.metric)
    except Exception:
        print("[automl] could not load the dataset:\n" + traceback.format_exc(), file=sys.stderr)
        return 1
    out = a.output or a.output_pos or str(Path("outputs") / spec.name)
    try:
        solve(spec, out, t0=T0, time_limit=a.time_limit)
    except Exception:  # solve() already guards modelling; this is the last line of defence
        print("[automl] unexpected error:\n" + traceback.format_exc(), file=sys.stderr)
        return 0 if (Path(out) / config.PREDICTIONS_FILE).exists() else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
