"""Central knobs. Change these at kickoff to match the official contract / scoring rules."""

# Hard limit per dataset given by the organisers is 60 s. We aim to finish well before it,
# measured from process start (imports included).
TIME_LIMIT_SECONDS = 60.0
SAFETY_MARGIN_SECONDS = 12.0
# The watchdog in main.py ends the process this many seconds before the hard limit, whatever
# is running; outputs on disk are always complete.
WATCHDOG_MARGIN_SECONDS = 3.0
# Speed also earns points: stop *evaluating new candidates* after this many seconds from start
# (the final refit may still use time up to the hard limit minus the safety margin).
SOFT_SEARCH_SECONDS = 22.0
# ...and stop earlier once this many consecutive candidates fail to improve the ensemble.
SEARCH_PATIENCE = 3
# The model search always gets at least this long (still capped by the hard deadline).
MIN_SEARCH_SECONDS = 12.0
# Time allowed for the irrelevant-feature analysis (it is advisory; modelling must not starve).
RELEVANCE_BUDGET_SECONDS = 5.0

# Metric used for *model selection*. "auto" picks a sensible default per task.
#   classification: f1_macro | accuracy | balanced_accuracy | roc_auc | log_loss
#   regression:     rmse | mae | r2 | rmsle
DEFAULT_CLASSIFICATION_METRIC = "f1_macro"
DEFAULT_REGRESSION_METRIC = "rmse"

# Rows used for cross-validated model selection on very large tables.
MAX_SELECTION_ROWS = 30000

# Output file names / columns (adjust to the official contract if it differs).
PREDICTIONS_FILE = "predictions.csv"
PROBABILITIES_FILE = "probabilities.csv"
PROFILE_FILE = "profile.json"
PREDICTION_COLUMN = "prediction"
DEFAULT_ID_COLUMN = "id"
# Force the id column name in outputs (None = keep the dataset's own id column name).
OUTPUT_ID_COLUMN = None
# Clustering with a separate test table: labels for the test rows go to PREDICTIONS_FILE, and labels
# for every row (train rows first, then test rows) go here, in case the contract expects all rows.
ALL_ROWS_PREDICTIONS_FILE = "predictions_all_rows.csv"

RANDOM_STATE = 0


def available_cpus() -> int:
    """CPUs this process may actually use (respects affinity / container limits, unlike os.cpu_count)."""
    import os
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        n = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):  # Linux
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count()
    return max(1, n or 1)


N_JOBS = available_cpus()
