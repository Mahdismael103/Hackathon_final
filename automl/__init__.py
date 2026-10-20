"""Unknown-dataset AutoML package.

One-call API (the same pipeline as ``python main.py``):

    import automl
    profile = automl.run("path/to/dataset_folder", "path/to/output_folder")
    profile = automl.run(train="train.csv", test="test.csv", task="regression", target="SalePrice",
                         output="out")
"""


def run(data=None, output="outputs/run", *, train=None, test=None, meta=None, task=None, target=None,
        n_clusters=None, id_column=None, metric=None, time_limit=None):
    """Load a dataset, solve its task, write predictions.csv + profile.json and return the profile dict."""
    import time

    from .api import solve
    from .io_contract import load_task

    t0 = time.time()
    spec = load_task(data=data, train=train, test=test, meta=meta, task=task, target=target,
                     n_clusters=n_clusters, id_column=id_column, metric=metric)
    return solve(spec, output, t0=t0, time_limit=time_limit)
