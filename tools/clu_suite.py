"""A wide clustering benchmark (in memory): real labelled datasets with labels hidden + synthetic shapes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import datasets as skd


def suite(seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for name, loader in [("iris", skd.load_iris), ("wine", skd.load_wine),
                         ("breast_cancer", skd.load_breast_cancer), ("digits", skd.load_digits)]:
        b = loader(as_frame=True)
        out.append((name, b.data.copy(), b.target.values, len(np.unique(b.target))))

    X, y = skd.make_blobs(1500, 4, centers=4, cluster_std=1.3, random_state=1)
    df = pd.DataFrame(np.hstack([X * [1, 30, .1, 500], rng.normal(0, 1, (1500, 6))]))
    out.append(("blobs_4inf_6noise", df.add_prefix("v"), y, 4))

    X, y = skd.make_blobs(1200, 2, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    out.append(("varied_density", pd.DataFrame(X, columns=["a", "b"]), y, 3))

    X, y = skd.make_blobs(1200, 2, centers=3, random_state=170)
    X = X @ np.array([[0.6, -0.64], [-0.41, 0.85]])
    out.append(("anisotropic", pd.DataFrame(X, columns=["a", "b"]), y, 3))

    X, y = skd.make_moons(1000, noise=0.06, random_state=0)
    out.append(("moons", pd.DataFrame(X, columns=["a", "b"]), y, 2))

    X, y = skd.make_circles(1000, noise=0.05, factor=0.45, random_state=0)
    out.append(("circles", pd.DataFrame(X, columns=["a", "b"]), y, 2))

    X, y = skd.make_blobs(2000, 50, centers=6, cluster_std=4.0, random_state=3)
    out.append(("highdim_50", pd.DataFrame(X).add_prefix("h"), y, 6))

    X, y = skd.make_blobs(2000, 5, centers=5, cluster_std=1.5, random_state=4)
    df = pd.DataFrame(np.hstack([X, rng.uniform(-10, 10, (2000, 15))])).add_prefix("u")
    out.append(("blobs_5inf_15uniform", df, y, 5))

    X, y = skd.make_classification(2000, 8, n_informative=4, n_redundant=2, n_classes=3, n_clusters_per_class=1,
                                   class_sep=2.0, random_state=5)
    out.append(("classif_as_clusters", pd.DataFrame(X).add_prefix("c"), y, 3))

    n = 1500
    y = rng.integers(0, 3, n)
    df = pd.DataFrame({"x1": rng.normal(y * 3, 1), "x2": rng.normal(-y * 2, 1.2),
                       "cat": np.where(rng.random(n) < .7, np.array(["p", "q", "r"])[y], rng.choice(["p", "q", "r"], n)),
                       "cat_noise": rng.choice(["a", "b", "c", "d"], n)})
    df.loc[rng.random(n) < .05, "x1"] = np.nan
    out.append(("mixed_cat_missing", df, y, 3))

    centers = np.array([[25, 80], [55, 50], [85, 15], [85, 85], [25, 20]])
    y = rng.integers(0, 5, 400)
    df = pd.DataFrame({"Gender": rng.choice(["Male", "Female"], 400), "Age": rng.integers(18, 70, 400),
                       "Income": centers[y, 0] + rng.normal(0, 7, 400), "Spending": centers[y, 1] + rng.normal(0, 7, 400)})
    out.append(("mall_with_age_gender", df, y, 5))

    X, y = skd.make_blobs([600, 150, 50], 3, cluster_std=1.0, random_state=8)
    out.append(("imbalanced_sizes", pd.DataFrame(X).add_prefix("i"), y, 3))

    X, y = skd.make_blobs(1000, 3, centers=4, cluster_std=1.0, random_state=9)
    X = np.exp(X / 4)  # skewed positive features
    out.append(("skewed_lognormal", pd.DataFrame(X).add_prefix("s"), y, 4))
    return out
