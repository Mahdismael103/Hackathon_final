"""Round 2: richer GMM candidates + hybrid selection rules, evaluated against hidden labels."""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from automl.preprocess import Preparer  # noqa: E402
from automl.profiling import profile_dataset  # noqa: E402
from clu_suite import suite  # noqa: E402


def gmm_fit(Z, k, cov, reg, init_labels=None):
    g = GaussianMixture(k, covariance_type=cov, n_init=3, reg_covar=reg, random_state=0, max_iter=300).fit(Z)
    return g.predict(Z), g.bic(Z)


def candidates(X, k, n_onehot):
    n, d = X.shape
    res = {}
    t = time.time()
    res["kmeans"] = (KMeans(k, n_init=10, random_state=0).fit_predict(X), None)
    if n <= 10000:
        res["ward"] = (AgglomerativeClustering(k, linkage="ward").fit_predict(X), None)
    if n <= 3000:
        res["spectral"] = (SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15,
                                              assign_labels="cluster_qr", random_state=0).fit_predict(X), None)
    reg = 1e-2 if n_onehot else 1e-3
    q = min(d, 10)
    Zp = PCA(q, random_state=0).fit_transform(X) if d > 10 else X
    for cov in ("full", "tied", "diag", "spherical"):
        if d <= 20:
            res[f"gmm_{cov}"] = gmm_fit(X, k, cov, reg)
        if d > 10:
            res[f"pca_gmm_{cov}"] = gmm_fit(Zp, k, cov, reg)
    return res, Zp, time.time() - t


def main():
    rows = []
    for name, df, y, k in suite():
        prof, roles = profile_dataset(df, None, "clustering", None, {}, None, k)
        prep = Preparer(roles, clustering=True).fit(df)
        X = prep.transform_linear(df, onehot_weight=0.7)
        n_onehot = sum(1 for nm in prep.linear_names if "==" in nm)
        res, Zp, sec = candidates(X, k, n_onehot)
        a = {kk: ari(y, v[0]) for kk, v in res.items()}
        row = {"dataset": name, "d": X.shape[1], "sec": round(sec, 1), "oracle": max(a.values())}
        row.update({kk: a[kk] for kk in ("kmeans", "ward", "spectral") if kk in a})
        gm = {kk: v for kk, v in res.items() if "gmm" in kk}
        space = "gmm_" if X.shape[1] <= 20 else "pca_gmm_"
        bic_pick = min((kk for kk in gm if kk.startswith(space)), key=lambda kk: gm[kk][1])
        row["bic_pick"] = a[bic_pick]
        row["bic_name"] = bic_pick
        row["gmm_full_or_pca"] = a.get("gmm_full", a.get("pca_gmm_full"))
        # consensus among a compact, diverse pool
        pool = ["kmeans", "ward", bic_pick] + (["spectral"] if "spectral" in res else [])
        cons = {p: np.mean([ari(res[p][0], res[o][0]) for o in pool if o != p]) for p in pool}
        row["consensus4"] = a[max(cons, key=cons.get)]
        # hybrid: trust BIC-GMM unless it disagrees strongly with both kmeans and ward
        agree = max(ari(res[bic_pick][0], res["kmeans"][0]), ari(res[bic_pick][0], res.get("ward", res["kmeans"])[0]))
        row["agree_bic_km"] = agree
        row["hyb_gmm_unless_onehot"] = a["kmeans"] if n_onehot else row["bic_pick"]
        rows.append(row)
        print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    print(t.drop(columns=["bic_name"]).round(3).to_string(index=False))
    print(t.select_dtypes("number").drop(columns=["d", "sec"]).mean().sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
