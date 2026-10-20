"""Compare clustering candidate pools and selection rules against hidden labels (research script)."""
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
from sklearn.metrics import (adjusted_rand_score as ari, calinski_harabasz_score, davies_bouldin_score,  # noqa: E402
                             silhouette_score)
from sklearn.mixture import GaussianMixture  # noqa: E402

from automl.clustering import _select_features  # noqa: E402
from automl.preprocess import Preparer  # noqa: E402
from automl.profiling import profile_dataset  # noqa: E402
from clu_suite import suite  # noqa: E402


def pool(X, k):
    n, d = X.shape
    spaces = {"all": X}
    keep = _select_features(X, k, 4)
    if 0 < len(keep) < d:
        spaces["sel"] = X[:, keep]
    if d > 10:
        spaces["pca"] = PCA(0.9, random_state=0).fit_transform(X)
    res = []
    for sn, Z in spaces.items():
        algs = {
            "kmeans": lambda: KMeans(k, n_init=10, random_state=0).fit_predict(Z),
            "gmm_full": lambda: GaussianMixture(k, covariance_type="full", n_init=3, reg_covar=1e-4, random_state=0).fit(Z),
            "gmm_diag": lambda: GaussianMixture(k, covariance_type="diag", n_init=3, reg_covar=1e-4, random_state=0).fit(Z),
            "ward": lambda: AgglomerativeClustering(k, linkage="ward").fit_predict(Z),
            "average": lambda: AgglomerativeClustering(k, linkage="average").fit_predict(Z),
            "spectral": lambda: SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15,
                                                   assign_labels="cluster_qr", random_state=0).fit_predict(Z),
        }
        for an, f in algs.items():
            if Z.shape[1] > 30 and an == "gmm_full":
                continue
            t = time.time()
            try:
                r = f()
            except Exception:
                continue
            bic = None
            if hasattr(r, "predict"):
                bic = r.bic(Z)
                r = r.predict(Z)
            lab = np.asarray(r)
            if len(np.unique(lab)) < 2:
                continue
            idx = np.random.RandomState(0).choice(n, min(n, 3000), replace=False)
            res.append(dict(name=f"{sn}/{an}", space=sn, alg=an, labels=lab, bic=bic,
                            sil=silhouette_score(Z[idx], lab[idx]), ch=calinski_harabasz_score(Z, lab),
                            db=davies_bouldin_score(Z, lab), sec=time.time() - t,
                            ngroups=len(np.unique(lab)), minsize=np.bincount(lab).min()))
    return res


def main():
    rows = []
    for name, df, y, k in suite():
        prof, roles = profile_dataset(df, None, "clustering", None, {}, None, k)
        prep = Preparer(roles, clustering=True).fit(df)
        X = prep.transform_linear(df, onehot_weight=0.7)
        res = pool(X, k)
        for r in res:
            r["ari"] = ari(y, r["labels"])
        # consensus: mean agreement with all other candidates
        for r in res:
            r["consensus"] = np.mean([ari(r["labels"], o["labels"]) for o in res if o is not r])
        row = {"dataset": name, "n": len(df), "k": k, "oracle": max(r["ari"] for r in res)}
        best_by = lambda key, sign=1: max(res, key=lambda r: sign * r[key])
        row["sil"] = best_by("sil")["ari"]
        row["consensus"] = best_by("consensus")["ari"]
        row["kmeans_all"] = next(r["ari"] for r in res if r["name"] == "all/kmeans")
        row["gmm_full_all"] = next((r["ari"] for r in res if r["name"] == "all/gmm_full"), np.nan)
        row["ward_all"] = next(r["ari"] for r in res if r["name"] == "all/ward")
        row["sel_kmeans"] = next((r["ari"] for r in res if r["name"] == "sel/kmeans"), row["kmeans_all"])
        row["best_name"] = max(res, key=lambda r: r["ari"])["name"]
        row["sil_pick"] = best_by("sil")["name"]
        row["cons_pick"] = best_by("consensus")["name"]
        rows.append(row)
        print({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in row.items()}, flush=True)
        pd.to_pickle({"name": name, "y": y, "res": [{k2: v for k2, v in r.items()} for r in res]},
                     Path(__file__).resolve().parents[1] / "outputs" / f"clu_{name}.pkl")
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.drop(columns=["best_name", "sil_pick", "cons_pick"]).round(3).to_string(index=False))
    print(t.select_dtypes("number").mean().round(4).to_string())


if __name__ == "__main__":
    (Path(__file__).resolve().parents[1] / "outputs").mkdir(exist_ok=True)
    main()
