"""Round 3: stability-based selection and co-association ensembles."""
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
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from automl.preprocess import Preparer  # noqa: E402
from automl.profiling import profile_dataset  # noqa: E402
from clu_suite import suite  # noqa: E402


def methods(k, n, d, reg):
    m = {
        "kmeans": lambda Z: KMeans(k, n_init=10, random_state=0).fit_predict(Z),
        "ward": lambda Z: AgglomerativeClustering(k, linkage="ward").fit_predict(Z),
        "spectral": lambda Z: SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=max(10, int(np.log(len(Z)) * 2.5)),
                                                 assign_labels="cluster_qr", random_state=0).fit_predict(Z),
    }

    def gmm_bic(Z):
        Zg = PCA(10, random_state=0).fit_transform(Z) if Z.shape[1] > 20 else Z
        best = None
        for cov in ("full", "tied", "diag", "spherical"):
            g = GaussianMixture(k, covariance_type=cov, n_init=2, reg_covar=reg, random_state=0).fit(Zg)
            b = g.bic(Zg)
            if best is None or b < best[0]:
                best = (b, g.predict(Zg))
        return best[1]
    m["gmm_bic"] = gmm_bic
    m["gmm_full"] = lambda Z: GaussianMixture(k, covariance_type="full" if Z.shape[1] <= 20 else "diag", n_init=2,
                                              reg_covar=reg, random_state=0).fit(Z).predict(Z)
    return m


def stability(f, Z, base, rng, reps=3, frac=0.8):
    n = len(Z)
    s = []
    for _ in range(reps):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        try:
            lab = f(Z[idx])
        except Exception:
            return 0.0
        s.append(ari(base[idx], lab))
    return float(np.mean(s))


def coassoc(labelings, k):
    n = len(labelings[0])
    C = np.zeros((n, n))
    for lab in labelings:
        C += (lab[:, None] == lab[None, :])
    C /= len(labelings)
    return AgglomerativeClustering(k, metric="precomputed", linkage="average").fit_predict(1 - C)


def main():
    rows = []
    for name, df, y, k in suite():
        prof, roles = profile_dataset(df, None, "clustering", None, {}, None, k)
        prep = Preparer(roles, clustering=True).fit(df)
        X = prep.transform_linear(df, onehot_weight=0.7)
        n, d = X.shape
        n_onehot = sum(1 for nm in prep.linear_names if "==" in nm)
        M = methods(k, n, d, 1e-2 if n_onehot else 1e-3)
        rng = np.random.default_rng(0)
        labs, stab, t0 = {}, {}, time.time()
        for mn, f in M.items():
            labs[mn] = f(X)
            stab[mn] = stability(f, X, labs[mn], rng)
        a = {mn: ari(y, l) for mn, l in labs.items()}
        row = {"dataset": name, "oracle": max(a.values()), **a}
        row["stab_pick"] = a[max(stab, key=stab.get)]
        row["stab_name"] = max(stab, key=stab.get)
        row["coassoc"] = ari(y, coassoc(list(labs.values()), k)) if n <= 3000 else np.nan
        # consensus over the 5
        cons = {p: np.mean([ari(labs[p], labs[o]) for o in labs if o != p]) for p in labs}
        row["consensus5"] = a[max(cons, key=cons.get)]
        # stability x consensus
        sc = {p: stab[p] + cons[p] for p in labs}
        row["stab+cons"] = a[max(sc, key=sc.get)]
        for tol in (0.01, 0.02, 0.03, 0.05):
            top = max(stab.values())
            near = [p for p in labs if stab[p] >= top - tol]
            row[f"stab_tol{tol}"] = a[max(near, key=lambda p: cons[p])]
        row["sec"] = round(time.time() - t0, 1)
        rows.append(row)
        print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items()},
              {kk: round(v, 3) for kk, v in stab.items()}, flush=True)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    print(t.drop(columns=["stab_name"]).round(3).to_string(index=False))
    print(t.select_dtypes("number").drop(columns=["sec"]).mean().sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
