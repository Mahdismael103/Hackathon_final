"""Feature relevance for supervised tasks: two complementary tests, a feature is relevant if either fires.

1. Mutual information against a shuffled-target null: catches any *marginal* dependence,
   including non-monotonic ones. A column is relevant when its MI with the target beats the
   largest MI the estimator reports for permuted targets (with a small floor).
2. Permutation importance of a quick gradient-boosting model on a held-out split: catches features
   that only matter through *interactions* (e.g. machine failure = f(torque x speed)), which
   marginal MI misses. Relevant when the score drop is clearly above the permutation noise.

Verified on practice data with known ground truth: Friedman #1 (x5..x14 pure noise) and
the multiclass / blob sets with injected noise columns.
"""
from __future__ import annotations

import numpy as np

from . import config

MAX_ROWS = 3000
FLOOR = 0.005
PERM_RELATIVE = 0.03
CHUNK = 50               # columns per MI batch (lets the time budget be checked often)
PERM_MAX_COLUMNS = 200   # permutation importance costs columns x repeats predictions


def original_name(name: str) -> str:
    """Map a prepared column ("city==a", "when__hour", "x__missing") back to its table column."""
    for sep in ("==", "__"):
        if sep in name:
            return name.split(sep)[0]
    return name


def _mi_relevance(Xs, ys, task, rng, deadline):
    """MI per column in chunks; columns not reached before the deadline get NaN (= unknown)."""
    import time
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression  # lazy: heavy import
    d = Xs.shape[1]
    fn = mutual_info_classif if task == "classification" else mutual_info_regression
    n_null = 2 if d <= 100 else 1
    perms = [rng.permutation(ys) for _ in range(n_null)]
    mi = np.full(d, np.nan)
    null_max = 0.0
    for start in range(0, d, CHUNK):
        if time.time() > deadline:
            break
        cols = slice(start, min(d, start + CHUNK))
        Xc = Xs[:, cols]
        disc = np.array([len(np.unique(Xc[:, j])) <= 2 for j in range(Xc.shape[1])])
        kw = dict(discrete_features=disc, n_neighbors=3, random_state=config.RANDOM_STATE)
        mi[cols] = fn(Xc, ys, **kw)
        for yp in perms:
            null_max = max(null_max, float(np.max(fn(Xc, yp, **kw))))
    thr = max(FLOOR, null_max)
    return mi, thr


def _permutation_relevance(Xt, cat_mask, yt, task, rng, n_repeats=5, deadline=None):
    """Score drop when each tree-view column is permuted, on a 30% hold-out of a quick HGB."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    n = len(yt)
    idx = rng.permutation(n)
    cut = int(0.7 * n)
    tr, va = idx[:cut], idx[cut:]
    cats = cat_mask if cat_mask is not None and cat_mask.any() else None
    Model = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
    # one tree per class per iteration: keep the total number of trees bounded for many classes
    n_cls = len(np.unique(yt)) if task == "classification" else 1
    # small tables need far fewer boosting rounds (and every round costs a tree per class)
    base_iters = 60 if n < 500 else (100 if n < 2000 else 150)
    iters = base_iters if n_cls <= 5 else max(30, int(2 * base_iters / n_cls))
    m = Model(max_iter=iters, learning_rate=0.1 if n_cls <= 5 else 0.2, max_leaf_nodes=15,
              categorical_features=cats, random_state=config.RANDOM_STATE).fit(Xt[tr], yt[tr])
    if n_cls > 5:
        n_repeats = min(n_repeats, 3)
    if deadline is not None:
        # cost = columns x repeats predictions: shrink the repeats to fit the time left, or skip
        import time
        t = time.time()
        m.predict(Xt[va])
        per_pass = (time.time() - t) * Xt.shape[1]
        left = deadline - time.time()
        n_repeats = min(n_repeats, int(left / max(per_pass, 1e-6)))
        if n_repeats < 2:
            raise TimeoutError("not enough time for permutation importance")
    r = permutation_importance(m, Xt[va], yt[va], n_repeats=n_repeats, random_state=config.RANDOM_STATE)
    return r.importances_mean, r.importances_std


def feature_relevance(X: np.ndarray, y: np.ndarray, names: list[str], task: str,
                      tree_view=None, tree_names: list[str] | None = None, budget_seconds: float = 6.0) -> dict:
    """X / names: linear view. tree_view: optional (X_tree, cat_mask) for the permutation test."""
    n, d = X.shape
    if d == 0:
        return {"relevant_mask": np.zeros(0, bool), "scores": {}, "irrelevant_features": [], "threshold": 0.0}
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(n, MAX_ROWS, replace=False) if n > MAX_ROWS else np.arange(n)
    y = np.asarray(y)
    if task == "classification":
        y = np.unique(y.astype(str), return_inverse=True)[1]
    else:
        y = y.astype(float)

    import time
    t0 = time.time()
    # wide tables: fewer rows for the estimators (cost grows with rows x columns)
    if d > 200 and len(idx) > 1000:
        idx = idx[:1000]
    mi, thr = _mi_relevance(X[idx], y[idx], task, rng, t0 + 0.6 * budget_seconds)
    relevant: dict[str, bool] = {}
    mi_feat: dict[str, float] = {}
    for nm, v in zip(names, mi):
        o = original_name(nm)
        if np.isnan(v):            # not evaluated in time: keep it (the safe choice)
            relevant[o] = True
            continue
        mi_feat[o] = max(mi_feat.get(o, 0.0), float(v))
        relevant[o] = relevant.get(o, False) or v > thr

    perm_feat: dict[str, float] = {}
    if (tree_view is not None and tree_names is not None and len(idx) >= 60 and d <= PERM_MAX_COLUMNS
            and time.time() < t0 + 0.6 * budget_seconds):
        try:
            Xt, cat_mask = tree_view
            imp, sd = _permutation_relevance(Xt[idx], cat_mask, y[idx], task, rng,
                                             n_repeats=5 if Xt.shape[1] <= 50 else 3,
                                             deadline=t0 + budget_seconds)
            # clearly above permutation noise AND at least 3% of the strongest feature's importance
            floor = max(1e-3, PERM_RELATIVE * float(np.max(imp)) if len(imp) else 0.0)
            for nm, m_, s_ in zip(tree_names, imp, sd):
                o = original_name(nm)
                perm_feat[o] = max(perm_feat.get(o, 0.0), float(m_))
                if m_ > max(2.0 * s_, floor):
                    relevant[o] = True
        except Exception:
            pass  # MI alone is still a valid (more conservative) answer

    irrelevant = sorted(f for f, r in relevant.items() if not r)
    mask = np.array([relevant.get(original_name(nm), True) for nm in names], dtype=bool)
    return {"relevant_mask": mask, "scores": {k: round(v, 5) for k, v in mi_feat.items()},
            "permutation_importance": {k: round(v, 5) for k, v in perm_feat.items()},
            "irrelevant_features": irrelevant, "threshold": round(thr, 5)}


SCREEN_MIN_COLUMNS = 200  # wider tables are screened before model search
SCREEN_KEEP = 150


def screen_features(X_tree: np.ndarray, y: np.ndarray, task: str, tree_names: list[str]) -> dict:
    """Rank the columns of a very wide table with one fast Extra-Trees model and keep the top ones.

    Returns the original features to keep (all columns derived from a kept feature are kept)."""
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    rng = np.random.default_rng(config.RANDOM_STATE)
    n = len(y)
    idx = rng.choice(n, 2000, replace=False) if n > 2000 else np.arange(n)
    y = np.asarray(y)
    if task == "classification":
        y = np.unique(y.astype(str), return_inverse=True)[1]
    Model = ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor
    m = Model(n_estimators=150, max_features="sqrt", min_samples_leaf=2, n_jobs=config.N_JOBS,
              random_state=config.RANDOM_STATE).fit(X_tree[idx], y[idx].astype(float) if task != "classification" else y[idx])
    imp = m.feature_importances_
    order = np.argsort(-imp)
    keep_feats: list[str] = []
    for j in order:
        o = original_name(tree_names[j])
        if o not in keep_feats:
            keep_feats.append(o)
        if len(keep_feats) >= SCREEN_KEEP:
            break
    all_feats = sorted({original_name(nm) for nm in tree_names})
    return {"keep": keep_feats, "screened_out": [f for f in all_feats if f not in set(keep_feats)],
            "importance": {original_name(tree_names[j]): round(float(imp[j]), 6) for j in order[:SCREEN_KEEP]}}
