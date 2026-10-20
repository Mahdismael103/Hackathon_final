"""Feature preparation fitted on training rows only.

Two views of the same data are produced:
* tree view   - raw numbers (NaN kept, tree models handle it), categoricals as ordinal codes
                with a mask for native categorical support in HistGradientBoosting.
* linear view - imputed + missing indicators, clipped, log-transformed when skewed, standardised;
                categoricals one-hot encoded (top categories) plus frequency encoding.
                Used by linear models, kNN, SVM and clustering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profiling import parse_datetime, to_numeric

MAX_NATIVE_CATS = 250
TE_MIN_CARDINALITY = 10   # target-encode categoricals with more levels than this
TE_SMOOTHING = 20.0
MISSING = "__missing__"


def _cat(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna(MISSING).astype(object)


class Preparer:
    def __init__(self, roles: dict, max_onehot: int = 20, clustering: bool = False):
        self.roles = dict(roles)
        self.max_onehot = max_onehot
        self.clustering = clustering

    # ------------------------------------------------------------------ numeric block
    def _numeric_frame(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        parts = {}
        for c, r in self.roles.items():
            if r == "numeric":
                parts[str(c)] = to_numeric(df[c]).values
            elif r == "datetime":
                d = parse_datetime(df[c])
                feats = {
                    "year": d.dt.year, "month": d.dt.month, "day": d.dt.day,
                    "dayofweek": d.dt.dayofweek, "hour": d.dt.hour, "dayofyear": d.dt.dayofyear,
                    "minute": d.dt.minute,
                    "elapsed_days": (d - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0,
                }
                for k, v in feats.items():
                    parts[f"{c}__{k}"] = v.astype(float).values
                parts[f"{c}__is_weekend"] = (d.dt.dayofweek >= 5).astype(float).where(d.notna()).values
                # cyclical encodings: 23:00 is next to 00:00, December next to January
                for part, period in (("hour", 24.0), ("dayofweek", 7.0), ("month", 12.0)):
                    ang = 2 * np.pi * feats[part].astype(float).values / period
                    parts[f"{c}__{part}_sin"], parts[f"{c}__{part}_cos"] = np.sin(ang), np.cos(ang)
            elif r == "text":
                s = df[c].astype("string")
                parts[f"{c}__len"] = s.str.len().astype(float).values
                parts[f"{c}__words"] = s.str.split().str.len().astype(float).values
        frame = pd.DataFrame(parts, index=df.index)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        if fit:
            keep = [c for c in frame.columns if frame[c].nunique(dropna=True) > 1]
            self.num_names = keep
        return frame.reindex(columns=self.num_names)

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, y=None, task: str | None = None) -> "Preparer":
        num = self._numeric_frame(df, fit=True)
        self.medians = num.median()
        self.lo = num.quantile(0.001)
        self.hi = num.quantile(0.999)
        clipped = num.clip(self.lo, self.hi, axis=1)
        self.log_cols = []
        for c in num.columns:
            v = clipped[c].dropna()
            if len(v) > 10 and v.min() >= 0 and v.nunique() > 10 and abs(v.skew()) > 2:
                self.log_cols.append(c)
        lin = self._linear_numeric(num, fitted=False)
        self.mu = lin.mean()
        self.sd = lin.std().replace(0, 1).fillna(1)
        self.miss_cols = [c for c in num.columns if num[c].isna().any()] if not self.clustering else []

        self.cat_cols = [c for c, r in self.roles.items() if r in ("categorical", "text")]
        self.cat_codes, self.cat_freq, self.cat_top = {}, {}, {}
        n = len(df)
        for c in self.cat_cols:
            s = _cat(df[c])
            vc = s.value_counts()
            self.cat_codes[c] = {v: i for i, v in enumerate(vc.index)}
            self.cat_freq[c] = (vc / n).to_dict()
            min_count = max(2, int(0.005 * n))
            self.cat_top[c] = [v for v in vc.index[: self.max_onehot] if vc[v] >= min_count]
            f = s.map(self.cat_freq[c]).astype(float)
            self.cat_freq_stats = getattr(self, "cat_freq_stats", {})
            self.cat_freq_stats[c] = (float(f.mean()), float(f.std()) + 1e-9)
        self.tree_cat_native = [c for c in self.cat_cols if len(self.cat_codes[c]) <= MAX_NATIVE_CATS]
        self._fit_target_encoding(df, y, task)
        return self

    # ------------------------------------------------------------------ target encoding
    def _fit_target_encoding(self, df, y, task):
        """Smoothed mean-target encoding for high-cardinality categoricals.

        Training rows get out-of-fold encodings (5 folds) so models never see their own target."""
        self.te_cols, self.te_maps, self.te_train, self.te_std, self.te_prior = [], {}, {}, {}, None
        if y is None or self.clustering or task not in ("classification", "regression"):
            return
        cols = [c for c in self.cat_cols if len(self.cat_codes[c]) > TE_MIN_CARDINALITY]
        if not cols:
            return
        y = np.asarray(y)
        if task == "regression":
            T = ((y.astype(float) - y.astype(float).mean()) / (y.astype(float).std() + 1e-12))[:, None]
        else:
            classes, yi = np.unique(y.astype(str), return_inverse=True)
            if len(classes) > 10:
                return
            T = np.eye(len(classes))[yi]
            if len(classes) == 2:
                T = T[:, 1:]
        n = len(T)
        self.te_prior = T.mean(0)
        self.te_cols = cols
        rng = np.random.default_rng(0)
        folds = rng.integers(0, 5, n)
        for c in cols:
            s = _cat(df[c]).values
            self.te_maps[c] = self._te_table(s, T)
            oof = np.empty_like(T)
            for f in range(5):
                tr, va = folds != f, folds == f
                table = self._te_table(s[tr], T[tr])
                oof[va] = self._te_apply(s[va], table)
            self.te_train[c] = oof
            self.te_std[c] = oof.std(0) + 1e-9
        self.n_fit = n

    def _te_table(self, s, T):
        frame = pd.DataFrame(T, index=pd.Index(s, name="k"))
        g = frame.groupby(level=0)
        cnt = g.size().values[:, None]
        prior = T.mean(0)
        enc = (g.sum().values + TE_SMOOTHING * prior) / (cnt + TE_SMOOTHING)
        return dict(zip(g.size().index, enc)), prior

    @staticmethod
    def _te_apply(s, table):
        mapping, prior = table
        return np.array([mapping.get(v, prior) for v in s], dtype=float).reshape(len(s), -1)

    def _te_block(self, df, train):
        out, names = [], []
        for c in self.te_cols:
            if train and len(df) == self.n_fit:
                E = self.te_train[c]
            else:
                E = self._te_apply(_cat(df[c]).values, self.te_maps[c])
            out.append(E)
            names += [f"{c}__te{j}" for j in range(E.shape[1])]
        return out, names

    def _linear_numeric(self, num: pd.DataFrame, fitted: bool = True) -> pd.DataFrame:
        x = num.clip(self.lo, self.hi, axis=1)
        x = x.fillna(self.medians)
        for c in self.log_cols:
            x[c] = np.log1p(x[c].clip(lower=0))
        if fitted:
            x = (x - self.mu) / self.sd
        return x

    # ------------------------------------------------------------------ transforms
    def transform_tree(self, df: pd.DataFrame, train: bool = False):
        num = self._numeric_frame(df)
        blocks = [num.values.astype(float)]
        mask = [False] * num.shape[1]
        names = list(num.columns)
        for c in self.cat_cols:
            s = _cat(df[c])
            codes = s.map(self.cat_codes[c]).astype(float).values  # unseen -> NaN
            blocks.append(codes[:, None])
            native = c in self.tree_cat_native
            mask.append(native)
            names.append(f"{c}__code")
            if not native:
                blocks.append(s.map(self.cat_freq[c]).fillna(0.0).astype(float).values[:, None])
                mask.append(False)
                names.append(f"{c}__freq")
        te, te_names = self._te_block(df, train)
        blocks += te
        mask += [False] * len(te_names)
        names += te_names
        X = np.hstack(blocks) if blocks else np.zeros((len(df), 0))
        self.tree_names = names
        return X, np.array(mask, dtype=bool)

    def transform_linear(self, df: pd.DataFrame, onehot_weight: float = 1.0, train: bool = False):
        num = self._numeric_frame(df)
        blocks = [self._linear_numeric(num).values.astype(float)]
        names = list(num.columns)
        for c in self.miss_cols:
            blocks.append(num[c].isna().astype(float).values[:, None])
            names.append(f"{c}__missing")
        for c in self.cat_cols:
            s = _cat(df[c])
            top = self.cat_top[c]
            if top:
                oh = np.zeros((len(df), len(top)))
                idx = {v: i for i, v in enumerate(top)}
                pos = s.map(idx)
                ok = pos.notna().values
                oh[np.where(ok)[0], pos[ok].astype(int).values] = 1.0
                blocks.append(oh * onehot_weight)
                names += [f"{c}=={v}" for v in top]
            if len(self.cat_codes[c]) > self.max_onehot and not self.clustering:
                f = s.map(self.cat_freq[c]).fillna(0.0).astype(float).values
                fm, fs = self.cat_freq_stats[c]
                blocks.append(((f - fm) / fs)[:, None])
                names.append(f"{c}__freq")
        te, te_names = self._te_block(df, train)
        for c, E in zip(self.te_cols, te):
            blocks.append((E - self.te_prior) / self.te_std[c])
        names += te_names
        X = np.hstack(blocks) if blocks else np.zeros((len(df), 0))
        self.linear_names = names
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
