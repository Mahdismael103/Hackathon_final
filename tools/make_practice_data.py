"""Generate local practice datasets that mimic the hackathon scope (until the official ones arrive).

    python tools/make_practice_data.py            # writes practice/<name>/ and practice_solutions/

Each dataset folder holds train.csv, test.csv (supervised only) and meta.json.
Hidden labels go to practice_solutions/<name>.csv (never read by main.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "practice"
SOL = ROOT / "practice_solutions"
rng = np.random.default_rng(42)


def _messify(df, miss=0.05, cols=None, const=True, dup=True, id_col=True):
    df = df.copy()
    cols = cols if cols is not None else list(df.columns)
    for c in cols:
        m = rng.random(len(df)) < miss
        df.loc[m, c] = np.nan
    if const:
        df["constant_flag"] = 1
    if dup and len(cols):
        df[f"{cols[0]}_copy"] = df[cols[0]]
    if id_col:
        df.insert(0, "row_id", np.arange(100000, 100000 + len(df)))
    return df


def _save_supervised(name, X, y, task, target="target", test_frac=0.3, ftypes=None, metric=None):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    n = len(X)
    idx = rng.permutation(n)
    n_te = int(n * test_frac)
    te, tr = idx[:n_te], idx[n_te:]
    train = X.iloc[tr].copy()
    train[target] = np.asarray(y)[tr]
    test = X.iloc[te].copy()
    train.to_csv(d / "train.csv", index=False)
    test.to_csv(d / "test.csv", index=False)
    meta = {"task": task, "target": target, "id_column": "row_id" if "row_id" in X.columns else None}
    if ftypes:
        meta["feature_types"] = ftypes
    if metric:
        meta["metric"] = metric
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    SOL.mkdir(exist_ok=True)
    ids = test["row_id"].values if "row_id" in test.columns else np.arange(len(test))
    pd.DataFrame({"id": ids, "label": np.asarray(y)[te]}).to_csv(SOL / f"{name}.csv", index=False)


def _save_clustering(name, X, labels, k, ftypes=None):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    X.to_csv(d / "train.csv", index=False)
    meta = {"task": "clustering", "n_clusters": int(k), "id_column": "row_id" if "row_id" in X.columns else None}
    if ftypes:
        meta["feature_types"] = ftypes
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    SOL.mkdir(exist_ok=True)
    ids = X["row_id"].values if "row_id" in X.columns else np.arange(len(X))
    pd.DataFrame({"id": ids, "label": labels}).to_csv(SOL / f"{name}.csv", index=False)


# ------------------------------------------------------------------------------ classification
def churn_like(n=5000):
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    pay = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    monthly = np.round(20 + 50 * (internet == "Fiber optic") + 25 * (internet == "DSL") + rng.normal(0, 12, n), 2)
    total = np.round(monthly * tenure + rng.normal(0, 30, n), 2)
    logit = (-1.2 + 1.4 * (contract == "Month-to-month") - 1.2 * (contract == "Two year") - 0.035 * tenure
             + 0.9 * (internet == "Fiber optic") + 0.5 * (pay == "Electronic check") + 0.4 * senior)
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    total_s = total.astype(str)
    total_s[tenure == 0] = " "  # like the real Telco data
    X = pd.DataFrame({"customerID": [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJ'), 5))}{i}"
                                     for i in range(n)],
                      "gender": rng.choice(["Male", "Female"], n), "SeniorCitizen": senior, "tenure": tenure,
                      "Contract": contract, "InternetService": internet, "PaymentMethod": pay,
                      "MonthlyCharges": monthly, "TotalCharges": total_s})
    X.insert(0, "row_id", np.arange(n))
    _save_supervised("cls_churn_like", X, y, "classification", target="Churn")


def maintenance_like(n=10000):
    typ = rng.choice(["L", "M", "H"], n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300, 2, n)
    proc = air + 10 + rng.normal(0, 1, n)
    rpm = rng.normal(1540, 180, n)
    torque = np.clip(rng.normal(40, 10, n), 3, None)
    wear = rng.integers(0, 250, n)
    power = torque * rpm * 2 * np.pi / 60
    fail = ((power < 3500) | (power > 9000) | ((proc - air) < 8.6) & (rpm < 1380) | (wear * torque > 11000)
            | (rng.random(n) < 0.002)).astype(int)
    X = pd.DataFrame({"UDI": np.arange(1, n + 1), "Product ID": [f"{t}{rng.integers(10000, 99999)}{i}" for i, t in enumerate(typ)],
                      "Type": typ, "Air temperature [K]": air, "Process temperature [K]": proc,
                      "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear})
    X.insert(0, "row_id", np.arange(n))
    _save_supervised("cls_maintenance_like", X, fail, "classification", target="Machine failure")


def multiclass_noisy(n=3000):
    Xa, y = skd.make_classification(n, n_features=20, n_informative=6, n_redundant=3, n_classes=4,
                                    n_clusters_per_class=2, weights=[0.5, 0.25, 0.15, 0.1], random_state=1)
    X = pd.DataFrame(Xa * rng.choice([1, 10, 1000, 0.01], 20), columns=[f"f{i}" for i in range(20)])
    X["color"] = rng.choice(["red", "green", "blue"], n)
    X = _messify(X, miss=0.08)
    _save_supervised("cls_multiclass_noisy", X, np.array(["A", "B", "C", "D"])[y], "classification")


# ------------------------------------------------------------------------------ regression
def house_like(n=2000):
    hood = rng.choice([f"N{i}" for i in range(25)], n)
    hood_eff = {f"N{i}": rng.normal(0, 0.25) for i in range(25)}
    area = np.exp(rng.normal(7.3, 0.35, n))
    qual = rng.integers(1, 11, n)
    year = rng.integers(1890, 2011, n)
    garage = rng.choice([0, 1, 2, 3], n, p=[0.06, 0.25, 0.55, 0.14])
    style = rng.choice(["1Story", "2Story", "1.5Fin", "SLvl"], n)
    logp = (11.0 + 0.55 * np.log(area / 1500) + 0.11 * qual + 0.003 * (year - 1970) + 0.06 * garage
            + np.vectorize(hood_eff.get)(hood) + rng.normal(0, 0.12, n))
    X = pd.DataFrame({"Id": np.arange(1, n + 1), "Neighborhood": hood, "GrLivArea": area.round(),
                      "OverallQual": qual, "YearBuilt": year, "GarageCars": garage, "HouseStyle": style,
                      "LotFrontage": rng.normal(70, 20, n).round(), "MiscNoise": rng.normal(0, 1, n)})
    X.loc[rng.random(n) < 0.17, "LotFrontage"] = np.nan
    X.loc[rng.random(n) < 0.05, "GarageCars"] = np.nan
    X.insert(0, "row_id", np.arange(n))
    _save_supervised("reg_house_like", X, np.exp(logp).round(), "regression", target="SalePrice")


def bike_like(n=8000):
    t = pd.Timestamp("2011-01-01") + pd.to_timedelta(rng.integers(0, 2 * 365 * 24, n), unit="h")
    hour = t.hour.values
    season = ((t.month.values % 12) // 3) + 1
    temp = 15 + 10 * np.sin((t.dayofyear.values - 100) / 365 * 2 * np.pi) + rng.normal(0, 3, n)
    hum = np.clip(rng.normal(60, 18, n), 0, 100)
    weather = rng.choice([1, 2, 3, 4], n, p=[0.65, 0.25, 0.09, 0.01])
    workday = (t.dayofweek.values < 5).astype(int)
    peak = workday * (np.exp(-((hour - 8) ** 2) / 2) * 300 + np.exp(-((hour - 17.5) ** 2) / 3) * 350)
    leisure = (1 - workday) * np.exp(-((hour - 14) ** 2) / 12) * 250
    lam = np.clip((40 + peak + leisure) * (1 + 0.03 * (temp - 15)) * (1 - 0.25 * (weather - 1))
                  * (1 + 0.3 * (t.year.values - 2011)) * (1 - 0.004 * (hum - 60)), 1, None)
    y = rng.poisson(lam)
    X = pd.DataFrame({"datetime": t.strftime("%Y-%m-%d %H:%M:%S"), "season": season, "holiday": 0,
                      "workingday": workday, "weather": weather, "temp": temp.round(2),
                      "atemp": (temp + rng.normal(0, 1, n)).round(2), "humidity": hum.round(),
                      "windspeed": np.abs(rng.normal(12, 8, n)).round(3)})
    X.insert(0, "row_id", np.arange(n))
    _save_supervised("reg_bike_like", X, y, "regression", target="count")


def friedman_noisy(n=2500):
    Xa, y = skd.make_friedman1(n, n_features=15, noise=1.0, random_state=3)
    X = pd.DataFrame(Xa * rng.choice([1, 100, 0.001], 15), columns=[f"x{i}" for i in range(15)])
    X["grp"] = rng.choice(["a", "b", "c", "d"], n)
    y = y + 3 * (X["grp"] == "b").values
    X = _messify(X, miss=0.05)
    _save_supervised("reg_friedman_noisy", X, y, "regression")


# ------------------------------------------------------------------------------ clustering
def mall_like(n=400):
    centers = np.array([[25, 80], [55, 50], [85, 15], [85, 85], [25, 20]])
    lab = rng.integers(0, 5, n)
    inc = centers[lab, 0] + rng.normal(0, 7, n)
    spend = np.clip(centers[lab, 1] + rng.normal(0, 7, n), 1, 100)
    X = pd.DataFrame({"CustomerID": np.arange(1, n + 1), "Annual Income (k$)": inc.round(),
                      "Spending Score (1-100)": spend.round()})
    X.insert(0, "row_id", np.arange(n))
    _save_clustering("clu_mall_like", X, lab, 5)


def blobs_noise(n=3000):
    Xa, lab = skd.make_blobs(n, n_features=4, centers=4, cluster_std=1.2, random_state=7)
    noise = rng.normal(0, 1, (n, 6)) * 3
    X = pd.DataFrame(np.hstack([Xa * [1, 50, 0.1, 1000], noise]), columns=[f"v{i}" for i in range(10)])
    X = _messify(X, miss=0.03)
    _save_clustering("clu_blobs_noise", X, lab, 4)


def mixed_aniso(n=2000):
    Xa, lab = skd.make_blobs(n, n_features=3, centers=3, cluster_std=[1.0, 2.0, 0.6], random_state=11)
    Xa = Xa @ np.array([[0.8, -0.4, 0.1], [0.3, 0.9, 0.0], [0.0, 0.2, 1.0]])
    cat = np.where(rng.random(n) < 0.8, np.array(["alpha", "beta", "gamma"])[lab], rng.choice(["alpha", "beta", "gamma"], n))
    X = pd.DataFrame(Xa, columns=["a", "b", "c"])
    X["segment_hint"] = cat
    X["irrelevant"] = rng.uniform(0, 1, n)
    X.insert(0, "row_id", np.arange(n))
    _save_clustering("clu_mixed_aniso", X, lab, 3)


# ------------------------------------------------------------------------------ real sklearn sets
def real_sets():
    bc = skd.load_breast_cancer(as_frame=True)
    _save_supervised("real_cls_breast_cancer", bc.data, bc.target, "classification")
    w = skd.load_wine(as_frame=True)
    _save_supervised("real_cls_wine", w.data, w.target, "classification")
    db = skd.load_diabetes(as_frame=True, scaled=False)
    _save_supervised("real_reg_diabetes", db.data, db.target, "regression")
    ir = skd.load_iris(as_frame=True)
    _save_clustering("real_clu_iris", ir.data, ir.target.values, 3)
    _save_clustering("real_clu_wine", w.data, w.target.values, 3)


def supervised_suite():
    dg = skd.load_digits(as_frame=True)
    _save_supervised("sx_cls_digits", dg.data, dg.target, "classification")
    ir = skd.load_iris(as_frame=True)
    _save_supervised("sx_cls_iris", ir.data, ir.target, "classification")

    Xa, y = skd.make_classification(5000, 15, n_informative=8, n_redundant=2, n_clusters_per_class=3,
                                    flip_y=0.03, class_sep=0.8, random_state=21)
    _save_supervised("sx_cls_hard_nonlinear", pd.DataFrame(Xa).add_prefix("f"), y, "classification")

    Xa, y = skd.make_classification(8000, 12, n_informative=5, weights=[0.95], flip_y=0.01, random_state=22)
    X = _messify(pd.DataFrame(Xa).add_prefix("z"), miss=0.03)
    _save_supervised("sx_cls_imbalanced_5pct", X, y, "classification")

    n = 6000
    city = rng.choice([f"city_{i}" for i in range(300)], n)
    eff = {f"city_{i}": rng.normal(0, 1.2) for i in range(300)}
    x1 = rng.normal(size=n)
    logit = np.vectorize(eff.get)(city) + 0.8 * x1 + rng.normal(0, .5, n)
    X = pd.DataFrame({"city": city, "x1": x1, "x2": rng.normal(size=n), "seg": rng.choice(list("ABC"), n)})
    _save_supervised("sx_cls_highcard_signal", X, (logit > 0.3).astype(int), "classification")

    Xa, y = skd.make_friedman2(2000, noise=20, random_state=23)
    _save_supervised("sx_reg_friedman2", pd.DataFrame(Xa).add_prefix("p"), y, "regression")
    Xa, y = skd.make_friedman3(2000, noise=0.1, random_state=24)
    _save_supervised("sx_reg_friedman3", pd.DataFrame(Xa).add_prefix("p"), y, "regression")

    Xa, y = skd.make_regression(600, 100, n_informative=10, noise=10, random_state=25)
    _save_supervised("sx_reg_sparse_linear_100", pd.DataFrame(Xa).add_prefix("r"), y, "regression")

    n = 6000
    shop = rng.choice([f"shop_{i}" for i in range(400)], n)
    seff = {f"shop_{i}": rng.normal(0, 10) for i in range(400)}
    x1 = rng.uniform(0, 10, n)
    y = np.vectorize(seff.get)(shop) + 3 * np.sin(x1) * x1 + rng.normal(0, 2, n)
    _save_supervised("sx_reg_highcard_signal", pd.DataFrame({"shop": shop, "x1": x1, "x2": rng.normal(size=n)}), y,
                     "regression")

    n = 4000
    Xa = rng.normal(size=(n, 6))
    y = np.exp(1.5 + 0.6 * Xa[:, 0] - 0.4 * Xa[:, 1] * Xa[:, 2] + 0.3 * rng.normal(size=n))
    _save_supervised("sx_reg_lognormal_target", pd.DataFrame(Xa).add_prefix("q"), y, "regression")


def clustering_suite():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from clu_suite import suite
    for name, df, y, k in suite():
        _save_clustering(f"cs_{name}", df.reset_index(drop=True), y, k)


if __name__ == "__main__":
    for f in (churn_like, maintenance_like, multiclass_noisy, house_like, bike_like, friedman_noisy,
              mall_like, blobs_noise, mixed_aniso, real_sets, supervised_suite, clustering_suite):
        f()
        print("generated", f.__name__)
