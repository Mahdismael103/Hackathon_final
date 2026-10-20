"""Offline evaluation of clustering selection rules on the saved candidate pools (research script)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

OUT = Path(__file__).resolve().parents[1] / "outputs"


def balanced(r, n, k, frac=0.1):
    return r["minsize"] >= frac * n / k and r["ngroups"] == k


def consensus(cands):
    for r in cands:
        others = [o for o in cands if o is not r]
        r["_c"] = np.mean([ari(r["labels"], o["labels"]) for o in others]) if others else 0
    return max(cands, key=lambda r: r["_c"])


def rank_combo(cands, keys):
    df = pd.DataFrame([{k: r[k] for k in keys} for r in cands])
    if "db" in df:
        df["db"] = -df["db"]
    score = df.rank().mean(1).values
    return cands[int(np.argmax(score))]


RULES = {
    "silhouette_all": lambda c, n, k: max(c, key=lambda r: r["sil"]),
    "sil_balanced": lambda c, n, k: max([r for r in c if balanced(r, n, k)] or c, key=lambda r: r["sil"]),
    "consensus_all": lambda c, n, k: consensus(c),
    "consensus_balanced": lambda c, n, k: consensus([r for r in c if balanced(r, n, k)] or c),
    "consensus_no_avg_spec": lambda c, n, k: consensus([r for r in c if r["alg"] not in ("average", "spectral")]),
    "cons_bal_no_avg": lambda c, n, k: consensus([r for r in c if r["alg"] != "average" and balanced(r, n, k)] or c),
    "gmm_full_all": lambda c, n, k: next((r for r in c if r["name"] == "all/gmm_full"),
                                         next(r for r in c if r["name"] == "all/gmm_diag")),
    "bic_gmm_all_space": lambda c, n, k: min([r for r in c if r["space"] == "all" and r["bic"] is not None],
                                             key=lambda r: r["bic"]),
    "rank_sil_ch_db_bal": lambda c, n, k: rank_combo([r for r in c if balanced(r, n, k)] or c, ["sil", "ch", "db"]),
    "rank_sil_cons_bal": lambda c, n, k: rank_combo(_with_cons([r for r in c if balanced(r, n, k)] or c), ["sil", "_c"]),
}


def _with_cons(c):
    consensus(c)
    return c


def main():
    rows = []
    for p in sorted(OUT.glob("clu_*.pkl")):
        d = pd.read_pickle(p)
        y, res = d["y"], d["res"]
        n, k = len(y), len(np.unique(y))
        row = {"dataset": d["name"], "oracle": max(r["ari"] for r in res)}
        for rn, f in RULES.items():
            pick = f([dict(r) for r in res], n, k)
            row[rn] = pick["ari"]
        rows.append(row)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    print(t.round(3).to_string(index=False))
    print(t.select_dtypes("number").mean().sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
