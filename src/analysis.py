"""Threshold-free analysis of the feature groups (for Questions 2 and 3).

For each validation time tv the LightGBM models are trained on t <= tv - 14 days and their
probabilities at tv are scored with average precision (AP) per activity, for all test users
and separately for "warm" users (any activity in the 28 days before tv) and "cold" users (none).
Also writes the LightGBM gain importance per feature group.

    python src/analysis.py --data data/snapshots/train.parquet --val ... --out results
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import ACTS, DAY
from models import RELATIONAL, feature_columns, fit_lgbm, predict_lgbm


def group_of(c):
    if c.startswith("global"):
        return "global (data volume)"
    if c in RELATIONAL:
        return "relational"
    if c.startswith("cnt"):
        return "temporal: window counts"
    if c.startswith("decay"):
        return "temporal: decayed counts"
    if c.startswith("days_since"):
        return "temporal: recency"
    return "temporal: activity pattern"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", nargs="+", default=["2026-10-29 16:00", "2026-11-05 16:00", "2026-11-12 16:00"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    data = pd.read_parquet(a.data)
    allc = feature_columns(data)
    configs = {
        "counts only": [c for c in allc if c.startswith("cnt")],
        "+ decay, recency, pattern": feature_columns(data, ("temporal",)),
        "+ relational": feature_columns(data, ("temporal", "relational")),
        "+ global (all)": allc,
    }
    rows, imp = [], []
    for tv in map(pd.Timestamp, a.val):
        tr, va = data[data.t <= tv - 14 * DAY], data[data.t == tv]
        warm = va[[f"cnt28d_{k}" for k in ACTS]].sum(1) > 0
        for name, cols in configs.items():
            models = fit_lgbm(tr, cols)
            prob = predict_lgbm(models, va, cols)
            for k in ACTS:
                y = va[f"y_{k}"].to_numpy()
                for grp, m in (("all", np.ones(len(va), bool)), ("warm", warm.to_numpy()), ("cold", ~warm.to_numpy())):
                    if y[m].sum() > 0:
                        rows.append({"val_t": tv, "config": name, "activity": k, "users": grp,
                                     "ap": average_precision_score(y[m], prob[k].to_numpy()[m]),
                                     "pos": int(y[m].sum()), "n": int(m.sum())})
            if name == "+ global (all)":
                for k, mdl in models.items():
                    g = pd.Series(mdl.feature_importance("gain"), index=cols)
                    for grp, v in g.groupby(group_of).sum().items():
                        imp.append({"val_t": tv, "activity": k, "group": grp, "gain_share": v / g.sum()})
        print(tv, "done", flush=True)
    r = pd.DataFrame(rows)
    out = Path(a.out)
    r.to_csv(out / "analysis_ap.csv", index=False)
    pd.DataFrame(imp).to_csv(out / "importance.csv", index=False)
    piv = r[r.users == "all"].pivot_table(index="config", columns="activity", values="ap", sort=False)[ACTS]
    piv["mean"] = piv.mean(1)
    print("AP, all test users (mean over validation weeks)\n", piv.round(3).to_string())
    for grp in ("warm", "cold"):
        p = r[r.users == grp].pivot_table(index="config", columns="activity", values="ap", sort=False)[ACTS]
        p["mean"] = p.mean(1)
        print(f"\nAP, {grp} users\n", p.round(3).to_string())
    print("\nusers per group at each tv:", r[(r.config == "counts only") & (r.activity == "push")][["val_t", "users", "n", "pos"]].to_string())
    gi = pd.DataFrame(imp).pivot_table(index="group", columns="activity", values="gain_share")[ACTS]
    print("\ngain share by feature group\n", gi.round(3).to_string())
    summ = out / "summary"; summ.mkdir(exist_ok=True)
    # mean AP over the 6 activities and the validation weeks, per user group
    apt = r.groupby(["config", "users"]).ap.mean().unstack()[["all", "warm", "cold"]].reindex(list(configs))
    apt.to_csv(summ / "ap_table.csv")
    r[r.users == "cold"].pivot_table(index="config", columns="activity", values="ap", sort=False)[ACTS].to_csv(
        summ / "ap_cold_by_activity.csv")
    gi.to_csv(summ / "gain_share.csv")
    sizes = r[(r.config == r.config.iloc[0]) & (r.activity == "open_pr")][["val_t", "users", "n"]]
    sizes.to_csv(summ / "ap_group_sizes.csv", index=False)
    print("\n", apt.round(3).to_string())


if __name__ == "__main__":
    main()
