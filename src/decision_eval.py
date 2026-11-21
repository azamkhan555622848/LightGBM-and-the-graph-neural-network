"""Decision rules (probabilities -> 0/1) under changing data coverage, on the validation weeks.

  threshold  per-activity probability threshold that maximizes F1 on the threshold week
  count      predict the same number of positives as that threshold gave on the threshold week
  anchored   predict  n = (positives on the threshold week / users active in the last 7 days
             before its cut-off) x (users active in the last 7 days before the new cut-off),
             i.e. scale the number of positives with the activity GH Archive currently records

    python src/decision_eval.py --data train.parquet --gnn-avg gnn=gnn_count_agg,gnn_count_agg_s1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import ACTS, DAY
from models import LGB_PARAMS, best_thresholds, feature_columns, fit_lgbm, predict_lgbm, score

FOLDS = ["2026-10-29 16:00", "2026-11-05 16:00", "2026-11-12 16:00"]


def top_n(p, n):
    out = np.zeros(len(p), dtype=int)
    if n > 0:
        out[np.argsort(-p, kind="stable")[:n]] = 1
    return out


def decide(prob_th, th_rows, prob, rows, rule):
    th = best_thresholds(prob_th, th_rows)
    pred = {}
    for k in ACTS:
        n_th = int((prob_th[k] >= th[k]).sum())
        if rule == "threshold":
            pred[k] = (prob[k] >= th[k]).astype(int).to_numpy()
        elif rule == "count":
            pred[k] = top_n(prob[k].to_numpy(), n_th)
        elif rule == "anchored":
            act_th = int((th_rows[f"cnt7d_{k}"] > 0).sum())
            act = int((rows[f"cnt7d_{k}"] > 0).sum())
            pred[k] = top_n(prob[k].to_numpy(), int(round(n_th * act / max(act_th, 1))))
    return pd.DataFrame(pred, index=prob.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", nargs="+", default=FOLDS)
    ap.add_argument("--gnn-dir", default="results/gnn_ablation")
    ap.add_argument("--gnn-avg", nargs="*", default=[], help="NAME=run1,run2: GNN candidate = mean of runs")
    ap.add_argument("--out", default="results/decision.csv")
    a = ap.parse_args()
    data = pd.read_parquet(a.data)
    cols = feature_columns(data)
    rows = []
    for tv in map(pd.Timestamp, a.val):
        tr = data[data.t <= tv - 14 * DAY]
        th_rows, va = data[data.t == tv - 7 * DAY], data[data.t == tv]
        m = fit_lgbm(tr, cols, params=LGB_PARAMS)
        cands = {"lgbm": (predict_lgbm(m, th_rows, cols), predict_lgbm(m, va, cols))}
        for spec in a.gnn_avg:
            name, runs = spec.split("=")
            ps, pths = [], []
            for g in runs.split(","):
                p = pd.read_parquet(Path(a.gnn_dir) / f"{g}_{tv:%m%d}.parquet")[ACTS]; p.index = va.index
                pt = pd.read_parquet(Path(a.gnn_dir) / f"{g}_{tv:%m%d}_th.parquet")[ACTS]; pt.index = th_rows.index
                ps.append(p); pths.append(pt)
            cands[name] = (sum(pths) / len(pths), sum(ps) / len(ps))
            cands[f"lgbm+{name}"] = ((cands["lgbm"][0] + cands[name][0]) / 2, (cands["lgbm"][1] + cands[name][1]) / 2)
        for name, (pth, p) in cands.items():
            for rule in ("threshold", "count", "anchored"):
                s = score(decide(pth, th_rows, p, va, rule), va, va.github_user_id)
                rows.append({"val_t": str(tv), "model": name, "rule": rule, "score": s["score"],
                             "macro_f1": s["macro_f1"], "micro_f1": s["micro_f1"],
                             **{f"f1_{k}": v for k, v in s["per_event"].items()}})
                print(f"{tv:%m-%d} {name:28s} {rule:9s} score {s['score']:.4f}  push {s['per_event']['push']:.3f}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(a.out, index=False)
    print(r.pivot_table(index=["model", "rule"], columns="val_t", values="score", sort=False).assign(
        mean=lambda x: x.mean(1)).round(4).to_string())


if __name__ == "__main__":
    main()
