"""Score LightGBM + GNN ensembles on the validation weeks (same protocol as experiments.py).

Needs the GNN probabilities written by train_gnn.py (results/gnn/<name>_<mmdd>.parquet and
<name>_<mmdd>_th.parquet). LightGBM is retrained here with the same split.

    python src/ensemble_eval.py --data train.parquet --val ... --gnn gnn_notime gnn_bucket
"""
import argparse
from pathlib import Path

import pandas as pd

from features import ACTS, DAY
from models import LGB_PARAMS, LGB_PARAMS_REG, best_thresholds, binarize, feature_columns, fit_lgbm, predict_lgbm, score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", nargs="+", default=["2026-10-29 16:00", "2026-11-05 16:00", "2026-11-12 16:00"])
    ap.add_argument("--gnn", nargs="+", required=True)
    ap.add_argument("--gnn-dir", default="results/gnn")
    ap.add_argument("--out", default="results/ensemble.csv")
    ap.add_argument("--no-relational", action="store_true", help="LightGBM without the relational features")
    ap.add_argument("--reg-es", action="store_true", help="regularized LightGBM + early stopping on the threshold week")
    ap.add_argument("--seeds", type=int, default=1, help="LightGBM seeds to average")
    ap.add_argument("--gnn-avg", nargs="*", default=[],
                    help="NAME=run1,run2,...: one candidate = mean of several saved GNN runs (e.g. seeds)")
    a = ap.parse_args()
    data = pd.read_parquet(a.data)
    cols = feature_columns(data, ("temporal", "global") if a.no_relational else ("temporal", "relational", "global"))
    params = LGB_PARAMS_REG if a.reg_es else LGB_PARAMS
    rows = []
    for tv in map(pd.Timestamp, a.val):
        tr = data[data.t <= tv - 14 * DAY]
        th_rows, va = data[data.t == tv - 7 * DAY], data[data.t == tv]
        lg_th = lg = 0
        for seed in range(a.seeds):
            m = fit_lgbm(tr, cols, params={**params, "seed": seed}, valid=th_rows if a.reg_es else None)
            lg_th = lg_th + predict_lgbm(m, th_rows, cols) / a.seeds
            lg = lg + predict_lgbm(m, va, cols) / a.seeds
        probs = {"lgbm": (lg_th, lg)}
        def load(g):
            p = pd.read_parquet(Path(a.gnn_dir) / f"{g}_{tv:%m%d}.parquet")[ACTS]
            pth = pd.read_parquet(Path(a.gnn_dir) / f"{g}_{tv:%m%d}_th.parquet")[ACTS]
            p.index, pth.index = va.index, th_rows.index
            return pth, p

        for g in a.gnn:
            probs[g] = load(g)
        for spec in a.gnn_avg:
            name, runs = spec.split("=")
            loaded = [load(g) for g in runs.split(",")]
            probs[name] = (sum(x[0] for x in loaded) / len(loaded), sum(x[1] for x in loaded) / len(loaded))
        a.gnn = list(a.gnn) + [spec.split("=")[0] for spec in a.gnn_avg]
        cands = dict(probs)
        for g in a.gnn:
            cands[f"lgbm+{g}"] = ((probs["lgbm"][0] + probs[g][0]) / 2, (probs["lgbm"][1] + probs[g][1]) / 2)
        if len(a.gnn) > 1:
            n = len(a.gnn) + 1
            cands["lgbm+all_gnns"] = (sum(v[0] for v in probs.values()) / n, sum(v[1] for v in probs.values()) / n)
        for name, (pth, p) in cands.items():
            th = best_thresholds(pth, th_rows)
            s = score(binarize(p, th), va, va.github_user_id)
            rows.append({"val_t": str(tv), "model": name, "score": s["score"], "macro_f1": s["macro_f1"],
                         "micro_f1": s["micro_f1"], **{f"f1_{k}": v for k, v in s["per_event"].items()}})
            print(f"{tv:%m-%d} {name:36s} score {s['score']:.4f}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(a.out, index=False)
    print(r.groupby("model", sort=False)[["score", "macro_f1", "micro_f1"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
