"""Why a probability threshold fails when GH Archive coverage changes (push, week of Nov 12).

Writes results/summary/push_diag.json for the slides.

    python src/push_diag.py --data data/snapshots/train.parquet
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import DAY
from models import LGB_PARAMS, feature_columns

R = Path(__file__).resolve().parents[1] / "results"


def f1_at(p, y, c):
    pred = p >= c
    tp = int((pred & (y == 1)).sum())
    return 2 * tp / max(int(pred.sum()) + int(y.sum()), 1), int(pred.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tv", default="2026-11-12 16:00")
    a = ap.parse_args()
    d = pd.read_parquet(a.data)
    cols = feature_columns(d)
    tv = pd.Timestamp(a.tv)
    tr, th, va = d[d.t <= tv - 14 * DAY], d[d.t == tv - 7 * DAY], d[d.t == tv]
    m = lgb.train(LGB_PARAMS, lgb.Dataset(tr[cols].astype("float32"), label=tr.y_push), 400)
    pth, pva = m.predict(th[cols].astype("float32")), m.predict(va[cols].astype("float32"))
    yth, yva = th.y_push.to_numpy(), va.y_push.to_numpy()
    best_th = max((f1_at(pth, yth, c)[0], c) for c in np.unique(np.quantile(pth, np.linspace(0.5, 0.999, 300))))[1]
    f1_thr, n_sel = f1_at(pva, yva, best_th)
    f1_best = max(f1_at(pva, yva, c)[0] for c in np.unique(np.quantile(pva, np.linspace(0.5, 0.999, 300))))
    sep = d[d.t == d.t.min() + 2 * DAY]
    de = pd.read_csv(R / "decision_final.csv")
    ens = de[(de.model == "lgbm+gnn") & (de.val_t.str.startswith(str(tv.date())))].set_index("rule")
    gs = pd.read_csv(R / "summary" / "gain_share.csv", index_col=0)
    out = {"push_rate_sep": round(100 * sep.y_push.mean()), "push_rate_th": round(100 * yth.mean(), 1),
           "push_rate_va": round(100 * yva.mean(), 1), "n_selected": n_sel, "n_pos": int(yva.sum()),
           "threshold": float(best_th), "f1_threshold": f1_thr, "f1_best": f1_best,
           "ens_push_threshold": float(ens.loc["threshold", "f1_push"]),
           "ens_push_count": float(ens.loc["count", "f1_push"]),
           "global_gain_push": float(100 * gs.loc["global (data volume)", "push"])}
    (R / "summary" / "push_diag.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
