"""Final model: train on every labelled snapshot, pick thresholds, predict the test week.

The same recipe as the validation experiments:
  * train on the snapshots t <= t_last - 7 days (their label weeks end before t_last),
  * on the last labelled snapshot t_last, find per activity the probability threshold that
    maximizes F1 and the number n_k of users it selects,
  * predict the test snapshot (2026-11-21 16:00 UTC, features up to the end of the data) and
    mark the n_k users with the highest probability as 1 (rank-based decision: robust to the
    week-to-week changes of GH Archive coverage that shift the probabilities).
The probabilities are the average of LightGBM and the GNN (several seeds each).

    python src/predict.py --db data/db --data data/snapshots --out output_413540002.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from features import ACTS, DAY, load_db
from decision_eval import decide
from models import LGB_PARAMS, best_thresholds, feature_columns, fit_lgbm, predict_lgbm

# the final configuration, chosen on the 3 validation weeks (see results/summary)
FINAL = {
    "lgbm": {"groups": ["temporal", "relational", "global"]},
    "gnn": {"time_enc": "none", "shortcuts": False, "global_ctx": False, "layers": 2, "aggr": "meansum",
            "lookback": 56, "epochs": 12},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--data", required=True, help="folder with train.parquet and test.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="lgbm,gnn", help="comma list: lgbm, gnn")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--save-dir", default=None, help="also save probabilities and thresholds here")
    a = ap.parse_args()

    train = pd.read_parquet(Path(a.data) / "train.parquet")
    test = pd.read_parquet(Path(a.data) / "test.parquet")
    t_last = train.t.max()                     # last snapshot whose label week is complete
    fit_rows = train[train.t <= t_last - 7 * DAY]
    th_rows = train[train.t == t_last]
    t_test = test.t.iloc[0]
    print(f"train snapshots {fit_rows.t.min()} .. {fit_rows.t.max()} ({fit_rows.t.nunique()}), "
          f"thresholds on {t_last}, predict at {t_test} (feature cut-off {test.t_feat.iloc[0]})", flush=True)
    names = a.models.split(",")
    probs = {}
    if "lgbm" in names:
        cols = feature_columns(train, tuple(FINAL["lgbm"]["groups"]))
        p_th = p_te = 0
        for s in range(a.seeds):
            m = fit_lgbm(fit_rows, cols, params={**LGB_PARAMS, "seed": s})
            p_th = p_th + predict_lgbm(m, th_rows, cols) / a.seeds
            p_te = p_te + predict_lgbm(m, test, cols) / a.seeds
        probs["lgbm"] = (p_th, p_te)
        print("LightGBM done", flush=True)
    if "gnn" in names:
        from train_gnn import Trainer
        T = load_db(a.db)
        g = FINAL["gnn"]
        both = pd.concat([train, test.assign(**{f"y_{k}": 0 for k in ACTS})], ignore_index=True)
        times = sorted(fit_rows.t.unique())
        p_th = p_te = 0
        for s in range(a.seeds):
            tr = Trainer(T, both, time_enc=g["time_enc"], shortcuts=g["shortcuts"], seed=s, layers=g["layers"],
                         aggr=g["aggr"], lookback=g["lookback"], global_ctx=g["global_ctx"])
            tr.fit(times, epochs=g["epochs"])
            pt = tr.predict(t_last); pt.index = th_rows.index
            pe = tr.predict(t_test); pe.index = test.index
            p_th, p_te = p_th + pt / a.seeds, p_te + pe / a.seeds
            print(f"GNN seed {s} done", flush=True)
        probs["gnn"] = (p_th, p_te)
    prob_th = sum(v[0] for v in probs.values()) / len(probs)
    prob_te = sum(v[1] for v in probs.values()) / len(probs)
    th = best_thresholds(prob_th, th_rows)
    n_pos = {k: int((prob_th[k] >= th[k]).sum()) for k in ACTS}
    pred = decide(prob_th, th_rows, prob_te, test, "count")
    out = pd.DataFrame({"UserID": test.github_user_id.values, **{k: pred[k].values for k in ACTS}})
    out.to_csv(a.out, index=False)
    print("thresholds on the last labelled week:", {k: round(float(v), 4) for k, v in th.items()})
    print("users predicted 1 per activity (= count chosen on the last labelled week):", n_pos)
    print("predicted positive rate:", out[ACTS].mean().round(3).to_dict())
    print("last labelled week positive rate:", th_rows[[f"y_{k}" for k in ACTS]].mean().round(3).to_dict())
    if a.save_dir:
        d = Path(a.save_dir); d.mkdir(parents=True, exist_ok=True)
        for name, (pt, pe) in probs.items():
            pe.assign(UserID=test.github_user_id.values).to_parquet(d / f"test_prob_{name}.parquet")
        (d / "decision.json").write_text(json.dumps({"thresholds": {k: float(v) for k, v in th.items()},
                                                      "n_positive": n_pos}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
