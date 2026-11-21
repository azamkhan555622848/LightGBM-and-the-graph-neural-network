"""Time-split experiments: baselines and LightGBM ablations, scored with the course's Score.

For a validation time tv:  train on snapshots t <= tv - 14 days  (their label weeks end before
tv - 7 days), pick the thresholds on the snapshot tv - 7 days, and score the snapshot tv.
This is exactly what the final run does (train <= t_last - 7 days, thresholds on t_last,
predict t_last + 7 days + gap). No label week used for training or thresholds overlaps the
validation week.

    python src/experiments.py --data data/snapshots/train.parquet --out results/lgbm.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from decision_eval import decide
from features import ACTS, DAY
from models import (LGB_PARAMS_REG, best_thresholds, best_thresholds_joint, binarize, feature_columns,
                    fit_lgbm, predict_lgbm, score)

FOLDS = ["2026-10-29 16:00", "2026-11-05 16:00", "2026-11-12 16:00"]   # 3 non-overlapping weeks


def copy_last(df, days=7):
    return pd.DataFrame({k: (df[f"cnt{days}d_{k}"] > 0).astype(int) for k in ACTS}, index=df.index)


def run_lgbm(data, tv, cols, train_from=None, joint=False, fit_data=None, early_stop=False, params=None):
    """fit_data: rows to train / pick thresholds on, if not `data` (e.g. built without the gap).
    early_stop: choose the number of trees on the threshold week (no validation-week labels)."""
    src = data if fit_data is None else fit_data
    tr = src[src.t <= tv - 14 * DAY]
    if train_from is not None:
        tr = tr[tr.t >= pd.Timestamp(train_from)]
    th_rows, va = src[src.t == tv - 7 * DAY], data[data.t == tv]
    kw = {"params": params} if params is not None else {}
    models = fit_lgbm(tr, cols, valid=th_rows if early_stop else None, **kw)
    pick = best_thresholds_joint if joint else best_thresholds
    prob_th = predict_lgbm(models, th_rows, cols)
    th = pick(prob_th, th_rows)
    prob = predict_lgbm(models, va, cols)
    return binarize(prob, th), prob, (prob_th, th_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="train.parquet from dataset.py")
    ap.add_argument("--val", nargs="+", default=FOLDS, help="validation times (UTC)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--nogap-data", default=None, help="the same snapshots built without the feature gap")
    ap.add_argument("--rules", default="both", choices=["threshold", "both"],
                    help="decision rules to score: probability threshold, and also the rank-based count rule")
    a = ap.parse_args()
    data = pd.read_parquet(a.data)
    nogap = pd.read_parquet(a.nogap_data) if a.nogap_data else None
    allc = feature_columns(data)
    temporal = feature_columns(data, ("temporal",))
    C = {  # name -> settings
        "copy_last_week": {"baseline": 7},
        "copy_last_4_weeks": {"baseline": 28},
        # feature groups (Q2 / Q3)
        "lgbm: window counts only": {"cols": [c for c in allc if c.startswith("cnt")]},
        "lgbm: + decay, recency, pattern": {"cols": temporal},
        "lgbm: + relational": {"cols": feature_columns(data, ("temporal", "relational"))},
        "lgbm: + global volume (all)": {"cols": allc},
        "lgbm: all - relational": {"cols": feature_columns(data, ("temporal", "global"))},
        # how much history to train on (Q1 / Q3)
        "lgbm: train from Oct 1": {"cols": allc, "train_from": "2026-10-01"},
        "lgbm: train from Oct 15": {"cols": allc, "train_from": "2026-10-15"},
        # decision layer
        "lgbm: joint thresholds (Score)": {"cols": allc, "joint": True},
        # feature gap (Q3): model trained without the gap, applied to rows with the gap
        "lgbm: trained without gap": {"cols": allc, "nogap": True},
        # regularization (model selection on the threshold week only)
        "lgbm: all, early stopping": {"cols": allc, "es": True},
        "lgbm: all, regularized + early stopping": {"cols": allc, "es": True, "reg": True},
        "lgbm: all - relational, regularized + es": {"cols": feature_columns(data, ("temporal", "global")),
                                                      "es": True, "reg": True},
    }
    if a.configs:
        C = {k: v for k, v in C.items() if k in a.configs}
    rows = []
    for tv in map(pd.Timestamp, a.val):
        va = data[data.t == tv]
        for name, cfg in C.items():
            if "baseline" in cfg:
                preds = {"-": copy_last(va, cfg["baseline"])}
            else:
                if cfg.get("nogap") and nogap is None:
                    continue
                pred, prob, (prob_th, th_rows) = run_lgbm(
                    data, tv, cfg["cols"], cfg.get("train_from"), cfg.get("joint", False),
                    nogap if cfg.get("nogap") else None, cfg.get("es", False),
                    LGB_PARAMS_REG if cfg.get("reg") else None)
                preds = {"threshold": pred}
                if a.rules == "both":
                    preds["count"] = decide(prob_th, th_rows, prob, va, "count")
            for rule, pred in preds.items():
                s = score(pred, va, va.github_user_id)
                rows.append({"val_t": str(tv), "model": name, "rule": rule, "score": s["score"],
                             "macro_f1": s["macro_f1"], "micro_f1": s["micro_f1"],
                             **{f"f1_{k}": v for k, v in s["per_event"].items()}})
                print(f"{tv:%m-%d} {name:34s} {rule:9s} score {s['score']:.4f}  macro {s['macro_f1']:.4f}  "
                      f"micro {s['micro_f1']:.4f}  " + " ".join(f"{k[:8]} {v:.3f}" for k, v in s["per_event"].items()),
                      flush=True)
    res = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    g = res.groupby(["model", "rule"], sort=False)
    summary = g[["score", "macro_f1", "micro_f1"]].mean().join(g.score.std().rename("score_std"))
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
