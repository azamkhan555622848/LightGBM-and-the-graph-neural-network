"""Score every saved GNN run (results/gnn_ablation/<name>_<mmdd>[_th].parquet) under both decision
rules (probability threshold / rank-based count), without retraining.

    python src/gnn_rescore.py --data data/snapshots/train.parquet --dir results/gnn_ablation
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from decision_eval import decide
from features import ACTS, DAY
from models import score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dir", default="results/gnn_ablation")
    ap.add_argument("--avg", nargs="*", default=[], help="NAME=run1,run2: also score the mean of several runs")
    ap.add_argument("--out", default="results/gnn_rules.csv")
    a = ap.parse_args()
    data = pd.read_parquet(a.data)
    runs = {}
    for f in sorted(Path(a.dir).glob("*_[0-9][0-9][0-9][0-9].parquet")):
        m = re.match(r"(.+)_(\d{4})$", f.stem)
        runs.setdefault(m.group(1), []).append(m.group(2))
    specs = {n: [n] for n in runs} | {s.split("=")[0]: s.split("=")[1].split(",") for s in a.avg}
    rows = []
    for name, members in specs.items():
        for mmdd in sorted(set.intersection(*[set(runs.get(m, [])) for m in members])):
            tv = pd.Timestamp(f"2026-{mmdd[:2]}-{mmdd[2:]} 16:00")
            va, th_rows = data[data.t == tv], data[data.t == tv - 7 * DAY]
            p = sum(pd.read_parquet(Path(a.dir) / f"{m}_{mmdd}.parquet")[ACTS].set_axis(va.index)
                    for m in members) / len(members)
            pth = sum(pd.read_parquet(Path(a.dir) / f"{m}_{mmdd}_th.parquet")[ACTS].set_axis(th_rows.index)
                      for m in members) / len(members)
            for rule in ("threshold", "count"):
                s = score(decide(pth, th_rows, p, va, rule), va, va.github_user_id)
                rows.append({"val_t": str(tv), "model": name, "rule": rule, "score": s["score"],
                             "macro_f1": s["macro_f1"], "micro_f1": s["micro_f1"],
                             **{f"f1_{k}": v for k, v in s["per_event"].items()}})
    r = pd.DataFrame(rows)
    r.to_csv(a.out, index=False)
    t = r.pivot_table(index=["model", "rule"], columns="val_t", values="score")
    t["mean"], t["weeks"] = t.mean(1), t.notna().sum(1) - 1
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
