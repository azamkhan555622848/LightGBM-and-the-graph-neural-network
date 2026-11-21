"""Train the temporal heterogeneous GNN (gnn.py) on snapshots and score it like experiments.py.

    python src/train_gnn.py --db DB --data train.parquet --val 2026-11-12T16:00 --name gnn_time
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import GHOST
from features import ACTS, DAY, load_db
from gnn import RelGNN, SnapshotGraphs
from models import best_thresholds, binarize, feature_columns, score

import os
torch.set_num_threads(int(os.environ.get("GNN_THREADS", "16")))


def user_feature_matrix(df, cols, stats):
    x = df[cols].astype("float64").to_numpy()
    x = np.sign(x) * np.log1p(np.abs(x))
    return (x - stats[0]) / stats[1]


class Trainer:
    def __init__(self, T, data, time_enc="sincos", shortcuts=False, user_feats=False, hidden=64, seed=0,
                 layers=2, aggr="meansum", lookback=56, global_ctx=False):
        torch.manual_seed(seed); np.random.seed(seed)
        self.T, self.data = T, data
        self.layers, self.aggr = layers, aggr
        self.graphs = SnapshotGraphs(T, time_enc=time_enc, shortcuts=shortcuts, lookback=lookback)
        self.ctx_cols = [c for c in data.columns if c.startswith("global")] if global_ctx else []
        if self.ctx_cols:
            g = np.log1p(data.groupby("t")[self.ctx_cols].first().to_numpy(dtype="float64"))
            self.ctx_stats = (g.mean(0), g.std(0) + 1e-6)
        self.user_feats = user_feats
        self.cols = feature_columns(data) if user_feats else None
        if user_feats:
            x = data[self.cols].astype("float64").to_numpy()
            x = np.sign(x) * np.log1p(np.abs(x))
            self.stats = (x.mean(0), x.std(0) + 1e-6)
        self.cache = {}
        self.hidden = hidden
        self.model = None

    def snapshot(self, t):
        if t in self.cache:
            return self.cache[t]
        rows = self.data[self.data.t == t]
        tf = rows.t_feat.iloc[0] if "t_feat" in rows else t       # feature cut-off (t - gap)
        lo_u, hi_u = self.graphs.node_range("Users", tf)
        u = rows.user.to_numpy()
        target = np.where((u != GHOST) & (u < hi_u), u, hi_u)        # absent so far -> ghost node
        uf = None
        if self.user_feats:
            uf = torch.zeros(hi_u + 1, len(self.cols))
            uf[torch.tensor(target)] = torch.tensor(user_feature_matrix(rows, self.cols, self.stats), dtype=torch.float32)
        g, _ = self.graphs.build(tf, uf)
        y = torch.tensor(rows[[f"y_{k}" for k in ACTS]].to_numpy(dtype="float32")) if "y_open_pr" in rows else None
        ctx = None
        if self.ctx_cols:
            c = np.log1p(rows[self.ctx_cols].iloc[0].to_numpy(dtype="float64"))
            ctx = torch.tensor((c - self.ctx_stats[0]) / self.ctx_stats[1], dtype=torch.float32)[None]
        item = (g, torch.tensor(target), y, rows.index, ctx)
        self.cache[t] = item
        return item

    def fit(self, times, epochs=12, lr=3e-3):
        g0 = self.snapshot(times[-1])[0]
        in_dims = {n: g0[n].x.shape[1] for n in g0.node_types}
        self.model = RelGNN(g0.metadata(), in_dims, hidden=self.hidden, layers=self.layers, aggr=self.aggr,
                            n_ctx=len(self.ctx_cols))
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        ys = torch.cat([self.snapshot(t)[2] for t in times])
        pos = ys.mean(0).clamp(1e-3, 0.5)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).sqrt())
        for ep in range(epochs):
            self.model.train(); tot = 0.0
            for i in np.random.permutation(len(times)):
                g, tgt, y, _, ctx = self.snapshot(times[i])
                opt.zero_grad()
                loss = lossf(self.model(g, tgt, ctx), y)
                loss.backward(); opt.step(); tot += loss.item()
            print(f"    epoch {ep + 1}/{epochs} loss {tot / len(times):.4f}", flush=True)
        return self

    @torch.no_grad()
    def predict(self, t):
        self.model.eval()
        g, tgt, _, idx, ctx = self.snapshot(t)
        p = torch.sigmoid(self.model(g, tgt, ctx)).numpy()
        return pd.DataFrame(p, columns=ACTS, index=idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", nargs="+", default=["2026-10-29 16:00", "2026-11-05 16:00", "2026-11-12 16:00"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-time", action="store_true", help="same as --time-enc none")
    ap.add_argument("--time-enc", default="sincos", choices=["none", "sincos", "bucket"])
    ap.add_argument("--shortcuts", action="store_true")
    ap.add_argument("--user-feats", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--aggr", default="meansum", choices=["mean", "meansum"])
    ap.add_argument("--lookback", type=int, default=56, help="days of event rows in each graph")
    ap.add_argument("--global-ctx", action="store_true", help="give the head the global data-volume features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-from", default=None)
    ap.add_argument("--every", type=int, default=1, help="use every n-th daily snapshot for training")
    ap.add_argument("--out", default="work/exp_gnn")
    a = ap.parse_args()
    T = load_db(a.db)
    data = pd.read_parquet(a.data)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tv in map(pd.Timestamp, a.val):
        t0 = time.time()
        tr = Trainer(T, data, time_enc="none" if a.no_time else a.time_enc, shortcuts=a.shortcuts,
                     user_feats=a.user_feats, seed=a.seed, layers=a.layers, aggr=a.aggr, lookback=a.lookback,
                     global_ctx=a.global_ctx)
        times = sorted(t for t in data.t.unique() if t <= tv - 14 * DAY
                       and (a.train_from is None or t >= pd.Timestamp(a.train_from)))
        times = times[::-1][::a.every][::-1]
        tr.fit(times, epochs=a.epochs)
        th_t = tv - 7 * DAY
        th_rows, va = data[data.t == th_t], data[data.t == tv]
        prob_th = tr.predict(th_t)
        th = best_thresholds(prob_th, th_rows)
        prob = tr.predict(tv)
        prob.assign(t=tv, github_user_id=va.github_user_id.values).to_parquet(out / f"{a.name}_{tv:%m%d}.parquet")
        prob_th.assign(t=th_t, github_user_id=th_rows.github_user_id.values).to_parquet(
            out / f"{a.name}_{tv:%m%d}_th.parquet")
        s = score(binarize(prob, th), va, va.github_user_id)
        rows.append({"val_t": str(tv), "model": a.name, "score": s["score"], "macro_f1": s["macro_f1"],
                     "micro_f1": s["micro_f1"], **{f"f1_{k}": v for k, v in s["per_event"].items()}})
        print(f"{tv:%m-%d} {a.name:28s} score {s['score']:.4f}  macro {s['macro_f1']:.4f}  micro {s['micro_f1']:.4f}  "
              + " ".join(f"{k[:8]} {v:.3f}" for k, v in s["per_event"].items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(out / f"{a.name}.csv", index=False)


if __name__ == "__main__":
    main()
