"""Relational deep learning: a temporal heterogeneous GNN on the 8-table graph.

Every table is a node type and every foreign key an edge type (both directions). For a
prediction time t the graph holds only rows whose row time is before t (event rows only for
the last LOOKBACK days), the "known only later" values only if they are before t, and every
node gets a time encoding of how long before t its row appeared. A 2-layer heterogeneous
GraphSAGE (mean + sum aggregation, so it can count events) reads the graph and a head
predicts the 6 activities of each user.

Options (for the ablations):
  time_enc   relative-time encoding of every node: "none", "sincos" (log age + sin/cos at
             periods 1, 7, 28 days) or "bucket" (one-hot age bucket 0-1, 1-3, 3-7, 7-14,
             14-28, 28-56, >56 days, + log age); with sum aggregation, buckets let the GNN
             learn windowed activity counts
  shortcuts  add derived User-Repository edges ("worked in", weighted by recency)
  user_feats feed the engineered temporal/relational features of features.py to user nodes
"""
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from features import ACTS, DAY

LOOKBACK = 56  # days of event rows (comments, reviews, pushes) in each snapshot graph

EVENT_TIME = {"IssueComments": "created_at", "PRReviews": "first_seen_at",
              "PRReviewComments": "created_at", "Pushes": "created_at"}
ENTITY_TIME = {"Users": "first_seen_at", "Repositories": "first_seen_at",
               "PullRequests": "first_seen_at", "Issues": "first_seen_at"}
# (source table, fk column, target table, relation name)
FKS = [
    ("PullRequests", "repo_id", "Repositories", "pr_in_repo"),
    ("PullRequests", "author_id", "Users", "pr_by"),
    ("Issues", "repo_id", "Repositories", "issue_in_repo"),
    ("Issues", "author_id", "Users", "issue_by"),
    ("Issues", "pull_request_id", "PullRequests", "issue_of_pr"),
    ("Issues", "assignee_id", "Users", "issue_assigned"),
    ("IssueComments", "issue_id", "Issues", "comment_on"),
    ("IssueComments", "author_id", "Users", "comment_by"),
    ("IssueComments", "repo_id", "Repositories", "comment_in_repo"),
    ("PRReviews", "pr_id", "PullRequests", "review_of"),
    ("PRReviews", "reviewer_id", "Users", "review_by"),
    ("PRReviewComments", "pr_id", "PullRequests", "rcomment_on"),
    ("PRReviewComments", "author_id", "Users", "rcomment_by"),
    ("PRReviewComments", "pull_request_review_id", "PRReviews", "rcomment_in_review"),
    ("Pushes", "user_id", "Users", "push_by"),
    ("Pushes", "repo_id", "Repositories", "push_to"),
]
FREQS = [1.0, 7.0, 28.0]  # periods (days) of the sinusoidal time encoding
BUCKETS = [1, 3, 7, 14, 28, 56]  # age bucket edges (days) of the bucket time encoding


def _len(s):
    return np.log1p(s.fillna("").str.len().to_numpy(dtype=float))


def time_encoding(age_days, kind="sincos"):
    """sincos: [log(1 + age), sin/cos of age at periods 1, 7, 28 days]
    bucket: [log(1 + age), one-hot age bucket]"""
    a = np.asarray(age_days, dtype=float)
    cols = [np.log1p(np.clip(a, 0, None)) / 5.0]
    if kind == "bucket":
        b = np.searchsorted(BUCKETS, a, side="right")
        cols += [(b == i).astype(float) for i in range(len(BUCKETS) + 1)]
    else:
        for p in FREQS:
            cols += [np.sin(2 * math.pi * a / p), np.cos(2 * math.pi * a / p)]
    return np.stack(cols, 1)


class SnapshotGraphs:
    """Builds the graph of the database as seen at time t."""

    def __init__(self, T, time_enc="sincos", shortcuts=False, lookback=LOOKBACK):
        self.T, self.time_enc, self.shortcuts, self.lookback = T, time_enc, shortcuts, lookback
        self.rt = {name: T[name][col].to_numpy() for name, col in {**ENTITY_TIME, **EVENT_TIME}.items()}
        U = T["Users"]
        self.cohort = U.is_cohort.astype(bool).to_numpy()
        # static per-row features (do not depend on t)
        P, I = T["PullRequests"], T["Issues"]
        C, RV, RC, PU, R = T["IssueComments"], T["PRReviews"], T["PRReviewComments"], T["Pushes"], T["Repositories"]
        st = RV.state.fillna("")
        self.static = {
            "Users": np.stack([U.is_bot.astype(float), self.cohort.astype(float)], 1),
            "Repositories": np.stack([R.org_id.notna().astype(float)], 1),
            "PullRequests": np.stack([P.is_cross_repo.astype(float).fillna(0), P.author_id.notna().astype(float),
                                      P.created_at.notna().astype(float)], 1),
            "Issues": np.stack([I.is_pull_request.astype(float), np.log1p(I.num_labels.fillna(0).astype(float)),
                                I.assignee_id.notna().astype(float), _len(I.body), _len(I.title) / 3,
                                I.created_via_app.notna().astype(float)], 1),
            "IssueComments": np.stack([C.is_pull_request.astype(float), _len(C.body), C.via_app.notna().astype(float)], 1),
            "PRReviews": np.stack([(st == "approved").astype(float), (st == "commented").astype(float),
                                   (st == "changes_requested").astype(float), _len(RV.body)], 1),
            "PRReviewComments": np.stack([(RC.subject_type == "file").astype(float), _len(RC.body),
                                          RC.in_reply_to_id.notna().astype(float), _len(RC.diff_hunk) / 3], 1),
            "Pushes": np.stack([(PU.ref_type == "tag").astype(float), PU.is_new_ref.astype(float)], 1),
        }

    def node_range(self, name, t):
        rt = self.rt[name]
        hi = int(np.searchsorted(rt, np.datetime64(t), side="left"))      # rows are in row-time order
        lo = 0
        if name in EVENT_TIME:
            lo = int(np.searchsorted(rt, np.datetime64(t - self.lookback * DAY), side="left"))
        return lo, hi

    def build(self, t, user_feats=None):
        T = self.T
        data = HeteroData()
        ranges = {name: self.node_range(name, t) for name in self.rt}
        tnp = np.datetime64(t)
        for name, (lo, hi) in ranges.items():
            x = [self.static[name][lo:hi]]
            if name == "PullRequests":
                P = T[name].iloc[lo:hi]
                x.append(np.stack([(P.merged_at.notna() & (P.merged_at < t)).astype(float),
                                   (P.closed_at.notna() & (P.closed_at < t)).astype(float)], 1))
            if name == "Issues":
                I = T[name].iloc[lo:hi]
                x.append(np.stack([(I.closed_at.notna() & (I.closed_at < t)).astype(float)], 1))
            if self.time_enc and self.time_enc != "none":
                age = (tnp - self.rt[name][lo:hi]) / np.timedelta64(1, "D")
                x.append(time_encoding(age, self.time_enc))
            data[name].x = torch.tensor(np.concatenate(x, 1), dtype=torch.float32)
        # one extra "ghost" user node (index hi) for test users with no row before t
        lo_u, hi_u = ranges["Users"]
        data["Users"].x = torch.cat([data["Users"].x, torch.zeros(1, data["Users"].x.shape[1])])
        if user_feats is not None:
            data["Users"].x = torch.cat([data["Users"].x, user_feats], 1)
        for src, col, dst, rel in FKS:
            slo, shi = ranges[src]
            dlo, dhi = ranges[dst]
            fk = T[src][col].iloc[slo:shi].to_numpy(dtype="float64", na_value=np.nan)
            ok = ~np.isnan(fk) & (fk >= dlo) & (fk < dhi)
            s = np.nonzero(ok)[0]
            d = fk[ok].astype(np.int64) - dlo
            ei = torch.tensor(np.stack([s, d]), dtype=torch.long)
            data[src, rel, dst].edge_index = ei
            data[dst, f"rev_{rel}", src].edge_index = ei.flip(0)
        # merged_by is known only after the merge
        P = T["PullRequests"].iloc[ranges["PullRequests"][0]:ranges["PullRequests"][1]]
        mb = P.merged_by_id.to_numpy(dtype="float64", na_value=np.nan)
        ok = ~np.isnan(mb) & (P.merged_at < t).to_numpy() & (mb < hi_u)
        ei = torch.tensor(np.stack([np.nonzero(ok)[0], mb[ok].astype(np.int64)]), dtype=torch.long)
        data["PullRequests", "merged_by", "Users"].edge_index = ei
        data["Users", "rev_merged_by", "PullRequests"].edge_index = ei.flip(0)
        if self.shortcuts:
            self._add_shortcuts(data, ranges, t)
        return data, ranges

    def _add_shortcuts(self, data, ranges, t):
        """Derived User -> Repository edges: every (user, repo) pair with activity in the
        lookback (pushes, comments, reviews, review comments), so a user reaches the other
        users and items of their repositories in 2 hops instead of 3-4."""
        T = self.T
        pairs = []
        for name, ucol in (("Pushes", "user_id"), ("IssueComments", "author_id"),
                           ("PRReviews", "reviewer_id"), ("PRReviewComments", "author_id")):
            lo, hi = ranges[name]
            df = T[name].iloc[lo:hi][[ucol, "repo_id"]].dropna()
            pairs.append(df.rename(columns={ucol: "u", "repo_id": "r"}))
        pairs = pd.concat(pairs).astype("int64").drop_duplicates()
        pairs = pairs[(pairs.u < ranges["Users"][1]) & (pairs.r < ranges["Repositories"][1])]
        ei = torch.tensor(pairs[["u", "r"]].to_numpy().T, dtype=torch.long)
        data["Users", "works_in", "Repositories"].edge_index = ei
        data["Repositories", "rev_works_in", "Users"].edge_index = ei.flip(0)


class RelGNN(nn.Module):
    def __init__(self, metadata, in_dims, hidden=64, layers=2, dropout=0.1, aggr="meansum", n_ctx=0):
        """aggr: "mean" (cannot count neighbours) or "meansum" (mean and sum, concatenated).
        n_ctx: size of a per-snapshot context vector (global data volume) given to the head."""
        super().__init__()
        agg = ["mean", "sum"] if aggr == "meansum" else "mean"
        self.enc = nn.ModuleDict({n: nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                                  for n, d in in_dims.items()})
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((hidden, hidden), hidden, aggr=agg) for et in metadata[1]}, aggr="sum")
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({n: nn.LayerNorm(hidden) for n in in_dims}) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden + n_ctx, hidden), nn.ReLU(), nn.Linear(hidden, len(ACTS)))

    def forward(self, data, target, ctx=None):
        h = {n: self.enc[n](data[n].x) for n in self.enc}
        h0 = h["Users"]
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, data.edge_index_dict)
            h = {n: self.drop(torch.relu(norm[n](out[n] + h[n]))) if n in out else h[n] for n in h}
        z = torch.cat([h["Users"][target], h0[target]], 1)
        if ctx is not None:
            z = torch.cat([z, ctx.expand(len(target), -1)], 1)
        return self.head(z)
