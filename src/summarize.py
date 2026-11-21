"""Collect the experiment CSVs into the tables used in the slides (all on the same 3 validation
weeks; Score in %, mean ± std over the weeks).

    python src/summarize.py
"""
from pathlib import Path

import pandas as pd

ACTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "summary"


def stats(df):
    g = df.groupby("label", sort=False)
    t = pd.DataFrame({"Score": g.score.mean() * 100, "± std": g.score.std() * 100,
                      "macro-F1": g.macro_f1.mean() * 100, "micro-F1": g.micro_f1.mean() * 100,
                      **{k: g[f"f1_{k}"].mean() * 100 for k in ACTS}, "weeks": g.score.size()})
    return t.round(1)


def pick(df, spec):
    """spec: list of (label, model, rule)"""
    parts = []
    for label, model, rule in spec:
        sel = df[(df.model == model) & ((df.rule == rule) if rule else True)]
        parts.append(sel.assign(label=label))
    return pd.concat(parts, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lg = pd.read_csv(R / "lgbm_rules.csv")
    gn = pd.read_csv(R / "gnn_rules.csv")
    de = pd.read_csv(R / "decision_final.csv")
    base = lg[lg.model == "copy_last_week"]

    q2 = stats(pick(gn, [
        ("Hetero GraphSAGE, mean aggregation", "gnn_mean_agg", "count"),
        ("+ mean & sum aggregation (counts)", "gnn_count_agg", "count"),
        ("+ user→repo shortcut edges", "gnn_shortcuts", "count"),
        ("   same with 1 layer (1 hop)", "gnn_sc_1layer", "count"),
        ("+ shortcuts + global context", "gnn_shortcuts_ctx", "count"),
    ]))
    q2_lgbm = stats(pick(lg, [
        ("LightGBM temporal features", "lgbm: + decay, recency, pattern", "count"),
        ("+ relational features", "lgbm: + relational", "count"),
        ("all features", "lgbm: + global volume (all)", "count"),
        ("all − relational", "lgbm: all - relational", "count"),
    ]))
    q3 = stats(pd.concat([
        pick(base, [("Copy last week (no model)", "copy_last_week", None)]),
        pick(lg, [
            ("LightGBM: window counts only", "lgbm: window counts only", "count"),
            ("+ decay, recency, rhythm", "lgbm: + decay, recency, pattern", "count"),
            ("+ relational + global volume (all)", "lgbm: + global volume (all)", "count"),
            ("History: train from Oct 15 (4 wk)", "lgbm: train from Oct 15", "count"),
            ("History: train from Oct 1 (6 wk)", "lgbm: train from Oct 1", "count"),
            ("History: from Sep 22 (7 wk, ours)", "lgbm: + global volume (all)", "count"),
            ("Trained without the 11 h gap", "lgbm: trained without gap", "count"),
        ]),
        pick(gn, [
            ("GNN: no time encoding", "gnn_shortcuts", "count"),
            ("GNN: sin/cos time encoding", "gnn_sc_sincos", "count"),
            ("GNN: age-bucket time encoding", "gnn_sc_bucket", "count"),
            ("GNN: all history (no 56-day window)", "gnn_all_history", "count"),
        ]),
        pick(de, [
            ("Ensemble, probability threshold", "lgbm+gnn", "threshold"),
            ("Ensemble, rank-based count (ours)", "lgbm+gnn", "count"),
        ]),
    ], ignore_index=True))
    final = stats(pd.concat([
        pick(base, [("Copy last week", "copy_last_week", None)]),
        pick(de, [("LightGBM", "lgbm", "count"), ("GNN (3 seeds)", "gnn", "count"),
                  ("LightGBM + GNN, probability threshold", "lgbm+gnn", "threshold"),
                  ("LightGBM + GNN, rank-based (final)", "lgbm+gnn", "count")]),
    ], ignore_index=True))
    for name, t in (("q2_gnn", q2), ("q2_lgbm", q2_lgbm), ("q3", q3), ("final", final)):
        t.to_csv(OUT / f"{name}_table.csv")
        print(f"\n== {name}\n" + t.to_string())


if __name__ == "__main__":
    main()
