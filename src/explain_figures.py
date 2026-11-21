"""Explanation figures for the slides: relational paths (Q2), temporal sampling (Q3), and a
slide-shaped framework figure (Q4).

    python src/explain_figures.py --out figures
"""
import argparse
from pathlib import Path

import graphviz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

NAVY = "#1f2f8a"


def relational_paths(out):
    """How a user's prediction reaches other tables: 1-hop own rows, 2-hop rows of others."""
    g = graphviz.Digraph("paths", graph_attr={"rankdir": "LR", "fontname": "Helvetica", "nodesep": "0.25",
                                              "ranksep": "0.7", "bgcolor": "white"},
                         node_attr={"fontname": "Helvetica", "fontsize": "12", "style": "rounded,filled",
                                    "shape": "box", "color": "#33475b"},
                         edge_attr={"fontname": "Helvetica", "fontsize": "10", "color": "#33475b"})
    g.node("u", "User u\n(target)", fillcolor="#f2c14e", fontsize="14")
    for n, lab in [("push", "Pushes"), ("pr", "PullRequests\n(authored)"), ("iss", "Issues\n(authored / assigned)"),
                   ("rev", "PRReviews /\nReviewComments"), ("ic", "IssueComments")]:
        g.node(n, lab, fillcolor="#e3f4e1")
    g.node("repo", "Repositories\nu works in", fillcolor="#dbe8f6")
    g.node("others", "Other users'\nrows in u's repos\n(repo heat, co-workers)", fillcolor="#fdebd3")
    g.node("onmine", "Other users' reviews /\ncomments / merges\non u's PRs & issues", fillcolor="#fdebd3")
    g.node("collab", "Collaborators'\nlast-week activity", fillcolor="#fdebd3")
    g.edge("push", "u", label="user_id"); g.edge("pr", "u", label="author_id")
    g.edge("iss", "u", label="author_id\nassignee_id"); g.edge("rev", "u", label="reviewer_id\nauthor_id")
    g.edge("ic", "u", label="author_id")
    g.edge("u", "repo", label="shortcut edge\n(derived)", style="dashed", color="#c0392b", fontcolor="#c0392b")
    g.edge("others", "repo", label="repo_id"); g.edge("onmine", "pr", label="pr_id"); g.edge("onmine", "iss", label="issue_id")
    g.edge("collab", "onmine", label="same users", style="dotted")
    with g.subgraph() as s:
        s.attr(rank="same"); s.node("others"); s.node("onmine"); s.node("collab")
    g.render(str(Path(out) / "relational_paths"), format="png", cleanup=True)


def temporal_sampling(out):
    """One training example: history before t - gap -> features; [t, t+7d) -> labels."""
    fig, ax = plt.subplots(figsize=(10, 2.9))
    ax.set_xlim(-60, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    # history with decaying weight
    for d in range(-56, 0):
        ax.add_patch(Rectangle((d, 1.6), 1, 0.6, color=NAVY, alpha=0.08 + 0.6 * 0.5 ** (-d / 7), lw=0))
    ax.add_patch(Rectangle((-0.46, 1.6), 0.46, 0.6, color="#bbbbbb", lw=0))
    ax.add_patch(Rectangle((0, 1.6), 7, 0.6, color="#e0b33c", lw=0))
    ax.text(-28, 2.4, "history used for features: rows with row time < t - gap\n"
            "(56-day windows, decayed counts, recency; GNN: graph of these rows)", ha="center", fontsize=9)
    ax.text(3.5, 2.4, "label week\n[t, t + 7 d)", ha="center", fontsize=9)
    ax.annotate("t - gap\n(feature cut-off)", xy=(-0.46, 1.6), xytext=(-9, 0.55), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.annotate("t = 16:00 UTC\n(00:00 Taiwan)", xy=(0, 1.6), xytext=(1.5, 0.55), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#555"))
    for x, lab in [(-56, "-56 d"), (-28, "-28 d"), (-14, "-14 d"), (-7, "-7 d")]:
        ax.plot([x, x], [1.5, 1.6], color="#555", lw=1); ax.text(x, 1.3, lab, ha="center", fontsize=8)
    ax.text(-60, 0.15, "Known-only-later columns (merged_at, merged_by_id, closed_at, state_reason, dismissed_at) are used only "
            "when their value is before the cut-off.", fontsize=8.5, color="#333")
    fig.tight_layout()
    fig.savefig(Path(out) / "temporal_sampling.png", dpi=170)


def framework_slide(out):
    """The Q4 framework figure, laid out to fill a 16:9 slide."""
    g = graphviz.Digraph("fw", graph_attr={"rankdir": "TB", "fontname": "Helvetica", "nodesep": "0.3",
                                           "ranksep": "0.35", "newrank": "true", "size": "13,7.2"},
                         node_attr={"fontname": "Helvetica", "fontsize": "12", "shape": "box",
                                    "style": "rounded,filled", "color": "#33475b", "margin": "0.18,0.08"},
                         edge_attr={"color": "#33475b", "arrowsize": "0.8"})
    c_data, c_feat, c_model, c_out = "#dbe8f6", "#fdebd3", "#e3f4e1", "#f6e1ef"
    g.node("gha", "① GH Archive hour files\n2026-09-15 00:00 → 2026-11-21 04:00 UTC (1,613 files)", fillcolor=c_data)
    g.node("pre", "② scripts/preprocess.py (course script, unchanged)\n9,000 cohort users · anonymized", fillcolor=c_data)
    g.node("db", "8 relational tables\nUsers · Repositories · PullRequests · Issues\nIssueComments · PRReviews · PRReviewComments · Pushes", fillcolor=c_data)
    g.node("snap", "③ Time-aware sampling: a prediction time t every day (00:00 Taiwan)\n"
           "inputs = rows with row time < t − 11 h · labels = 6 activities in [t, t + 7 d)", fillcolor=c_feat)
    g.node("fe", "④a Relational-temporal features (112)\ntemporal 70: windows 1–56 d, decay, recency, rhythm\n"
           "relational 30: repos, own PRs/issues, others on them, collaborators\nglobal 12: data volume", fillcolor=c_feat)
    g.node("graph", "④b Temporal heterogeneous graph at t − 11 h\n8 node types (one per table) · 17 FK edge types\n"
           "(+ reverse) · event rows of the last 56 days", fillcolor=c_feat)
    g.node("lgb", "⑤a LightGBM\n6 binary models · 400 trees", fillcolor=c_model)
    g.node("gnn", "⑤b Hetero GraphSAGE, 2 layers\nmean + sum aggregation → 6 logits", fillcolor=c_model)
    g.node("ens", "⑥ Average of the probabilities (3 seeds each)", fillcolor=c_out)
    g.node("th", "⑦ Rank-based decision per activity: on the last labelled week the F1-best threshold\n"
           "selects n_k users → in the test week the n_k most likely users get 1", fillcolor=c_out)
    g.node("csv", "⑧ output_413540002.csv → scripts/evaluate.py format check", fillcolor=c_out)
    for a, b in [("gha", "pre"), ("pre", "db"), ("db", "snap"), ("snap", "fe"), ("snap", "graph"), ("fe", "lgb"),
                 ("graph", "gnn"), ("lgb", "ens"), ("gnn", "ens"), ("ens", "th"), ("th", "csv")]:
        g.edge(a, b)
    with g.subgraph() as s:
        s.attr(rank="same"); s.node("fe"); s.node("graph")
    with g.subgraph() as s:
        s.attr(rank="same"); s.node("lgb"); s.node("gnn")
    g.render(str(Path(out) / "framework_slide"), format="png", cleanup=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    relational_paths(a.out)
    temporal_sampling(a.out)
    framework_slide(a.out)


if __name__ == "__main__":
    main()
