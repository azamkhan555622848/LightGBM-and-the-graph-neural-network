"""Question 4: the framework figure (full pipeline), drawn with Graphviz.

    python src/framework_figure.py --out figures/framework
"""
import argparse

import graphviz


def box(g, name, text, fill, shape="box"):
    g.node(name, text, shape=shape, style="rounded,filled", fillcolor=fill, color="#33475b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/framework")
    a = ap.parse_args()
    g = graphviz.Digraph("framework", graph_attr={"rankdir": "LR", "fontname": "Helvetica", "nodesep": "0.35",
                                                   "ranksep": "0.45", "compound": "true"},
                         node_attr={"fontname": "Helvetica", "fontsize": "11", "margin": "0.15,0.08"},
                         edge_attr={"color": "#33475b", "arrowsize": "0.7"})
    data, feat, model, out = "#dbe8f6", "#fdebd3", "#e3f4e1", "#f6e1ef"
    with g.subgraph(name="cluster_data") as c:
        c.attr(label="1  Data", style="dashed", color="#8899aa", fontname="Helvetica")
        box(c, "gha", "GH Archive hour files\n2026-09-15 → 2026-11-21 (UTC)", data)
        box(c, "pre", "scripts/preprocess.py\n(course script: cohort filter,\nanonymization)", data)
        box(c, "db", "8 relational tables\nUsers · Repositories · PullRequests\nIssues · IssueComments · PRReviews\nPRReviewComments · Pushes", data)
        c.edge("gha", "pre"); c.edge("pre", "db")
    with g.subgraph(name="cluster_snap") as c:
        c.attr(label="2  Time-aware sampling", style="dashed", color="#8899aa", fontname="Helvetica")
        box(c, "snap", "Daily prediction times t\n(16:00 UTC = 00:00 Taiwan)\nrows with row time < t only;\nlabels = 6 activities in [t, t+7d)", feat)
    with g.subgraph(name="cluster_models") as c:
        c.attr(label="3  Two relational models", style="dashed", color="#8899aa", fontname="Helvetica")
        box(c, "fe", "Relational-temporal features\n· temporal: windows 1–56d, decay,\n  recency, active days, trend\n· relational: repos, own PRs/issues,\n  others acting on them, collaborators\n· global: data-volume level", feat)
        box(c, "lgb", "LightGBM\n(one binary model\nper activity)", model)
        box(c, "graph", "Temporal heterogeneous graph\n8 node types, 17 FK edge types (+reverse)\n+ user→repo shortcut edges\n+ relative-time encoding per node", feat)
        box(c, "gnn", "Hetero GraphSAGE (2 layers)\n+ user features → 6 logits", model)
        c.edge("fe", "lgb"); c.edge("graph", "gnn")
    with g.subgraph(name="cluster_out") as c:
        c.attr(label="4  Decision", style="dashed", color="#8899aa", fontname="Helvetica")
        box(c, "ens", "Average of\nprobabilities", out)
        box(c, "th", "Per-activity threshold\nmaximizing F1 on the\nlast labelled week", out)
        box(c, "csv", "output_STUDENTID.csv\n(0/1 × 6 activities)\n→ evaluate.py check", out)
        c.edge("ens", "th"); c.edge("th", "csv")
    g.edge("db", "snap")
    g.edge("snap", "fe"); g.edge("snap", "graph")
    g.edge("lgb", "ens"); g.edge("gnn", "ens")
    for fmt in ("png", "pdf", "svg"):
        g.render(a.out, format=fmt, cleanup=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
