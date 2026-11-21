"""Question 1: draw the schema graph of the 8 tables with Graphviz.

Each table is a node listing its primary key (PK), foreign keys (FK) and row-time column;
each foreign key is an edge FK -> referenced PK.

    python src/schema_graph.py --out figures/schema_graph
"""
import argparse

import graphviz

TABLES = {
    "Users": ("user_id", [], "first_seen_at", ["github_user_id", "login", "is_bot", "is_cohort"]),
    "Repositories": ("repo_id", [], "first_seen_at", ["github_repo_id", "name", "owner_login", "org_id", "org_login"]),
    "PullRequests": ("pr_id", [("repo_id", "Repositories"), ("author_id", "Users"), ("merged_by_id", "Users")],
                     "first_seen_at", ["github_pr_id", "number", "title", "draft", "head/base_ref", "head/base_sha",
                                       "head_repo_id", "is_cross_repo", "created_at", "merged_at*", "closed_at*"]),
    "Issues": ("issue_id", [("repo_id", "Repositories"), ("author_id", "Users"),
                            ("pull_request_id", "PullRequests"), ("assignee_id", "Users")],
               "first_seen_at", ["github_issue_id", "number", "is_pull_request", "title", "body", "labels",
                                 "num_labels", "milestone_title", "issue_type", "locked", "created_via_app",
                                 "created_at", "closed_at*", "state_reason*"]),
    "IssueComments": ("comment_id", [("issue_id", "Issues"), ("repo_id", "Repositories"), ("author_id", "Users")],
                      "created_at", ["github_comment_id", "is_pull_request", "body", "via_app", "event_id"]),
    "PRReviews": ("review_id", [("pr_id", "PullRequests"), ("repo_id", "Repositories"), ("reviewer_id", "Users")],
                  "first_seen_at", ["github_review_id", "state", "body", "commit_id", "submitted_at",
                                    "dismissed_at*", "event_id"]),
    "PRReviewComments": ("comment_id", [("pr_id", "PullRequests"), ("repo_id", "Repositories"),
                                        ("author_id", "Users"), ("pull_request_review_id", "PRReviews"),
                                        ("in_reply_to_id", "PRReviewComments")],
                         "created_at", ["github_comment_id", "path", "file_extension", "subject_type",
                                        "position", "original_position", "commit_id", "diff_hunk", "body", "event_id"]),
    "Pushes": ("push_id", [("user_id", "Users"), ("repo_id", "Repositories")],
               "created_at", ["github_push_id", "ref", "ref_type", "branch", "is_new_ref", "event_id"]),
}
COLOR = {"Users": "#d9e8fb", "Repositories": "#d9e8fb", "PullRequests": "#fdebd3", "Issues": "#fdebd3",
         "IssueComments": "#e3f4e1", "PRReviews": "#e3f4e1", "PRReviewComments": "#e3f4e1", "Pushes": "#e3f4e1"}


def label(name, pk, fks, rt, cols, full):
    rows = [f'<TR><TD BGCOLOR="#1f3b5c" PORT="title"><FONT COLOR="white"><B>{name}</B></FONT></TD></TR>',
            f'<TR><TD ALIGN="LEFT" PORT="{pk}"><B>PK</B>  {pk}</TD></TR>']
    rows += [f'<TR><TD ALIGN="LEFT" PORT="{c}"><B>FK</B>  {c} → {ref}</TD></TR>' for c, ref in fks]
    rows.append(f'<TR><TD ALIGN="LEFT"><I>row time: {rt}</I></TD></TR>')
    if full:
        rows.append('<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">' + "<BR ALIGN=\"LEFT\"/>".join(cols) + '<BR ALIGN="LEFT"/></FONT></TD></TR>')
    return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="{COLOR[name]}">{"".join(rows)}</TABLE>>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/schema_graph")
    ap.add_argument("--full", action="store_true", help="also list the other columns")
    a = ap.parse_args()
    g = graphviz.Digraph("schema", graph_attr={"rankdir": "LR", "splines": "spline", "nodesep": "0.4",
                                                "ranksep": "1.1", "fontname": "Helvetica"},
                         node_attr={"shape": "plaintext", "fontname": "Helvetica", "fontsize": "11"},
                         edge_attr={"color": "#555555", "arrowsize": "0.7", "fontname": "Helvetica", "fontsize": "9"})
    for name, (pk, fks, rt, cols) in TABLES.items():
        g.node(name, label(name, pk, fks, rt, cols, a.full))
    for name, (pk, fks, rt, cols) in TABLES.items():
        for col, ref in fks:
            g.edge(f"{name}:{col}:e", f"{ref}:{TABLES[ref][0]}:w")
    for fmt in ("png", "pdf", "svg"):
        g.render(a.out, format=fmt, cleanup=True)
    print("wrote", a.out + ".{png,pdf,svg}")


if __name__ == "__main__":
    main()
