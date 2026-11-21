"""Build the report slides (course template style) -> slides/Asg2_413540002.pptx (+ PDF via LibreOffice).

All numbers come from results/summary/*.csv and data/db, written by the experiment scripts.

    python slides/build_slides.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pptx_helpers import BAR, GOOD, GREY, LIGHT, NAVY, WHITE, Deck  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results" / "summary"
CFG = json.loads((ROOT / "slides" / "slide_config.json").read_text())
STUDENT_ID, NAME, REPO = CFG["student_id"], CFG["name"], CFG["repo"]
ACTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
SHORT = {"open_pr": "open PR", "open_issue": "open issue", "issue_comment": "issue cmt", "pr_review": "PR review",
         "pr_review_comment": "review cmt", "push": "push"}


def tab(name):
    return pd.read_csv(RES / f"{name}_table.csv", index_col=0)


def fmt(v, d=1):
    return f"{v:.{d}f}"


def result_rows(t, names, label=None, cols=("Score", "± std", "macro-F1", "micro-F1")):
    rows = []
    for n in names:
        if n not in t.index:
            continue
        r = t.loc[n]
        rows.append([label.get(n, n) if label else n] + [fmt(r[c]) for c in cols])
    return rows


def build():
    d = Deck()
    L = Inches

    # 1 ------------------------------------------------------------ title
    s = d.slide(notes=CFG["notes"].get("1", ""))
    d.text(s, L(0.5), L(1.25), L(9), L(1.1), [("Assignment 2:", {"bold": True, "size": 32}),
                                              ("Predicting GitHub User Activity", {"size": 28})],
           align=PP_ALIGN.CENTER)
    line = s.shapes.add_connector(1, L(1.75), L(2.62), L(8.25), L(2.62))
    line.line.color.rgb = GREY; line.line.width = Pt(2.5)
    d.text(s, L(0.5), L(2.85), L(9), L(1.6), [
        ((f"{NAME}   ·   " if NAME and NAME != "NAME" else "") + f"Student ID {STUDENT_ID}", {"size": 18, "bold": True}),
        (f"Code: {REPO}" if "USER/REPO" not in REPO else "Code: https://github.com/<to be added>", {"size": 13}),
        ("Big Data Analytics Techniques and Applications · Fall 2026", {"size": 13}),
    ], align=PP_ALIGN.CENTER, spacing=1.15)

    # 2 ------------------------------------------------------------ task & data
    s = d.slide(notes=CFG["notes"].get("2", ""))
    d.header(s, "Task, data and what we found first", underline=False)
    st = CFG["stats"]
    d.text(s, L(0.4), L(0.8), L(4.35), L(4.4), [
        ("**Task**: for 4,465 test users, predict 6 binary labels — open_pr, open_issue, issue_comment, "
         "pr_review, pr_review_comment, push — for **Nov 22–29, 2026 (Taiwan)**.", {"size": 12, "space_after": 4}),
        ("**Score** = (macro-F1 + micro-F1) / 2 over the 6 activities.", {"size": 12, "space_after": 4}),
        ("**Input**: only the 8 tables built by the course's preprocess.py (9,000 cohort users).",
         {"size": 12, "space_after": 8}),
        ("**Finding 1 – the data is disappearing.** GH Archive has lost most events since mid-Oct: cohort pushes "
         f"fell from ~{st['push_per_day_oct']:,}/day (early Oct) to ~{st['push_per_day_nov']}/day (Nov). "
         f"Weekly positive rate of push among test users: {st['push_rate_sep']}% (late Sep) → {st['push_rate_nov']}% (Nov).",
         {"size": 12, "space_after": 4}),
        ("**Finding 2 – our labels match the course's.** Copy-last-week on the TAs' reference week "
         f"(Sep 27–Oct 4): **{st['copy_ref_ours']}%** vs TAs' **49.61%**.", {"size": 12, "space_after": 4}),
        ("**Finding 3 – positives are rare** in Nov: 2–9% of test users per activity and week.", {"size": 12}),
    ])
    d.image(s, FIG / "weekly_volume.png", L(4.85), L(0.9), L(5.0), L(3.9))

    # 3 ------------------------------------------------------------ Q1 schema
    s = d.slide(notes=CFG["notes"].get("3", ""))
    d.header(s, "(1%) Question 1", "Draw the Schema Graph and provide Dataset Time Period you use.")
    d.image(s, FIG / "schema_graph.png", L(0.3), L(1.05), L(7.2), L(4.25))
    d.text(s, L(7.55), L(1.15), L(2.3), L(4.1), [
        ("**8 tables = nodes**", {"size": 11, "space_after": 3}),
        ("**20 foreign keys = edges** (FK → referenced PK)", {"size": 11, "space_after": 3}),
        ("PK / FK marked in each table; every PK is a row number", {"size": 11, "space_after": 3}),
        ("**Row time** of each table: when the row may be used (first_seen_at / created_at)", {"size": 11, "space_after": 3}),
        ("Blue = entities, orange = PRs & issues, green = activity events", {"size": 11}),
    ], fill=LIGHT)

    # 4 ------------------------------------------------------------ Q1 period
    s = d.slide(notes=CFG["notes"].get("4", ""))
    d.header(s, "Question 1 (cont.) — dataset time period", underline=False)
    d.image(s, FIG / "timeline.png", L(0.25), L(0.75), L(9.5), L(3.05))
    q3 = tab("q3")
    rows_db = {t: len(pd.read_parquet(ROOT / "data" / "db" / f"{t}.parquet", columns=[c]))
               for t, c in (("Users", "user_id"), ("Repositories", "repo_id"), ("PullRequests", "pr_id"),
                            ("Issues", "issue_id"), ("IssueComments", "comment_id"), ("PRReviews", "review_id"),
                            ("PRReviewComments", "comment_id"), ("Pushes", "push_id"))}
    d.text(s, L(0.4), L(3.8), L(9.2), L(1.4), [
        ("**GH Archive hour files 2026-09-15 00:00 → 2026-11-21 04:00 UTC** (Taiwan 09-15 08:00 → 11-21 12:59): "
         "1,613 files, 67 days (the course allows data from 2025-12-01).", {"size": 12, "space_after": 3}),
        ("Why: 56-day feature windows + 52 daily labelled times; **6–7 weeks of training history beat 4 weeks** "
         f"despite the data shift (LightGBM Score, training from Sep 22 **{v(q3, 'History: from Sep 22 (7 wk, ours)'):.1f}**, "
         f"Oct 1 {v(q3, 'History: train from Oct 1 (6 wk)'):.1f}, Oct 15 {v(q3, 'History: train from Oct 15 (4 wk)'):.1f}); "
         "data until noon of the deadline day so the pipeline can run before 23:55.", {"size": 12, "space_after": 3}),
        ("Rows: " + " · ".join(f"{k} {n:,}" for k, n in rows_db.items()), {"size": 9.5, "color": GREY}),
    ])

    for builder in (q2_slides, q3_slides, q4_slides, conclusion_slide):
        builder(d)
    out = ROOT / "slides" / f"Asg2_{STUDENT_ID}.pptx"
    d.save(out)
    print("wrote", out, d.n, "slides")


# =================================================================== Q2
def v(t, label, col="Score"):
    return float(t.loc[label, col])


def q2_slides(d):
    L = Inches
    s = d.slide(notes=CFG["notes"].get("q2a", ""))
    d.header(s, "(3%) Question 2", "Explain how you enhanced the modeling of relationships across different tables.")
    d.image(s, FIG / "relational_paths.png", L(0.25), L(1.1), L(5.9), L(3.5))
    d.text(s, L(6.25), L(1.1), L(3.55), L(4.1), [
        ("Two complementary relational views", {"bold": True, "size": 13, "space_after": 4}),
        ("**A. Relational deep learning (GNN)**: every row is a node, every foreign key an edge "
         "(8 node types, 17 FK edge types + reverse). Message passing lets user u see rows 2 hops away.",
         {"size": 11, "space_after": 4}),
        ("**B. Relational features** for LightGBM: joins + aggregations along the same FK paths "
         "(30 features).", {"size": 11, "space_after": 4}),
        ("Paths used (left): u's own rows (1 hop) and **other users' rows on u's repositories, PRs and "
         "issues** (2 hops) — e.g. a review on u's PR often triggers a push or a reply by u.", {"size": 11}),
    ], fill=LIGHT)
    q2_gnn(d)
    q2_features(d)


def q2_gnn(d):
    L = Inches
    s = d.slide(notes=CFG["notes"].get("q2b", ""))
    d.header(s, "Question 2 (cont.) — enhancing the relational GNN", underline=False)
    t = tab("q2_gnn")
    rows = [["GNN variant (all: no time encoding)", "Score", "± std", "macro-F1", "micro-F1"]]
    rows += [[i] + [fmt(t.loc[i, c]) for c in ("Score", "± std", "macro-F1", "micro-F1")] for i in t.index]
    d.table(s, rows, L(0.35), L(0.8), L(5.6), L(0.34) * len(rows), col_widths=[5.2, 1, 1, 1.1, 1.1], size=10,
            highlight=[2])
    gain = v(t, "+ mean & sum aggregation (counts)") - v(t, "Hetero GraphSAGE, mean aggregation")
    hop = v(t, "+ user→repo shortcut edges") - v(t, "   same with 1 layer (1 hop)")
    weeks = pd.read_csv(ROOT / "results" / "gnn_rules.csv")
    w = weeks[weeks.rule == "count"].pivot(index="model", columns="val_t", values="score")
    better = int((w.loc["gnn_count_agg"] > w.loc["gnn_mean_agg"]).sum())
    d.text(s, L(6.1), L(0.8), L(3.7), L(4.35), [
        ("What changed and why", {"bold": True, "size": 12, "space_after": 4}),
        (f"**Count-aware aggregation (+{gain:.1f})**: mean aggregation only sees the average neighbour, so a "
         "user with 10 pushes looks like one with 1. Adding sum aggregation (mean ‖ sum) lets messages count "
         f"rows. Better in {better} of 3 weeks.", {"size": 11, "space_after": 4}),
        (f"**Two hops (+{hop:.1f} vs 1 hop)**: u → u's PRs / issues / repos → other users' rows on them. "
         "Relations of relations carry signal.", {"size": 11, "space_after": 4}),
        ("**Shortcut user→repo edges** and the global context: no gain beyond the week-to-week noise; "
         "the final GNN keeps the plain FK graph.", {"size": 11}),
    ], fill=LIGHT)
    d.text(s, L(0.35), L(0.85) + L(0.34) * len(rows), L(5.6), L(0.5), [
        ("Score (%) = (macro-F1 + micro-F1) / 2, mean ± std over the 3 validation weeks, rank-based "
         "decision, one seed per variant. Highlighted = the GNN used in the final model.", {"size": 9, "color": GREY})])
    d.image(s, FIG / "gnn_agg_by_week.png", L(0.35), L(3.35), L(5.6), L(1.95))


def q2_features(d):
    L = Inches
    s = d.slide(notes=CFG["notes"].get("q2c", ""))
    d.header(s, "Question 2 (cont.) — relational features vs. the graph", underline=False)
    lg, ap, fin = tab("q2_lgbm"), pd.read_csv(RES / "ap_table.csv", index_col=0), tab("final")
    rows = [["LightGBM features", "Score", "± std"]]
    rows += [[i, fmt(lg.loc[i, "Score"]), fmt(lg.loc[i, "± std"])] for i in lg.index]
    d.table(s, rows, L(0.35), L(0.8), L(4.4), L(0.3) * len(rows), col_widths=[3.0, 0.8, 0.8], size=10)
    top2 = L(0.95) + L(0.3) * len(rows)
    rows2 = [["Average precision", "all users", "warm", "cold"]]
    rows2 += [[i, fmt(ap.loc[i, "all"], 3), fmt(ap.loc[i, "warm"], 3), fmt(ap.loc[i, "cold"], 3)] for i in ap.index]
    d.table(s, rows2, L(0.35), top2, L(4.4), L(0.28) * len(rows2), col_widths=[2.4, 0.9, 0.8, 0.8], size=9.5)
    sizes = pd.read_csv(RES / "ap_group_sizes.csv")
    gs = pd.read_csv(RES / "gain_share.csv", index_col=0)
    rel_share = gs.loc["relational"].mean() * 100
    d.text(s, L(0.35), top2 + L(0.28) * len(rows2) + L(0.05), L(4.4), L(0.6), [
        ("AP = average precision (threshold-free), mean of 6 activities × 3 weeks. Cold = no own activity in the 28 "
         f"days before the cut-off ({int(sizes[sizes.users == 'cold'].n.min()):,}–{int(sizes[sizes.users == 'cold'].n.max()):,} users).",
         {"size": 9, "color": GREY})])
    ens = v(fin, "LightGBM + GNN, rank-based (final)")
    apr = pd.read_csv(ROOT / "results" / "analysis_ap.csv")
    cp = apr[(apr.config == apr.config.iloc[0]) & (apr.users == "cold")].pos
    cold_min, cold_max = int(cp.min()), int(cp.max())
    d.text(s, L(4.95), L(0.8), L(4.85), L(4.35), [
        ("What we found", {"bold": True, "size": 12, "space_after": 4}),
        ("**30 relational features** follow the FK paths: repositories a user works in and how busy they are, the "
         "user's open PRs / issues, other users' reviews, comments and merges on them, collaborators' activity.",
         {"size": 10.5, "space_after": 3}),
        (f"**The trees use them** ({rel_share:.0f}% of the split gain) **but they do not improve held-out results**: "
         f"Score {v(lg, 'all features'):.1f} with vs {v(lg, 'all − relational'):.1f} without; AP "
         f"{ap.loc['+ decay, recency, pattern', 'all']:.3f} → {ap.loc['+ relational', 'all']:.3f}. "
         "A negative result we report.", {"size": 10.5, "space_after": 3}),
        ("Likely reason: only 2–9% of test users do a given activity in a week, and the signal \"others act on my "
         "items\" mostly concerns users who are already active, so their own history carries it. Cold users have "
         f"only {cold_min}–{cold_max} positives per activity and week — too few to learn from.", {"size": 10.5, "space_after": 3}),
        (f"**What does help is learning on the graph and combining the two views**: GNN {v(fin, 'GNN (3 seeds)'):.1f}, "
         f"LightGBM {v(fin, 'LightGBM'):.1f}, average of both **{ens:.1f}** (better than LightGBM in all 3 weeks).",
         {"size": 10.5}),
    ], fill=LIGHT)


# =================================================================== Q3
def q3_slides(d):
    L = Inches
    s = d.slide(notes=CFG["notes"].get("q3a", ""))
    d.header(s, "(5%) Question 3", "Explain how you enhanced the modeling of temporal information in the data.")
    d.image(s, FIG / "temporal_sampling.png", L(0.3), L(1.05), L(9.4), L(2.0))
    d.text(s, L(0.4), L(3.05), L(9.2), L(2.2), CFG["q3_sampling_text"], size=11)
    q3_features(d)
    q3_shift(d)
    q3_results(d)


def q3_features(d):
    L = Inches
    s = d.slide(notes=CFG["notes"].get("q3b", ""))
    d.header(s, "Question 3 (cont.) — temporal features and time encodings", underline=False)
    d.text(s, L(0.4), L(0.75), L(4.6), L(2.9), CFG["q3_feat_left"], size=10.5, fill=LIGHT)
    d.text(s, L(5.15), L(0.75), L(4.5), L(2.9), CFG["q3_feat_right"], size=10.5, fill=LIGHT)
    d.image(s, FIG / "gain_share.png", L(0.4), L(3.55), L(6.4), L(1.8))
    gs = pd.read_csv(RES / "gain_share.csv", index_col=0)
    d.text(s, L(6.9), L(3.7), L(2.8), L(1.6), [
        ("What LightGBM uses", {"bold": True, "size": 10.5, "space_after": 2}),
        (f"Decayed counts: {gs.loc['temporal: decayed counts'].mean() * 100:.0f}% of the split gain on average; "
         f"for push the global volume features take {gs.loc['global (data volume)', 'push'] * 100:.0f}% and "
         f"rhythm {gs.loc['temporal: activity pattern', 'push'] * 100:.0f}%.", {"size": 9.5})])


def q3_shift(d):
    L = Inches
    s = d.slide(notes=CFG["notes"].get("q3c", ""))
    d.header(s, "Question 3 (cont.) — GH Archive coverage changes", underline=False)
    d.image(s, FIG / "weekly_volume.png", L(0.3), L(0.8), L(5.3), L(2.55))
    q3 = tab("q3")
    dg = json.loads((RES / "push_diag.json").read_text())
    d.text(s, L(5.75), L(0.8), L(4.05), L(4.35), [
        ("Problem", {"bold": True, "size": 12, "space_after": 3}),
        (f"Share of test users with a recorded push: {dg['push_rate_sep']}% (late Sep) → "
         f"{dg['push_rate_th']}% (week of Nov 5) → {dg['push_rate_va']}% (week of Nov 12).", {"size": 11, "space_after": 3}),
        (f"A probability threshold tuned on the Nov 5 week selects only **{dg['n_selected']} users** in the "
         f"Nov 12 week, where **{dg['n_pos']}** pushed → push F1 {dg['f1_threshold']:.2f}, although the ranking "
         f"could reach F1 {dg['f1_best']:.2f}.", {"size": 11, "space_after": 6}),
        ("Fix: rank-based decision", {"bold": True, "size": 12, "space_after": 3}),
        ("On the last labelled week the F1-best threshold selects n_k users; in the new week the n_k users "
         "with the highest probability get 1. Ranking adapts to the new week, n_k keeps the base rate.",
         {"size": 11, "space_after": 3}),
        (f"Ensemble Score {v(q3, 'Ensemble, probability threshold'):.1f} → "
         f"**{v(q3, 'Ensemble, rank-based count (ours)'):.1f}**; push F1 in the Nov 12 week "
         f"{dg['ens_push_threshold']:.2f} → {dg['ens_push_count']:.2f}.", {"size": 11}),
    ], fill=LIGHT)
    d.text(s, L(0.4), L(3.45), L(5.2), L(1.75), [
        ("Also against the drift:", {"bold": True, "size": 11, "space_after": 2}),
        (f"**Long, sliding training history**: 7 weeks of daily snapshots across the drop beat the recent "
         f"4 weeks only: {v(q3, 'History: from Sep 22 (7 wk, ours)'):.1f} vs "
         f"{v(q3, 'History: train from Oct 15 (4 wk)'):.1f}.", {"size": 10.5, "space_after": 2, "b": True}),
        ("**Global volume features** tell the model how complete recent data is "
         f"(LightGBM gain share {dg['global_gain_push']:.0f}% for push).", {"size": 10.5, "b": True}),
    ])


def q3_results(d):
    L = Inches
    s = d.slide(notes=CFG["notes"].get("q3d", ""))
    d.header(s, "Question 3 (cont.) — temporal ablations (3 validation weeks)", underline=False)
    q3 = tab("q3")
    rows = [["Temporal design", "Score", "± std", "macro-F1", "micro-F1"]]
    rows += [[i] + [fmt(q3.loc[i, c]) for c in ("Score", "± std", "macro-F1", "micro-F1")] for i in q3.index]
    hl = [i + 1 for i, n in enumerate(q3.index) if n in ("+ relational + global volume (all)",
                                                        "History: from Sep 22 (7 wk, ours)",
                                                        "Ensemble, rank-based count (ours)")]
    d.table(s, rows, L(0.3), L(0.75), L(6.05), L(0.262) * len(rows), col_widths=[5.4, 1, 1, 1.1, 1.1], size=9,
            highlight=hl)
    g = lambda n: v(q3, n)
    lg = tab("q2_lgbm")
    d.text(s, L(6.5), L(0.75), L(3.3), L(4.45), [
        ("Take-aways", {"bold": True, "size": 12, "space_after": 3}),
        (f"**Learning from history** beats copying last week: {g('Copy last week (no model)'):.1f} → "
         f"{g('LightGBM: window counts only'):.1f}.", {"size": 10, "space_after": 2}),
        (f"**Global volume context** {v(lg, 'all features') - v(lg, '+ relational features'):+.1f} "
         f"({v(lg, '+ relational features'):.1f} → {v(lg, 'all features'):.1f}): the model learns how complete "
         "recent data is.", {"size": 10, "space_after": 2}),
        (f"**Longer history**: 4 → 7 weeks of training {g('History: from Sep 22 (7 wk, ours)') - g('History: train from Oct 15 (4 wk)'):+.1f}.",
         {"size": 10, "space_after": 2}),
        (f"**Rank-based decision** {g('Ensemble, rank-based count (ours)') - g('Ensemble, probability threshold'):+.1f} "
         "for the ensemble.", {"size": 10, "space_after": 2}),
        (f"**No gain**: decay / recency / rhythm ({g('+ decay, recency, rhythm') - g('LightGBM: window counts only'):+.1f}), "
         "GNN time encodings, the 56-day window, aligning the 11 h gap. Here the amount of recent activity matters "
         "more than its exact timing; the GNN already sums over a graph of past rows only. (RelBench samples a few "
         "neighbours, so it needs a time encoding.)", {"size": 10}),
    ], fill=LIGHT)
    d.text(s, L(0.3), L(0.8) + L(0.262) * len(rows), L(6.05), L(0.5), [
        ("Score (%) mean ± std over 3 weeks; LightGBM and GNN rows use the rank-based decision.",
         {"size": 9, "color": GREY})])


# =================================================================== Q4
def q4_slides(d):
    L = Inches
    s = d.slide(notes=CFG["notes"].get("q4a", ""))
    d.header(s, "(5%) Question 4", "Please provide your framework figure, and describe the full pipeline of your method.")
    d.image(s, FIG / "framework_slide.png", L(0.25), L(1.0), L(6.3), L(4.3))
    d.text(s, L(6.65), L(1.05), L(3.15), L(4.2), CFG["q4_fw_text"], size=10.5, fill=LIGHT)

    s = d.slide(notes=CFG["notes"].get("q4b", ""))
    d.header(s, "Question 4 (cont.) — pipeline, step by step", underline=False)
    d.text(s, L(0.4), L(0.72), L(9.2), L(4.5), CFG["q4_steps"], size=10.5)

    s = d.slide(notes=CFG["notes"].get("q4c", ""))
    d.header(s, "Question 4 (cont.) — results on 3 validation weeks", underline=False)
    fin = tab("final")
    rows = [["Model", "Score", "± std", "macro", "micro"] + [SHORT[k] for k in ACTS]]
    for i in fin.index:
        r = fin.loc[i]
        rows.append([i] + [fmt(r[c]) for c in ("Score", "± std", "macro-F1", "micro-F1")] + [fmt(r[k]) for k in ACTS])
    d.table(s, rows, L(0.3), L(0.8), L(9.4), L(0.33) * len(rows), col_widths=[3.4, .75, .7, .75, .75] + [.85] * 6,
            size=9, highlight=[len(rows) - 1])
    top = L(0.9) + L(0.33) * len(rows)
    d.image(s, FIG / "final_by_week.png", L(0.3), top, L(5.6), L(5.25) - top)
    d.text(s, L(6.05), top + L(0.1), L(3.7), L(5.15) - top, CFG["q4_results_text"], size=10, fill=LIGHT)


def conclusion_slide(d):
    L = Inches
    s = d.slide(notes=CFG["notes"].get("end", ""))
    d.header(s, "Conclusions and limitations", underline=False)
    fin, q2, q3 = tab("final"), tab("q2_gnn"), tab("q3")
    f, base = v(fin, "LightGBM + GNN, rank-based (final)"), v(fin, "Copy last week")
    d.text(s, L(0.4), L(0.8), L(4.5), L(4.35), [
        ("Conclusions", {"bold": True, "size": 13, "space_after": 4}),
        (f"**Final model: {f:.1f}** Score vs {base:.1f} for copying last week ({f - base:+.1f}) on 3 held-out weeks.",
         {"size": 11, "space_after": 3}),
        (f"**Relations (Q2)**: a GNN on the FK graph works only with count-aware aggregation "
         f"({v(q2, 'Hetero GraphSAGE, mean aggregation'):.1f} → {v(q2, '+ mean & sum aggregation (counts)'):.1f}); "
         "2 hops beat 1; graph + feature model together are best.", {"size": 11, "space_after": 3}),
        (f"**Time (Q3)**: leakage-safe daily snapshots, global volume context, a long sliding history "
         f"({v(q3, 'History: train from Oct 15 (4 wk)'):.1f} → {v(q3, 'History: from Sep 22 (7 wk, ours)'):.1f}) "
         f"and a rank-based decision against coverage changes "
         f"({v(q3, 'Ensemble, probability threshold'):.1f} → {v(q3, 'Ensemble, rank-based count (ours)'):.1f}).",
         {"size": 11, "space_after": 3}),
        ("**Negative results** we keep: hand-built relational features, GNN time encodings, user→repo shortcuts "
         "and joint threshold tuning gave no gain here.", {"size": 11}),
    ], fill=LIGHT)
    d.text(s, L(5.1), L(0.8), L(4.5), L(4.35), CFG["limitations"], size=11.5, fill=LIGHT)


if __name__ == "__main__":
    build()
