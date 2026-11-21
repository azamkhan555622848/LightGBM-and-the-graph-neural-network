"""Figures for the slides (data volume over time, experiment results, feature importance).

    python src/figures.py --db data/db --results results --out figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from features import ACTS, activity_log, load_db  # noqa: E402

COLORS = {"open_pr": "#1f77b4", "open_issue": "#ff7f0e", "issue_comment": "#2ca02c",
          "pr_review": "#d62728", "pr_review_comment": "#9467bd", "push": "#8c564b"}


def weekly_volume(db, out):
    T = load_db(db)
    log = activity_log(T)
    a = log[log.kind.isin(ACTS) & (log.t >= T["Users"].first_seen_at.min())]
    day = pd.crosstab(a.t.dt.floor("D"), a.kind)[ACTS]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k in ACTS:
        ax.plot(day.index, day[k].rolling(7, min_periods=1).mean(), label=k, color=COLORS[k], lw=2)
    ax.set_yscale("log")
    ax.set_ylabel("cohort actions per day (7-day mean, log)")
    ax.axvspan(pd.Timestamp("2026-11-21 16:00"), pd.Timestamp("2026-11-28 16:00"), color="#f2c14e", alpha=.35,
               label="test week")
    ax.annotate("push events almost vanish\nfrom GH Archive", xy=(pd.Timestamp("2026-11-01"), 8),
                xytext=(pd.Timestamp("2026-10-08"), 12), fontsize=9, color="#8c564b",
                arrowprops=dict(arrowstyle="->", color="#8c564b"))
    ax.set_xlim(day.index.min(), pd.Timestamp("2026-11-29"))
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.grid(alpha=.3)
    ax.set_title("Activity of the 9,000 cohort users in the 8 tables (UTC days)")
    fig.tight_layout()
    fig.savefig(Path(out) / "weekly_volume.png", dpi=160)
    day.to_csv(Path(out) / "daily_volume.csv")


def ablation_bars(csvs, names, out, fname, title):
    rows = []
    for c in csvs:
        r = pd.read_csv(c)
        rows.append(r.groupby("model", sort=False)[["score", "macro_f1", "micro_f1"]].mean())
    r = pd.concat(rows)
    r = r.loc[[n for n in names if n in r.index]]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(r) + 1.2))
    ax.barh(range(len(r)), r.score * 100, color="#3b6ea5")
    for i, v in enumerate(r.score * 100):
        ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=9)
    ax.set_yticks(range(len(r)), r.index)
    ax.invert_yaxis()
    ax.set_xlabel("Score = (macro-F1 + micro-F1) / 2, %  (mean of 4 validation weeks)")
    ax.set_title(title)
    ax.set_xlim(0, max(r.score * 100) + 6)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(Path(out) / fname, dpi=160)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    Path(a.out).mkdir(exist_ok=True)
    weekly_volume(a.db, a.out)
    timeline(a.out)
    result_figures(a.out)



def timeline(out):
    """Question 1: the data period and how it is used (UTC)."""
    import matplotlib.dates as mdates
    D = pd.Timestamp
    rows = [
        ("GH Archive data used", D("2026-09-15 00:00"), D("2026-11-21 05:00"), "#3b6ea5"),
        ("Training prediction times\n(daily, final model)", D("2026-09-22 16:00"), D("2026-11-06 16:00"), "#7aa6d6"),
        ("Threshold week (final)", D("2026-11-13 16:00"), D("2026-11-20 16:00"), "#d98c3a"),
        ("Validation weeks\n(experiments)", None, None, "#5a9e5a"),
        ("Feature gap 11 h", D("2026-11-21 05:00"), D("2026-11-21 16:00"), "#999999"),
        ("Test week (Nov 22-29 Taiwan)", D("2026-11-21 16:00"), D("2026-11-28 16:00"), "#e0b33c"),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i, (name, s, e, c) in enumerate(rows):
        y = len(rows) - 1 - i
        if name.startswith("Validation"):
            for vs in ("2026-10-29 16:00", "2026-11-05 16:00", "2026-11-12 16:00"):
                ax.barh(y, pd.Timedelta(days=7), left=D(vs), height=0.55, color=c, edgecolor="white")
        else:
            ax.barh(y, e - s, left=s, height=0.55, color=c)
        ax.text(D("2026-09-13"), y, name, ha="right", va="center", fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(D("2026-09-13"), D("2026-11-30"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SU))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=.3)
    ax.set_title("Dataset time period (UTC): 2026-09-15 00:00 to 2026-11-21 05:00 = 67 days, 1,613 hour files", fontsize=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.12)
    fig.savefig(Path(out) / "timeline.png", dpi=170)



def result_figures(out, results="results"):
    """Per-week bars for the final models and the GNN aggregation ablation; LightGBM gain share."""
    R = Path(results)
    de = pd.read_csv(R / "decision_final.csv")
    lg = pd.read_csv(R / "lgbm_rules.csv")
    gn = pd.read_csv(R / "gnn_rules.csv")
    wk = lambda s: pd.Timestamp(s).strftime("week of %b %-d")

    def grouped(series, path, title, colors, ylim=None):
        weeks = sorted(series[0][1].keys())
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        n, w = len(series), 0.8 / len(series)
        for i, (name, vals) in enumerate(series):
            xs = [j + (i - (n - 1) / 2) * w for j in range(len(weeks))]
            ys = [100 * vals[k] for k in weeks]
            ax.bar(xs, ys, w * 0.95, label=name, color=colors[i])
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.3, f"{y:.1f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(weeks)), [wk(k) + " (UTC 16:00)" for k in weeks], fontsize=8)
        ax.set_ylabel("Score (%)", fontsize=8)
        ax.set_ylim(*(ylim or (20, 45)))
        ax.legend(fontsize=7, ncol=n, loc="upper left", frameon=False)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    sel = lambda df, m, r: df[(df.model == m) & ((df.rule == r) if r else True)].set_index("val_t").score.to_dict()
    grouped([("Copy last week", sel(lg, "copy_last_week", None)), ("LightGBM", sel(de, "lgbm", "count")),
             ("GNN (3 seeds)", sel(de, "gnn", "count")), ("LightGBM + GNN (final)", sel(de, "lgbm+gnn", "count"))],
            Path(out) / "final_by_week.png", "Score per validation week (rank-based decision)",
            ["#9aa5b1", "#3b6ea5", "#5a9e5a", "#e0a020"])
    grouped([("mean aggregation", sel(gn, "gnn_mean_agg", "count")),
             ("mean + sum aggregation", sel(gn, "gnn_count_agg", "count"))],
            Path(out) / "gnn_agg_by_week.png", "GNN Score per validation week", ["#9aa5b1", "#3b6ea5"])

    gs = pd.read_csv(R / "summary" / "gain_share.csv", index_col=0)
    order = ["temporal: window counts", "temporal: decayed counts", "temporal: recency",
             "temporal: activity pattern", "relational", "global (data volume)"]
    colors = ["#c6d8ef", "#3b6ea5", "#7aa6d6", "#a8c686", "#e0a020", "#c0504d"]
    gs = gs.loc[order]
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    left = pd.Series(0.0, index=gs.columns)
    for g, c in zip(order, colors):
        ax.barh(range(len(gs.columns)), gs.loc[g] * 100, left=left * 100, color=c,
                label=g.replace("temporal: activity pattern", "temporal: rhythm"))
        left += gs.loc[g]
    ax.set_yticks(range(len(gs.columns)), gs.columns, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of LightGBM split gain (%)", fontsize=8)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(Path(out) / "gain_share.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
