"""Turn the 8 tables into (user, prediction time) training rows: features + 6 labels.

Every feature for prediction time t uses only rows whose row time is before t, and the
"known only later" columns (merged_at, merged_by_id, closed_at, state_reason, dismissed_at)
only when their value is before t. Labels are the 6 activities in [t, t + 7 days).
"""
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments",
          "PRReviews", "PRReviewComments", "Pushes"]
ACTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
DAY = pd.Timedelta(days=1)
WINDOWS = [1, 3, 7, 14, 28, 56]          # days, for counts
HALF_LIVES = [2, 7, 21]                  # days, for exponentially decayed counts
LOOKBACK = 84                            # days of history used by any feature


def load_db(db_dir):
    return {t: pd.read_parquet(Path(db_dir) / f"{t}.parquet") for t in TABLES}


def activity_log(T):
    """One row per action of a cohort user.

    user  Users row id of the actor          kind  one of ACTS (+ 'merge_pr', 'close_item')
    t     when the action happened          rt    row time: from when the row may be used
    repo, pr, issue: the objects involved (row ids, may be NA)
    owner: Users row id of the author of the PR / issue acted on (for 'who did I respond to')
    """
    U, R = T["Users"], T["Repositories"]
    pr, iss = T["PullRequests"], T["Issues"]
    ic, rv, rc, pu = T["IssueComments"], T["PRReviews"], T["PRReviewComments"], T["Pushes"]
    na = pd.Series(pd.NA, dtype="Int64")

    def frame(user, kind, t, rt, repo, pr_=None, issue=None, owner=None):
        n = len(user)
        col = lambda x: na.reindex(range(n)).values if x is None else pd.array(np.asarray(x), dtype="Int64")
        return pd.DataFrame({"user": pd.array(np.asarray(user), dtype="Int64"), "kind": kind,
                             "t": np.asarray(t, dtype="datetime64[ns]"),
                             "rt": np.asarray(rt, dtype="datetime64[ns]"),
                             "repo": col(repo), "pr": col(pr_), "issue": col(issue), "owner": col(owner)})

    pr_author = pr.set_index("pr_id")["author_id"]
    iss_author = iss.set_index("issue_id")["author_id"]
    opened = pr[pr.author_id.notna() & pr.created_at.notna()]
    issues_opened = iss[~iss.is_pull_request.astype(bool) & iss.created_at.notna()]
    merged = pr[pr.merged_by_id.notna() & pr.merged_at.notna()]
    parts = [
        frame(opened.author_id, "open_pr", opened.created_at, opened.first_seen_at, opened.repo_id,
              opened.pr_id, None, opened.author_id),
        frame(issues_opened.author_id, "open_issue", issues_opened.created_at,
              issues_opened.first_seen_at, issues_opened.repo_id, None, issues_opened.issue_id,
              issues_opened.author_id),
        frame(ic.author_id, "issue_comment", ic.created_at, ic.created_at, ic.repo_id,
              iss.set_index("issue_id")["pull_request_id"].reindex(ic.issue_id).values,
              ic.issue_id, iss_author.reindex(ic.issue_id).values),
        frame(rv.reviewer_id, "pr_review", rv.first_seen_at, rv.first_seen_at, rv.repo_id,
              rv.pr_id, None, pr_author.reindex(rv.pr_id).values),
        frame(rc.author_id, "pr_review_comment", rc.created_at, rc.created_at, rc.repo_id,
              rc.pr_id, None, pr_author.reindex(rc.pr_id).values),
        frame(pu.user_id, "push", pu.created_at, pu.created_at, pu.repo_id),
        # merged_at is "known only later": the merge row time is the merge itself
        frame(merged.merged_by_id, "merge_pr", merged.merged_at, merged.merged_at, merged.repo_id,
              merged.pr_id, None, merged.author_id),
    ]
    log = pd.concat(parts, ignore_index=True)
    log = log[log.user.notna()]
    coh = set(U.user_id[U.is_cohort.astype(bool)])
    log = log[log.user.isin(coh)].copy()
    log["user"] = log.user.astype("int64")
    # repo ownership: the repo owner's login equals the actor's login
    login = U.set_index("user_id")["login"]
    owner_login = R.set_index("repo_id")["owner_login"]
    log["own_repo"] = (owner_login.reindex(log.repo).values == login.reindex(log.user).values)
    log["is_org"] = R.set_index("repo_id")["org_id"].notna().reindex(log.repo).fillna(False).values
    log["self_item"] = (log.owner == log.user).fillna(False).astype(bool)
    return log.sort_values("t", ignore_index=True)


def labels(log, users, t):
    """6 binary labels: did the user do each activity in [t, t + 7 days)?"""
    w = log[(log.t >= t) & (log.t < t + 7 * DAY) & log.kind.isin(ACTS)]
    y = pd.crosstab(w.user, w.kind).reindex(index=users, columns=ACTS, fill_value=0)
    return (y > 0).astype("int8")


def _counts(df, users, prefix, by="kind", kinds=ACTS):
    c = pd.crosstab(df.user, df[by]).reindex(index=users, columns=kinds, fill_value=0)
    c.columns = [f"{prefix}{k}" for k in kinds]
    return c


def temporal_features(log, users, t):
    """Per-user history of the 6 activities: windowed counts, decayed counts, recency,
    active days, trend, regularity and time-of-week profile."""
    hist = log[(log.rt < t) & (log.t < t) & (log.t >= t - LOOKBACK * DAY)]
    act = hist[hist.kind.isin(ACTS)]
    age = (t - act.t) / DAY
    feats = []
    for w in WINDOWS:
        feats.append(_counts(act[age < w], users, f"cnt{w}d_"))
    for h in HALF_LIVES:
        wgt = np.power(0.5, age / h)
        d = pd.DataFrame({"user": act.user, "kind": act.kind, "w": wgt})
        p = d.pivot_table(index="user", columns="kind", values="w", aggfunc="sum")
        p = p.reindex(index=users, columns=ACTS).fillna(0.0)
        p.columns = [f"decay{h}d_{k}" for k in ACTS]
        feats.append(p)
    last = act.groupby(["user", "kind"]).t.max().unstack().reindex(index=users, columns=ACTS)
    rec = ((t - last) / DAY).fillna(LOOKBACK * 2)
    rec.columns = [f"days_since_{k}" for k in ACTS]
    feats.append(rec)
    first = act.groupby("user").t.min().reindex(users)
    lastany = act.groupby("user").t.max().reindex(users)
    day = act.t.dt.floor("D")
    f = pd.DataFrame(index=users)
    f["days_since_any"] = ((t - lastany) / DAY).fillna(LOOKBACK * 2)
    f["days_since_first"] = ((t - first) / DAY).fillna(0)
    for w in (7, 28, 84):
        m = age < w
        f[f"active_days{w}d"] = day[m].groupby(act.user[m]).nunique().reindex(users).fillna(0)
    wk = ((t - act.t) // (7 * DAY)).astype(int)
    f["active_weeks12"] = wk.groupby(act.user).nunique().reindex(users).fillna(0)
    c7 = (age < 7).groupby(act.user).sum().reindex(users).fillna(0)
    c28 = (age < 28).groupby(act.user).sum().reindex(users).fillna(0)
    f["trend_7_28"] = (c7 + 1) / (c28 / 4 + 1)
    # mean gap between active days (regularity) and weekend share (Taiwan time)
    days = pd.DataFrame({"user": act.user, "day": day}).drop_duplicates().sort_values(["user", "day"])
    gaps = days.groupby("user").day.diff() / DAY
    f["mean_gap_days"] = gaps.groupby(days.user).mean().reindex(users).fillna(LOOKBACK)
    tw = act.t + pd.Timedelta(hours=8)
    f["weekend_share"] = (tw.dt.dayofweek >= 5).groupby(act.user).mean().reindex(users).fillna(0)
    f["n_kinds_28d"] = act[age < 28].groupby("user").kind.nunique().reindex(users).fillna(0)
    feats.append(f)
    return pd.concat(feats, axis=1)


def relational_features(T, log, users, t):
    """Features that follow foreign keys to other tables: repositories the user works in, the
    state of the user's own PRs and issues, what other cohort users did on them, and how
    active the user's repositories and collaborators are."""
    hist = log[(log.rt < t) & (log.t < t) & (log.t >= t - LOOKBACK * DAY)]
    age = (t - hist.t) / DAY
    f = pd.DataFrame(index=users)

    # --- User -> Repositories (via every activity table) ---
    h28 = hist[age < 28]
    f["n_repos_28d"] = h28.groupby("user").repo.nunique().reindex(users).fillna(0)
    f["n_repos_84d"] = hist.groupby("user").repo.nunique().reindex(users).fillna(0)
    f["own_repo_share"] = hist.groupby("user").own_repo.mean().reindex(users).fillna(0)
    f["org_repo_share"] = hist.groupby("user").is_org.mean().reindex(users).fillna(0)
    f["push_own_repo_28d"] = h28[(h28.kind == "push") & h28.own_repo].groupby("user").size().reindex(users).fillna(0)
    # repo heat: what everybody (other cohort users) did in my repos during the last 7 days
    ur = hist[["user", "repo"]].dropna().drop_duplicates()
    last7 = hist[age < 7]
    repo_heat = last7.groupby("repo").size()
    repo_users = last7.groupby("repo").user.nunique()
    ur = ur.assign(heat=repo_heat.reindex(ur.repo).fillna(0).values,
                   others=(repo_users.reindex(ur.repo).fillna(0).values))
    mine7 = last7.groupby(["user", "repo"]).size()
    ur["mine"] = mine7.reindex(pd.MultiIndex.from_frame(ur[["user", "repo"]])).fillna(0).values
    ur["others_heat"] = ur.heat - ur.mine
    g = ur.groupby("user")
    f["repo_others_heat7d"] = g.others_heat.sum().reindex(users).fillna(0)
    f["repo_max_heat7d"] = g.heat.max().reindex(users).fillna(0)
    f["n_hot_repos7d"] = g.heat.apply(lambda s: (s > 0).sum()).reindex(users).fillna(0)
    f["repo_coworkers7d"] = (g.others.sum() - g.mine.apply(lambda s: (s > 0).sum())).reindex(users).fillna(0)

    # --- User -> PullRequests they authored (PullRequests.author_id) ---
    pr = T["PullRequests"]
    pr = pr[(pr.first_seen_at < t) & pr.author_id.notna()]
    closed = pr.closed_at.notna() & (pr.closed_at < t)
    merged = pr.merged_at.notna() & (pr.merged_at < t)
    open_pr = pr[~closed]
    recent = open_pr[open_pr.created_at.notna() & (open_pr.created_at >= t - 28 * DAY)]
    f["open_prs_authored"] = open_pr.groupby("author_id").size().reindex(users).fillna(0)
    f["open_prs_authored_28d"] = recent.groupby("author_id").size().reindex(users).fillna(0)
    pr28 = pr[pr.created_at.notna() & (pr.created_at >= t - 28 * DAY) & (pr.created_at < t)]
    f["prs_merged_share_28d"] = (merged.reindex(pr28.index)).groupby(pr28.author_id).mean().reindex(users).fillna(0)
    f["cross_repo_pr_share"] = pr28.is_cross_repo.astype(float).groupby(pr28.author_id).mean().reindex(users).fillna(0)

    # --- User -> Issues they authored / are assigned (Issues.author_id, assignee_id) ---
    iss = T["Issues"]
    iss = iss[iss.first_seen_at < t]
    iss_open = iss[~(iss.closed_at.notna() & (iss.closed_at < t))]
    f["open_issues_authored"] = iss_open[~iss_open.is_pull_request.astype(bool)].groupby("author_id").size().reindex(users).fillna(0)
    f["open_issues_assigned"] = iss_open.groupby("assignee_id").size().reindex(users).fillna(0)

    # --- Other cohort users acting on my PRs / issues (2 hops: me <- item <- them) ---
    inbound = hist[hist.owner.notna() & ~hist.self_item & hist.kind.isin(
        ["issue_comment", "pr_review", "pr_review_comment", "merge_pr"])]
    in_age = (t - inbound.t) / DAY
    for w in (3, 7, 28):
        m = in_age < w
        c = inbound[m].groupby(["owner", "kind"]).size().unstack()
        c = c.reindex(index=users, columns=["issue_comment", "pr_review", "pr_review_comment", "merge_pr"]).fillna(0)
        c.columns = [f"inbound{w}d_{k}" for k in c.columns]
        f = f.join(c)
    # --- Me acting on other people's items (reviewer / commenter role) ---
    outbound = hist[hist.owner.notna() & ~hist.self_item]
    f["outbound_share"] = (outbound.groupby("user").size().reindex(users).fillna(0)
                           / hist.groupby("user").size().reindex(users).fillna(0).clip(lower=1))
    # --- Collaborators: users I interacted with, and how active they were last week ---
    pairs = pd.concat([
        outbound[["user", "owner"]].rename(columns={"user": "a", "owner": "b"}),
        inbound[["owner", "user"]].rename(columns={"owner": "a", "user": "b"}),
    ]).dropna().astype("int64").drop_duplicates()
    act7 = hist[(age < 7) & hist.kind.isin(ACTS)].groupby("user").size()
    pairs["b_act7"] = act7.reindex(pairs.b).fillna(0).values
    f["n_collaborators"] = pairs.groupby("a").size().reindex(users).fillna(0)
    f["collab_act7d"] = pairs.groupby("a").b_act7.sum().reindex(users).fillna(0)
    return f


def global_features(log, t):
    """Cohort-wide volume of each activity (same value for every user at time t). GH Archive
    lost much of its data from mid-October 2026, so this tells the model how complete the
    recent data is."""
    act = log[(log.rt < t) & (log.t < t) & log.kind.isin(ACTS)]
    age = (t - act.t) / DAY
    g = {}
    for k in ACTS:
        a = age[act.kind == k]
        g[f"global7d_{k}"] = float((a < 7).sum())
        g[f"global_ratio_{k}"] = float((a < 7).sum() + 1) / float((a < 28).sum() / 4 + 1)
    return g


def snapshot(T, log, users, t, with_labels=True, relational=True, gap=pd.Timedelta(0)):
    """Features from the data before t_feat = t - gap; labels in [t, t + 7 days).
    gap: the final data ends some hours before the test week starts (GH Archive lag, and we
    must run before the deadline), so training rows are built with the same gap."""
    tf = t - gap
    X = temporal_features(log, users, tf)
    if relational:
        X = X.join(relational_features(T, log, users, tf))
    for k, v in global_features(log, tf).items():
        X[k] = v
    X.insert(0, "t", t)
    X.insert(1, "t_feat", tf)
    if with_labels:
        X = X.join(labels(log, users, t).add_prefix("y_"))
    return X


def build_snapshots(T, log, users, times, **kw):
    return pd.concat([snapshot(T, log, users, t, **kw) for t in times])
