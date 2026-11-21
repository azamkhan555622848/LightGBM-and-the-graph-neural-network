"""Models, thresholds and scoring shared by the experiments and the final prediction."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import scores  # noqa: E402  (the course's own scoring code)

from features import ACTS  # noqa: E402

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=50,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=int(__import__("os").environ.get("LGB_THREADS", "16")), seed=0, deterministic=True, force_row_wise=True)
NUM_ROUNDS = 400


def feature_columns(df, groups=("temporal", "relational", "global")):
    cols = [c for c in df.columns if c not in ("github_user_id", "user", "t", "t_feat") and not c.startswith("y_")]
    keep = []
    for c in cols:
        g = ("global" if c.startswith("global") else
             "relational" if c in RELATIONAL else "temporal")
        if g in groups:
            keep.append(c)
    return keep


RELATIONAL = {
    "n_repos_28d", "n_repos_84d", "own_repo_share", "org_repo_share", "push_own_repo_28d",
    "repo_others_heat7d", "repo_max_heat7d", "n_hot_repos7d", "repo_coworkers7d",
    "open_prs_authored", "open_prs_authored_28d", "prs_merged_share_28d", "cross_repo_pr_share",
    "open_issues_authored", "open_issues_assigned", "outbound_share", "n_collaborators",
    "collab_act7d"} | {f"inbound{w}d_{k}" for w in (3, 7, 28)
                       for k in ("issue_comment", "pr_review", "pr_review_comment", "merge_pr")}


LGB_PARAMS_REG = {**LGB_PARAMS, "num_leaves": 15, "min_child_samples": 200, "feature_fraction": 0.5,
                  "lambda_l2": 10.0}


def fit_lgbm(train, cols, rounds=NUM_ROUNDS, params=LGB_PARAMS, valid=None, max_rounds=1500):
    """One binary LightGBM per activity. With `valid` (the threshold week), the number of
    trees is chosen by early stopping on its log-loss instead of a fixed `rounds`."""
    models = {}
    for k in ACTS:
        ds = lgb.Dataset(train[cols].astype("float32"), label=train[f"y_{k}"])
        if valid is None:
            models[k] = lgb.train(params, ds, num_boost_round=rounds)
        else:
            dv = lgb.Dataset(valid[cols].astype("float32"), label=valid[f"y_{k}"], reference=ds)
            models[k] = lgb.train({**params, "metric": "binary_logloss"}, ds, num_boost_round=max_rounds,
                                  valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    return models


def predict_lgbm(models, df, cols):
    X = df[cols].astype("float32")
    return pd.DataFrame({k: m.predict(X) for k, m in models.items()}, index=df.index)


def best_thresholds(prob, truth):
    """Per activity, the probability threshold that maximizes F1 on (prob, truth)."""
    th = {}
    for k in ACTS:
        p, y = prob[k].to_numpy(), truth[f"y_{k}"].to_numpy()
        cand = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 300)))
        best = (0.0, 0.5)
        for c in cand:
            pred = p >= c
            tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
            f = 2 * tp / max(2 * tp + fp + fn, 1)
            if f > best[0]:
                best = (f, c)
        th[k] = best[1]
    return th


def best_thresholds_joint(prob, truth, rounds=3):
    """Thresholds tuned together to maximize the course Score = (macro-F1 + micro-F1) / 2,
    by coordinate ascent starting from the per-activity F1 optimum."""
    th = best_thresholds(prob, truth)
    P = {k: prob[k].to_numpy() for k in ACTS}
    Y = {k: truth[f"y_{k}"].to_numpy() == 1 for k in ACTS}

    def counts(k, c):
        pred = P[k] >= c
        return np.array([(pred & Y[k]).sum(), (pred & ~Y[k]).sum(), (~pred & Y[k]).sum()])

    cur = {k: counts(k, th[k]) for k in ACTS}

    def total(cnt):
        f1s = [2 * c[0] / max(2 * c[0] + c[1] + c[2], 1) for c in cnt.values()]
        s = sum(cnt.values())
        return (np.mean(f1s) + 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)) / 2

    best = total(cur)
    for _ in range(rounds):
        changed = False
        for k in ACTS:
            for c in np.unique(np.quantile(P[k], np.linspace(0.5, 0.999, 300))):
                trial = dict(cur); trial[k] = counts(k, c)
                v = total(trial)
                if v > best + 1e-9:
                    best, cur, th[k], changed = v, trial, c, True
        if not changed:
            break
    return th


def binarize(prob, th):
    return pd.DataFrame({k: (prob[k] >= th[k]).astype(int) for k in ACTS}, index=prob.index)


def score(pred01, truth, ids):
    """The course's Score on one snapshot: pred01 / truth indexed alike, ids = UserIDs."""
    p = pd.DataFrame({"UserID": ids.values, **{k: pred01[k].values for k in ACTS}})
    t = pd.DataFrame({"UserID": ids.values, **{k: truth[f"y_{k}"].values for k in ACTS}})
    return scores(p, t)
