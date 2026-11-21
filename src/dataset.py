"""Build the snapshot dataset: one row per (user, prediction time t) with features and labels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import ACTS, DAY, activity_log, load_db, snapshot

GHOST = -1  # a user with no rows at all: stands in for test users absent from the data


def population(T, test_input, which="test"):
    """Users row ids to build rows for, and their GitHub user codes.
    which='test': the official test users; 'cohort': all 9,000 cohort users."""
    U = T["Users"]
    code2row = dict(zip(U.github_user_id, U.user_id))
    if which == "test":
        codes = pd.read_csv(test_input)["UserID"].tolist()
    else:
        cohort_file = Path(__file__).resolve().parents[1] / "scripts" / "cohort.txt"
        codes = [int(x) for x in cohort_file.read_text().split()]
    rows = [code2row.get(c, GHOST) for c in codes]
    return pd.DataFrame({"github_user_id": codes, "user": rows})


def snapshot_times(first, last):
    """Daily prediction times at 16:00 UTC (00:00 Taiwan time), like the test week start."""
    return list(pd.date_range(pd.Timestamp(first).normalize() + pd.Timedelta(hours=16),
                              pd.Timestamp(last), freq="D"))


def build(T, log, pop, times, with_labels=True, gap=pd.Timedelta(0)):
    users = np.unique(np.append(pop.user.values, GHOST))
    frames = []
    for t in times:
        X = snapshot(T, log, users, t, with_labels=with_labels, gap=gap)
        X = pop.join(X, on="user")           # test users absent from the data -> GHOST row
        frames.append(X)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="folder with the 8 parquet tables")
    ap.add_argument("--test-input", default="submission/test_input.csv")
    ap.add_argument("--population", default="test", choices=["test", "cohort"])
    ap.add_argument("--first", required=True, help="first snapshot day (UTC)")
    ap.add_argument("--data-end", required=True, help="end of the downloaded data (UTC hour)")
    ap.add_argument("--predict-at", default="2026-11-21 16:00", help="test week start (UTC)")
    ap.add_argument("--gap-hours", type=float, default=None,
                    help="feature cut-off this many hours before each prediction time "
                         "(default: predict-at minus data-end, the gap the test row has)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    T = load_db(a.db)
    log = activity_log(T)
    pop = population(T, a.test_input, a.population)
    end, t_test = pd.Timestamp(a.data_end), pd.Timestamp(a.predict_at)
    gap = (pd.Timedelta(hours=a.gap_hours) if a.gap_hours is not None
           else max(t_test - end, pd.Timedelta(0)))
    times = snapshot_times(a.first, end - 7 * DAY)          # label window must be complete
    print(f"{len(pop)} users ({(pop.user == GHOST).sum()} without any row), "
          f"{len(times)} labelled snapshots {times[0]} .. {times[-1]}, feature gap {gap}")
    train = build(T, log, pop, times, gap=gap)
    test = build(T, log, pop, [t_test], with_labels=False, gap=gap)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    train.to_parquet(Path(a.out) / "train.parquet")
    test.to_parquet(Path(a.out) / "test.parquet")
    print("label rates:", train.filter(like="y_").mean().round(3).to_dict())


if __name__ == "__main__":
    main()
