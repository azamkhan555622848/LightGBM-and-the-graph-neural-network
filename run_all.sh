#!/usr/bin/env bash
# Reproduce output_STUDENTID.csv from scratch: download -> preprocess -> snapshots -> model -> check.
#
#   ./run_all.sh 413540002
#
# Data period (UTC, GH Archive hour files, both ends included). Taiwan time = UTC + 8.
set -euo pipefail
STUDENT_ID="${1:?usage: ./run_all.sh STUDENTID}"
FIRST_HOUR="2026-09-15 00:00"          # first GH Archive hour file used
LAST_HOUR="2026-11-21 04:00"           # last GH Archive hour file used (12:00 Taiwan)
FIRST_SNAPSHOT="2026-09-22"            # first daily training snapshot (16:00 UTC = 00:00 Taiwan)
PREDICT_AT="2026-11-21 16:00"          # test week start: 2026-11-22 00:00 Taiwan time
PY="${PYTHON:-$(command -v python3 || command -v python)}"
export GNN_THREADS="${GNN_THREADS:-16}" LGB_THREADS="${LGB_THREADS:-16}"

DATA_END=$(date -u -d "$LAST_HOUR UTC + 1 hour" "+%Y-%m-%d %H:%M")   # end of the last hour file

# 1. download (skips files already there, re-downloads damaged ones)
./fetch_gharchive.sh "$FIRST_HOUR" "$LAST_HOUR" raw

# 2. the 8 tables, from exactly the hour files of the period (other files in raw/ are ignored)
rm -rf raw_period && mkdir raw_period
"$PY" - "$FIRST_HOUR" "$LAST_HOUR" <<'PY'
import datetime as dt, os, sys
s, e = (dt.datetime.fromisoformat(x) for x in sys.argv[1:3])
n = 0
while s <= e:
    name = f"{s:%Y-%m-%d}-{s.hour}.json.gz"
    if os.path.exists(f"raw/{name}"):
        os.symlink(os.path.abspath(f"raw/{name}"), f"raw_period/{name}"); n += 1
    s += dt.timedelta(hours=1)
print(f"{n} hour files in the period")
PY
"$PY" scripts/preprocess.py --raw raw_period --out data

# 3. daily snapshots: features (rows before t - gap) and labels ([t, t + 7 days))
"$PY" src/dataset.py --db data/db --first "$FIRST_SNAPSHOT" --data-end "$DATA_END" \
    --predict-at "$PREDICT_AT" --out data/snapshots

# 4. LightGBM + GNN, rank-based decision -> output file
"$PY" src/predict.py --db data/db --data data/snapshots --out "output_${STUDENT_ID}.csv" --save-dir data/final

# 5. format check with the course script
"$PY" scripts/evaluate.py --pred "output_${STUDENT_ID}.csv" --input submission/test_input.csv
