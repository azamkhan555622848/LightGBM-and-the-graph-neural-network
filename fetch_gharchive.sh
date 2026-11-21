#!/usr/bin/env bash
# Download the GH Archive hour files START..END (UTC, both included), 12 at a time.
# Files already present are kept; damaged files (e.g. from an interrupted run) are downloaded again.
# Usage: ./fetch_gharchive.sh "2026-09-15 00:00" "2026-11-21 04:00" raw
set -u
START="$1"; END="$2"; OUT="${3:-raw}"
PY="$(command -v python3 || command -v python)"
mkdir -p "$OUT"
"$PY" - "$START" "$END" <<'PY' > "$OUT/.urls"
import sys, datetime as dt
s = dt.datetime.fromisoformat(sys.argv[1]); e = dt.datetime.fromisoformat(sys.argv[2])
hours = []
while s <= e:
    hours.append(f"https://data.gharchive.org/{s:%Y-%m-%d}-{s.hour}.json.gz"); s += dt.timedelta(hours=1)
print("\n".join(reversed(hours)))
PY
echo "$(wc -l < "$OUT/.urls") hour files requested"
for attempt in 1 2 3; do
  xargs -P 12 -n 1 wget -q -nc -P "$OUT" < "$OUT/.urls"
  bad=0
  while read -r url; do
    f="$OUT/$(basename "$url")"
    if [ -f "$f" ] && ! gzip -t "$f" 2>/dev/null; then rm -f "$f"; bad=$((bad + 1)); fi
  done < "$OUT/.urls"
  [ "$bad" -eq 0 ] && break
  echo "re-downloading $bad damaged files"
done
missing=0
while read -r url; do [ -f "$OUT/$(basename "$url")" ] || missing=$((missing + 1)); done < "$OUT/.urls"
echo "done: $missing of the requested hour files are not available on GH Archive"
