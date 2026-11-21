#!/usr/bin/env bash
# GNN ablations (Questions 2 and 3): 3 validation weeks, one run per variant, 4 at a time.
#   ./run_gnn_ablation.sh DATA/snapshots/train.parquet results/gnn_ablation
set -u
DATA="${1:-data/snapshots/train.parquet}"; OUT="${2:-results/gnn_ablation}"; mkdir -p "$OUT" logs
run() { name=$1; shift; GNN_THREADS=${GNN_THREADS:-5} .venv/bin/python src/train_gnn.py --db data/db --data "$DATA" \
        --name "$name" --out "$OUT" "$@" > "logs/$name.log" 2>&1; }
# batch A: aggregation, shortcut edges, global context
run gnn_mean_agg        --aggr mean --time-enc none &
run gnn_count_agg       --time-enc none &
run gnn_shortcuts       --time-enc none --shortcuts &
run gnn_shortcuts_ctx   --time-enc none --shortcuts --global-ctx &
wait
# batch B: time encodings, hops, lookback window
run gnn_sc_bucket       --time-enc bucket --shortcuts &
run gnn_sc_sincos       --time-enc sincos --shortcuts &
run gnn_sc_1layer       --time-enc none --shortcuts --layers 1 &
run gnn_all_history     --time-enc none --shortcuts --lookback 400 &
wait
echo "GNN ablation done $(date)"
