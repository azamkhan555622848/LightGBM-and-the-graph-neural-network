# Predicting GitHub User Activity — Big Data Assignment 2 (413540002)

Azam Khan · Student ID 413540002 · https://github.com/azamkhan555622848/LightGBM-and-the-graph-neural-network

For each of the 4,465 test users, predict which of 6 activities (`open_pr`, `open_issue`,
`issue_comment`, `pr_review`, `pr_review_comment`, `push`) the user performs during the test week
**2026/11/22 00:00 – 11/29 00:00 (Taiwan time)**, using only the 8 relational tables built by the
course's `scripts/preprocess.py`.

## Data Source

The data comes from **[GH Archive](https://www.gharchive.org/)** (https://www.gharchive.org/), a
public archive of GitHub events. The 8 tables are built from the GH Archive hour files with the
course's preprocessing script (`scripts/preprocess.py`, unchanged), which keeps the 9,000 cohort
users and anonymizes the data. No other data (other event types, GitHub API, other websites) is
used. This work is for educational use within the course.

**Time period used:** GH Archive hour files **2026-09-15 00:00 → 2026-11-21 04:00 (UTC)**, both
included (1,613 files; Taiwan time 2026-09-15 08:00 → 2026-11-21 12:59).

## How to run

Requirements: Python 3.10+, the packages in `requirements.txt`, and Graphviz (`dot`) for the figures.
Everything runs on CPU.

```bash
pip install -r requirements.txt
./run_all.sh 413540002          # download -> tables -> snapshots -> models -> output_413540002.csv
```

`run_all.sh` runs these steps; each can also be run on its own:

| Step | Command | Output |
|---|---|---|
| 1. Download GH Archive | `./fetch_gharchive.sh "2026-09-15 00:00" "2026-11-21 04:00" raw` | `raw/*.json.gz` (1,613 hour files, ~17 GB) |
| 2. Build the 8 tables | `python scripts/preprocess.py --raw raw --out data` | `data/db/*.parquet` |
| 3. Snapshots (features + labels) | `python src/dataset.py --db data/db --first 2026-09-22 --data-end "2026-11-21 05:00" --out data/snapshots` | `train.parquet`, `test.parquet` |
| 4. Train, pick thresholds, predict | `python src/predict.py --db data/db --data data/snapshots --out output_413540002.csv` | `output_413540002.csv` |
| 5. Format check | `python scripts/evaluate.py --pred output_413540002.csv --input submission/test_input.csv` | |

Reproducibility: LightGBM runs with `deterministic=True`; the GNN uses fixed seeds and a fixed
number of CPU threads (`GNN_THREADS`, default 16). With the same thread count two runs agree to
~1e-7; a different thread count can change probabilities by ~1e-3, which may flip a few
borderline users.

## Method

See the slides (`slides/Asg2_413540002.pdf`) for the full description and the experiments.

1. **Time-aware samples** (`src/dataset.py`, `src/features.py`). One prediction time `t` per day at
   16:00 UTC (= 00:00 Taiwan, like the test week). Features use only rows whose row time is before
   the cut-off `t − gap` (and the "known only later" columns only when their value is before it);
   labels are the 6 activities in `[t, t + 7 days)`. The gap (11 h) is the distance between the
   end of our data and the start of the test week, so training rows look like the test row.
2. **Relational-temporal features for LightGBM** (112): multi-scale window counts, decayed counts,
   recency and rhythm of each activity (temporal); repositories, own PRs/issues and their state,
   other users acting on them, collaborators (relational); cohort-wide data volume (global).
3. **Temporal heterogeneous GNN** (`src/gnn.py`): every table is a node type, every foreign key an
   edge type (+ reverse); the graph at a cut-off holds only the rows visible then (event rows of
   the last 56 days); 2-layer heterogeneous GraphSAGE with mean + sum aggregation (so messages can
   count rows), 64 hidden units, trained full-batch on one graph per daily snapshot.
4. **Decision** (`src/predict.py`): average of the LightGBM and GNN probabilities (3 seeds each). On
   the last labelled week the F1-best threshold of each activity selects `n_k` users; in the test
   week the `n_k` users with the highest probability get 1. This rank-based rule is robust to the
   week-to-week changes of GH Archive coverage, which shift the probabilities.

Validation (`src/experiments.py`, `src/train_gnn.py`, `src/decision_eval.py`): 3 non-overlapping
weeks starting 2026-10-29, 11-05 and 11-12 16:00 UTC; for a validation week starting `tv`, train
on `t ≤ tv − 14 d` and choose the thresholds / `n_k` on `t = tv − 7 d` — the same recipe as the final
run. Scores use the course's `scripts/evaluate.py` functions.

| Model (rank-based decision unless noted) | Score | macro-F1 | micro-F1 |
|---|---|---|---|
| Copy last week (baseline) | 32.3 | 32.6 | 32.0 |
| LightGBM | 36.8 | 36.9 | 36.6 |
| GNN (3 seeds) | 36.6 | 36.7 | 36.5 |
| LightGBM + GNN, probability threshold | 36.2 | 35.9 | 36.6 |
| **LightGBM + GNN, rank-based (final)** | **37.4** | **37.6** | **37.2** |

Means over the 3 validation weeks (%). All ablations are in `results/summary/`
(`q2_gnn_table.csv`, `q2_lgbm_table.csv`, `q3_table.csv`, `final_table.csv`).

## Repository layout

```
run_all.sh, fetch_gharchive.sh   end-to-end pipeline, GH Archive download
run_gnn_ablation.sh              the GNN experiments of the slides
scripts/                         course scripts (preprocess.py, evaluate.py, cohort.txt), unchanged
submission/                      course test_input.csv and sample_submission.csv
src/features.py                  activity log, labels, temporal / relational / global features
src/dataset.py                   daily snapshots -> train.parquet / test.parquet
src/models.py                    LightGBM, thresholds, scoring (uses scripts/evaluate.py)
src/experiments.py               time-split baselines and LightGBM ablations
src/analysis.py                  threshold-free analysis (average precision, warm / cold users)
src/gnn.py, src/train_gnn.py     temporal heterogeneous GNN and its experiments
src/ensemble_eval.py, src/decision_eval.py, src/gnn_rescore.py, src/push_diag.py
                                 ensembles, decision rules, push diagnosis on the validation weeks
src/predict.py                   final model and output file
src/schema_graph.py, src/framework_figure.py, src/explain_figures.py, src/figures.py, src/summarize.py
                                 figures and result tables for the slides
slides/                          slide builder (python-pptx) and the slides
results/, figures/               experiment results and figures used in the slides
```
