# BT4222 Group 6 — Personalised News Recommendation

A two-stage recommender on MIND-large: retrieve Top-M eligible articles, then re-rank into Top-K.

## Project structure

| Folder | Purpose |
|---|---|
| `0_Data_Acquisition/` | Original MIND download + splitting (recovery path) |
| `1_Data_Preparation/` | Cleaning, validation and shared processed datasets |
| `2_EDA/` | Impression, user and article analysis |
| `3_Representations/` | Text/entity embeddings and user profiles |
| `4_Feature_Engineering/` | Historical, match, freshness and slate-context features |
| `5_Basic_Models/` | Baselines — planned |
| `6_Candidate_Generation/` | Top-M retrieval — planned |
| `7_Re_Ranking/` | Ranking models — planned |
| `8_Advanced_Features/`, `9_Analysis/` | Clustering, graph embeddings and ablations — planned |
| `10_End_to_End/` | Pipeline evaluation — planned |

Parts 3/4 show their core calculations in notebook cells; shared checks, cache handling and storage live in `src/newsrec/`.

## Setup

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For interactive notebooks, select `.venv/bin/python` as the kernel.
Start at step 1: it reuses local `train/`, `validation/`, and `test/` inputs or downloads missing
required files from the team's Drive. If Drive is unavailable, it directs you to step 0 to rebuild
from original MIND releases. Data stays local and is gitignored.

## Run

Run commands from the repository root with `.venv` activated:

```bash
python run_all.py                                  # normal: stages 1–4, with Drive fallback
python run_all.py --from-stage 0 --to-stage 0       # recovery: MIND download + split
python run_all.py --from-stage 2 --to-stage 2       # EDA only
python run_all.py --from-stage 3 --to-stage 4       # representations + features
python run_all.py --features-smoke 1000            # smaller feature smoke build
python run_all.py --notebook 3_Representations/3_01_news_embeddings.ipynb
```

Recovery: [download_split_process.ipynb](0_Data_Acquisition/download_split_process.ipynb).
Stages 2–4 need their upstream outputs; step 1 supplies missing source inputs. Existing splits are reused; use
`--rebuild-splits` to regenerate them. In stages 3/4, `REBUILD=False` loads verified saved outputs;
set `REBUILD=True` in the notebook after changing its calculations/settings.
`--overwrite-features` also rebuilds the selected feature run.
Outputs update directly in the source notebooks after each completed cell; no timestamped copies are created.
Save notebook edits before running, and reload the notebook in VS Code afterward if prompted.
Text output streams to the terminal, with elapsed-time updates every 30 seconds during long cells.

Saved Parquet/NumPy datasets survive VS Code restarts; rerun only when inputs or calculations change.
Builders apply to preparation/EDA only. Parts 3/4 import shared code automatically; no separate Python run is needed.

Details: [representations](3_Representations/README.md) · [feature engineering](4_Feature_Engineering/README.md).
Optional developer checks: `python -m pytest -q`. Full-data execution is pending.
