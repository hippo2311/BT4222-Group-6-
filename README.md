# BT4222 Group 6 — Personalised News Recommendation

A two-stage recommender on MIND-large: retrieve Top-M eligible articles, then re-rank into Top-K.

## Project structure

| Folder | Purpose |
|---|---|
| `0_Data_Acquisition/` | Temporal train/validation/test split |
| `1_Data_Preparation/` | Cleaning, validation and shared processed datasets |
| `2_EDA/` | Impression, user and article analysis |
| `3_Representations/` | Text/entity embeddings and user profiles |
| `4_Feature_Engineering/` | Historical, match, freshness and slate-context features |
| `5_Basic_Models/` | Baselines — planned |
| `6_Candidate_Generation/` | Top-M retrieval — planned |
| `7_Re_Ranking/` | Ranking models — planned |
| `8_Advanced_Features/`, `9_Analysis/` | Clustering, graph embeddings and ablations — planned |
| `10_End_to_End/` | Pipeline evaluation — planned |

Representation and feature code/settings are written directly in their notebooks.

## Setup

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For interactive notebooks, select `.venv/bin/python` as the kernel.
Place the agreed split Parquet files in `train/`, `validation/`, and `test/`, plus
`train/entity_embedding.vec`. To create splits instead, provide raw `behaviors.tsv`
and `news.tsv` in `train/` and `test/` (official dev release). Data is gitignored.

## Run

Run commands from the repository root with `.venv` activated:

```bash
python run_all.py                                  # stages 0–4; full features
python run_all.py --from-stage 0 --to-stage 2       # acquisition through EDA
python run_all.py --from-stage 2 --to-stage 2       # EDA only
python run_all.py --from-stage 3 --to-stage 4       # representations + features
python run_all.py --features-smoke 1000            # smaller feature smoke build
python run_all.py --notebook 3_Representations/3_01_news_embeddings.ipynb
```

Selected stages require their upstream files. Existing splits are reused; use
`--rebuild-splits` to regenerate them. In stages 3/4, `REBUILD=False` loads verified saved outputs;
set `REBUILD=True` in the notebook after changing its calculations/settings.
`--overwrite-features` also rebuilds the selected feature run.
Executed notebooks with outputs are saved under `runs/<timestamp>/`; source notebooks stay unchanged.

Saved Parquet/NumPy datasets survive VS Code restarts; rerun only when inputs or calculations change.
Builders apply to preparation/EDA only; stages 3/4 are edited directly in their notebooks.

Details: [representations](3_Representations/README.md) · [feature engineering](4_Feature_Engineering/README.md).
Optional developer checks: `python -m pytest -q`. Full-data execution is pending.
