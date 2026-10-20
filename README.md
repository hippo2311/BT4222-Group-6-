# BT4222 Group 6 — Personalised News Recommendation

A MIND-large recommendation project: prepare article/user representations, retrieve eligible articles, then re-rank them. Stages 0–4 provide the datasets for the model-development stages.

## Setup

Use Python 3.12 and run commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In VS Code, select `.venv/bin/python` as the notebook kernel.

## Workflow

1. **Start at stage 1.** It uses local `train/`, `validation/` and `test/` files, or downloads missing inputs from the team's Drive. If Drive is unavailable, stage 0 downloads/extracts original MIND releases and creates the splits.
2. **Prepare and inspect the data:** stage 1 writes cleaned datasets; stage 2 runs EDA.
3. **Build representations:** run `3_01`, `3_02` and `3_03` in order.
4. **Build ranking features:** stage 4 uses those representations and chronological event history.
5. **Develop models:** consume the saved datasets for baselines, retrieval and re-ranking; then run ablations and end-to-end evaluation.

| Stage | Notebook documentation |
|---|---|
| 0 — Download and split | [Data acquisition](0_Data_Acquisition/README.md) |
| 1 — Clean and validate | [Data preparation](1_Data_Preparation/README.md) |
| 2 — Explore the data | [EDA](2_EDA/README.md) |
| 3 — Encode articles and users | [Representations](3_Representations/README.md) |
| 4 — Build ranking inputs | [Feature engineering](4_Feature_Engineering/README.md) |

## Run from the terminal

```bash
python run_all.py                                  # stages 1–4; reuse compatible saved artifacts
python run_all.py --from-stage 1 --to-stage 4 --force  # rebuild stages 1–4 on the full dataset
python run_all.py --from-stage 3 --to-stage 4 --force  # rebuild representations and full features
python run_all.py --from-stage 4 --to-stage 4 --force  # rebuild full features using existing representations
python run_all.py --from-stage 2 --to-stage 2       # EDA only
python run_all.py --from-stage 3 --to-stage 4 --features-smoke 1000 --force
python run_all.py --notebook 3_Representations/3_01_news_embeddings.ipynb
```

Stage ranges are inclusive and need their upstream files. Add `--dry-run` to preview the selected notebooks. Stage 0 recovery is `python run_all.py --from-stage 0 --to-stage 0`; existing complete splits are preserved unless `--rebuild-splits` is supplied.

Stages 1/2 recalculate when executed. Stages 3/4 use `REBUILD=False` to validate and load compatible saved artifacts, building missing outputs. Use `--force` after changing calculations or settings; rebuild stages 3–4 together after changing representations. Formula-only notebook edits need an explicit rebuild.

The terminal runner defaults to **full features**. Directly opening the feature notebook defaults to a 1,000-timestamp smoke run: set `FULL_RUN=True` for full interactive execution and `REBUILD=True` to recalculate saved results. Terminal flags override these preferences for that execution.

Outputs are saved directly into source notebooks after completed cells. Save your edits before starting; avoid editing notebooks during execution. Progress and approximate feature-replay ETA appear in the terminal.

## Saved data and model handoff

| Location | Content |
|---|---|
| `train/`, `validation/`, `test/` | Local split inputs and entity vectors |
| `1_Data_Preparation/processed/` | Cleaned catalogue, users, impressions and candidate labels |
| `3_Representations/artifacts/` | Article vectors, title tokens, eligibility and supplied user profiles |
| `4_Feature_Engineering/artifacts/prequential-all/` | Full features, labels, causal profile references and model schema |
| `results/` | Coverage, cost, readiness and model diagnostic reports |

Saved datasets survive VS Code restarts and are gitignored. Smoke features use a separate folder. Parts 3/4 keep important calculations in notebook cells; `src/newsrec/` supplies shared checks, replay, caching, storage and the `newsrec.ranking` import interface (see [src/newsrec/README.md](src/newsrec/README.md)). No separate helper execution is required.

Model owners should select `model_columns`, encode `categorical_columns`, join labels on both `impression_uid` and `news_code`, fit preprocessing on training data, and evaluate complete held-out slates under the stated feedback protocol. Use Arrow/Polars batches for large artifacts.

```bash
PYTHONPATH=src python -m newsrec.sanity --max-train-rows 100000
python -m pytest -q
```

The first command compares a diagnostic logistic model against popularity on full validation slates; the second runs automated checks. [Notebook builders](notebook_builders/README.md) are optional generators for stages 1/2 and overwrite the notebooks they generate.

Methods explored, design decisions and the latest validated state are recorded in [journal.md](journal.md).
