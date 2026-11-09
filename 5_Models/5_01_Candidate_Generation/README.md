# 5.01 Candidate generation

Candidate generation retrieves a small set of articles from the time-eligible catalogue so a
later re-ranker can order them. This stage owns the shared candidate format, shared data and
matrix utilities, the unified recall and ranking-quality evaluator, and the baseline, neural,
sequential and combined candidate generators.

Retrieval is organised around two channels: a high-precision **popularity** channel (recent
prequential clicks) and a high-recall **niche/content** channel. The combined notebooks union
the two so the re-ranker sees popular and long-tail articles.

## Reusing saved artifacts

Most experiment notebooks start with a **Configure** cell. Notebooks with reusable artifacts (the
collaborative-filtering baseline and the neural/niche notebooks) expose `REUSE_ARTIFACTS`; set it
to `True` to load the saved checkpoint or cached intermediate state and skip training/rebuilding,
or `False` to rebuild from scratch. Candidate lists are always regenerated for `TARGET_SPLIT`, so a
new validation or test split is served normally. The content and rolling-popularity baselines have
no artifacts to reuse, so they expose only `ENVIRONMENT` and `TARGET_SPLIT` and always generate
their lists from scratch.

## Contents

### Shared modules

| File | What it does |
|---|---|
| [cf_models.py](cf_models.py) | Project context and static binary user-item matrix. Exposes `load_context`, `matrix_from_pairs`, `build_interaction_matrix` and `resolve_device`. |
| [cg_evaluation.py](cg_evaluation.py) | Unified evaluator for every candidate-generation model. Loads the split once, validates candidate shape and coverage, and returns recall plus long-tail, cold-article, novelty, catalogue-coverage and personalisation metrics for each requested Top-M set. |
| [cg_result_writer.py](cg_result_writer.py) | Shared `CandidateWriter` and `CANDIDATE_SCHEMA` for the standard long-format candidate table. |
| [neural_helpers.py](neural_helpers.py) | Shared per-epoch negative sampling, chronological tuning, slate NDCG and early-stopping helpers. |

### Baseline models — `1_baseline_models/`

| Notebook | Model name(s) | What it does |
|---|---|---|
| [CG_collaborative_filtering_baseline.ipynb](1_baseline_models/CG_collaborative_filtering_baseline.ipynb) | `cf_item_item` | Builds the binary train user-item matrix, derives item-item cosine Top-100 neighbours, scores request histories causally and writes Top-500 candidates. |
| [CG_content_baseline.ipynb](1_baseline_models/CG_content_baseline.ipynb) | `content_text`, `content_entity`, `content_text_entity` | Text, entity and text/entity cosine-similarity baselines over causal request profiles. History is excluded and empty profiles fall back to popularity. Writes Top-500 per model. |
| [CG_rolling_popularity_baseline.ipynb](1_baseline_models/CG_rolling_popularity_baseline.ipynb) | `popularity_freshness_prior_hour_clicks` | Non-personalised prequential control. Ranks by previous-hour clicks, tie-broken by previous-hour exposures and `news_code`. Writes Top-500 and sweeps M = 50–500. **Run this before the niche notebooks.** |

### Neural models — `2_neural_models/`

| Notebook | Model name(s) | What it does |
|---|---|---|
| [CG_content_two_tower_mlp.ipynb](2_neural_models/CG_content_two_tower_mlp.ipynb) | `content_two_tower_mlp_bpr` | Content-only two-tower MLP. The user tower consumes a pooled 166-d history profile; the item tower consumes intrinsic text/entity content and missing-vector flags. BPR with up to four exposed negatives redrawn per positive each epoch. Writes Top-500. |
| [CG_content_two_tower_popularity_mlp.ipynb](2_neural_models/CG_content_two_tower_popularity_mlp.ipynb) | `content_two_tower_popularity_mlp_bpr` | Popularity-aware two-tower MLP. The item tower consumes the same 166-d content input plus `log1p` prior-hour exposures and clicks (168-d). Serving adds frozen exposure/click weights tuned on the chronological tuning slates; the additive term never enters the training loss. Writes Top-500. |
| [CG_gmf.ipynb](2_neural_models/CG_gmf.ipynb) | `gmf_bpr` | Generalised matrix factorisation: 256-d user and item embeddings with a raw dot-product score, trained with BPR and per-epoch exposed negatives. Writes Top-500. |
| [CG_ncf.ipynb](2_neural_models/CG_ncf.ipynb) | `ncf_bpr` | Neural collaborative filtering: two 128-d embeddings concatenated through an MLP (`256 → 128 → 64 → 32 → 1`), trained with BPR and per-epoch exposed negatives. Writes Top-500. |
| [CG_ncf_features.ipynb](2_neural_models/CG_ncf_features.ipynb) | `ncf_features_early_bpr`, `ncf_features_late_bpr` | Feature-aware NCF: early fusion joins separate user/item towers, late fusion merges ID embeddings before appending features. Uses pooled text/entity history and article category/text features. Writes Top-500 per variant. |

### Sequential neural models — `3_sequential_neural_models/`

| Notebook | Model name(s) | What it does |
|---|---|---|
| [CG_niche_content_two_tower_sequential.ipynb](3_sequential_neural_models/CG_niche_content_two_tower_sequential.ipynb) | `niche_two_tower` | Niche two-tower with attention pooling over reading history, article text/entity/category/subcategory features and a learned freshness adjustment. Trains on clicks with shown-but-not-clicked negatives redrawn each epoch, excludes the popular Top-200 and reading history, saves Top-750 niche lists and sweeps M = 50–750. |

### Combined models — `4_combined_models/`

| Notebook | Model name(s) | What it does |
|---|---|---|
| [CG_baseline_content_popularity_combined.ipynb](4_combined_models/CG_baseline_content_popularity_combined.ipynb) | `baseline_content_popularity_combined` | Self-contained union of 200 rolling-popularity candidates and 250 `content_text_entity` candidates (450 total). |
| [CG_niche_content_two_tower_sequential_combined.ipynb](4_combined_models/CG_niche_content_two_tower_sequential_combined.ipynb) | `popular_niche_200_250` | Trains the niche two-tower and unions 200 rolling-popularity candidates with 250 niche candidates (450 total). Writes train, validation and test lists. |

### Analysis

| Notebook | Purpose |
|---|---|
| [results.ipynb](results.ipynb) | Reads the result CSVs, plots the recall and diversity curves, and records the candidate-generation findings and conclusions. |

## Shared data and matrix utilities

`cf_models.py` is limited to code reused across models:

- `load_context` loads the article index, availability timestamps, training popularity and user count.
- `matrix_from_pairs` builds a binary CSR matrix from `(user_code, article_row)` pairs.
- `build_interaction_matrix` builds or loads the cached binary `users × article_row` matrix from training interactions.
- `resolve_device` selects CPU or CUDA consistently for the neural notebooks.

Cosine-neighbour construction, item-item selection, ranking models and their evaluation helpers
live in the notebook that owns them. Do not add model-specific branches to `cf_models.py`.

## Standard candidate format

Every candidate-generation model writes the same long-format table:

| Column | Type | Meaning |
|---|---|---|
| `impression_uid` | Int64 | Request key from `impressions_<split>.parquet` |
| `cutoff` | Int16 | Candidate membership, `1..C` |
| `news_code` | Int32 | Retrieved article key |

Rules:

- Every impression must have exactly `C` rows.
- Each impression must contain every cutoff from `1` to `C`.
- `news_code` must be unique within an impression.
- Every impression in the target split must be present.
- Every candidate must satisfy `available_from <= impression timestamp`.
- No padding row or null key is allowed.
- `C` must be at least the largest requested `M`; the evaluator slices nested Top-M sets from the same artifact.

`C` is model-specific: 450 for the combined models, 500 for the baseline and neural models,
and 750 for the niche model.

`cutoff` is **not** a ranking metric. It only identifies nested Top-M sets so one Top-C artifact
can be evaluated at several values of M. Recall ignores the order of candidates.

Example:

```text
impression_uid  cutoff  news_code
10996690        1       98178
10996690        2       32154
10996690        3       30899
...
```

A model that emits a single candidate set can still use the standard by assigning `cutoff = 1..C`.

## Reusable evaluation

Import the evaluator from a model notebook:

```python
import sys
from pathlib import Path

ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "run_all.py").is_file()
)
sys.path.insert(0, str(ROOT / "5_Models" / "5_01_Candidate_Generation"))

from cg_evaluation import CandidateEvaluator

evaluator = CandidateEvaluator(root=ROOT, split="validation")
results = evaluator.evaluate(
    recommendations_path,
    model_name="popularity",
    m_values=(50, 100, 200, 500),
)
```

`recommendations_path` may point to a Parquet file, or a Polars DataFrame can be passed directly.
The evaluator raises on missing impressions, invalid cutoffs, duplicate candidates and missing
required columns (`impression_uid`, `news_code`, `cutoff`). Pass `impression_uids=` to evaluate a
bounded subset; candidates must cover exactly that subset.

Candidate tables only need to reach the largest requested `M`. All metrics are computed at every
requested `M` using that Top-M set.

### Output

Every result is one long table:

```
model, split, m, metric, value, numerator, denominator
```

Every row uses `m=<requested M>`. Call `wide_table(result)` to pivot this into one row per
model/split/M with one column per metric.

| Metric | Definition |
|---|---|
| `recall_pooled` | Total retrieved clicked articles / total clicked articles. Impressions with more clicks contribute more. |
| `recall_impression_mean` | Mean of per-impression recall. Every impression with at least one click has equal weight. |
| `hit_rate` | Share of impressions with at least one retrieved click. |
| `long_tail_recall` | Recall on clicks for articles outside the previous completed hour's Top-200 by clicks (tie-broken by exposures, then `news_code`). |
| `cold_article_recall` | Recall on clicks for articles never shown before the click's hour. |
| `novelty_bits` | Mean `-log2((clicks in the previous hour + 1) / (total previous-hour clicks + catalogue size))` over the recommended articles. Higher means less mainstream. |
| `catalogue_coverage` | Unique articles retrieved in the Top-M set divided by the eligible catalogue at the split's latest evaluated timestamp. |
| `personalisation` | `1 - mean Jaccard overlap` between the Top-M lists of different users. Each user is represented by their first impression; up to 10,000 sampled user pairs are used. Higher means the lists differ more between users. |

Long-tail recall, cold-article recall and novelty read exposure history from the Stage 1 candidate
logs of the evaluated split and every earlier split (`train` → `validation` → `test`). Only
exposures from hours before the click or request are used.

Every model calls `evaluate_to_csv`, which writes
`results/candidate_generation_<model>_<split>.csv` and updates
`results/candidate_generation_comparison_<split>.csv`.

## Shared modelling protocol

The neural and content notebooks follow the same causal protocol:

- Request profiles use `full_history_rows`; a later request's profile is never reused for an earlier request.
- Every article in the request's causal history is excluded from its candidates.
- Eligibility is `available_from <= timestamp`.
- Popularity or rolling prior-hour popularity is used only for tie-breaking and empty-history fallback, except `CG_content_two_tower_popularity_mlp.ipynb`, which feeds causal prior-hour exposure/click counts to the item tower and adds frozen serving-only weights.
- Neural models train on the full `labels_train` pool with up to four exposed negatives redrawn per positive each epoch.
- Training uses a chronological 90% tuning slice, early stopping on mean slate NDCG@k, then a refit on all training rows.
- Outputs are ranking scores, not calibrated click probabilities.

The baseline notebooks are deterministic and do not train parameters; the content baselines use
frozen Stage 3 representations.

## Running the notebooks

1. Run `1_baseline_models/CG_rolling_popularity_baseline.ipynb` first. The niche notebooks read its saved Top-500 list.
2. Run the other baseline and neural notebooks independently; they use the Stage 1/3/4 artifacts. The CF baseline additionally uses the cached interaction matrix.
3. Run `3_sequential_neural_models/CG_niche_content_two_tower_sequential.ipynb` to train the niche model and save its Top-750 lists.
4. Run the combined notebooks. `CG_niche_content_two_tower_sequential_combined.ipynb` trains its own niche model and writes train, validation and test candidate lists.
5. Most notebooks use `TARGET_SPLIT = "validation"`; set it to `"test"` to generate test artifacts. The sequential niche notebook evaluates validation, and the combined niche notebook writes train, validation and test lists in one run.

There is no `run_all.py` entry for stage 5; run the notebooks individually.

### Running on Google Colab

Every model notebook starts with an `ENVIRONMENT` switch and `COLAB_DATA_PATH` / `LOCAL_DATA_PATH`
in its configure cell. Set `ENVIRONMENT` to `"colab"` to mount Google Drive, read the project code
from `/content/drive/MyDrive/submission/4.Code`, and read and write candidate-generation artifacts
under `/content/drive/MyDrive/submission/3.Model_Artifacts/CG`. On Colab the prepared data zipped in
`<submission>/2.Feature_Engineering` is unpacked into the data path on the first run (default: the
project folder). Local runs keep `ENVIRONMENT = "local"`, resolve the checkout from `run_all.py`,
and write data and artifacts under the data root (artifacts in
`5_Models/5_01_Candidate_Generation/artifacts`). Set `COLAB_DATA_PATH` / `LOCAL_DATA_PATH`, or the
`NEWSREC_DATA_ROOT` / `NEWSREC_ROOT` environment variables, to put the prepared data somewhere else.

## Artifacts

Generated files are written under `artifacts/<model>/` locally and under
`/content/drive/MyDrive/submission/3.Model_Artifacts/CG/<model>/` on Colab (the `artifacts`
level is dropped there). Artifacts are gitignored. Only artifacts produced by the current
notebooks are listed here.

| Folder | Files |
|---|---|
| `artifacts/cf/` | `user_item_train.npz`, `user_item_train.parquet`, `matrix_manifest.json` (shared matrix cache) |
| `artifacts/cf_item_item/` | `item_item_cosine_topk_100.npz`, `recommendations_validation_top500.parquet` |
| `artifacts/content_text/`, `artifacts/content_entity/`, `artifacts/content_text_entity/` | `recommendations_validation_top500.parquet` |
| `artifacts/popularity_freshness_prior_hour_clicks/` | `recommendations_validation_top500.parquet` |
| `artifacts/two_tower_content_bpr/` | `model.pt`, `profile_index_{train,validation}.parquet`, `profile_uids_{train,validation}.npy`, `user_features_{train,validation}.npy`, `recommendations_validation_top500.parquet`, `training_log.csv` |
| `artifacts/two_tower_content_popularity_bpr/` | `model.pt`, `profile_index_{train,validation}.parquet`, `profile_uids_{train,validation}.npy`, `user_features_{train,validation}.npy`, `recommendations_validation_top500.parquet`, `training_log.csv`, `serving_weight_search.csv` |
| `artifacts/gmf_bpr/`, `artifacts/ncf_bpr/`, `artifacts/ncf_features_early_bpr/`, `artifacts/ncf_features_late_bpr/` | `model.pt`, `recommendations_validation_top500.parquet`, `training_log.csv` |
| `artifacts/niche_two_tower/` | `model.pt`, `recommendations_validation.parquet`, `training_history.csv`, `long_tail_recall_curve.csv`, `long_tail_recall_curve.png`, `run_summary.csv` |
| `artifacts/baseline_content_popularity_combined/` | `recommendations_validation_top450.parquet` |
| `artifacts/popular_niche_combined/` | `recommendations_{train,validation,test}_top450.parquet` |

Metric tables are written under the project-level `results/` folder.

## Results

The validation findings and comparisons are in [results.ipynb](results.ipynb). The underlying
metric tables are written to `results/candidate_generation_<model>_<split>.csv` and consolidated
into `results/candidate_generation_comparison_<split>.csv`.
