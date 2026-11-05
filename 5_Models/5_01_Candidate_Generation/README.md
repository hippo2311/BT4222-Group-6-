# 5.01 Candidate generation

Candidate generation retrieves a small set of articles from the full, time-eligible catalogue. A later re-ranker orders that set. This stage owns the shared candidate format, shared data access and matrix construction, the unified recall and ranking-quality evaluator, and the baseline and neural candidate generators.

## Contents

| File | What it does |
|---|---|
| [cf_models.py](cf_models.py) | Shared project context and static binary user-item matrix. Exposes `load_context`, `build_interaction_matrix` and `resolve_device`; model-specific logic stays in the notebooks. |
| [cg_evaluation.py](cg_evaluation.py) | Unified evaluator for every candidate-generation model. Loads the split once, validates candidate shape and coverage, and returns recall plus long-tail, cold-article, novelty, catalogue-coverage and personalisation metrics for each requested Top-M set. |
| [cg_result_writer.py](cg_result_writer.py) | Shared `CandidateWriter` and `CANDIDATE_SCHEMA` for the standard long-format candidate table. |
| [neural_helpers.py](neural_helpers.py) | Shared chronological tuning, slate NDCG and early-stopping helpers for the neural candidate generators. |
| [1_baseline_models/CG_popularity_baseline.ipynb](1_baseline_models/CG_popularity_baseline.ipynb) | Fits non-personalised popularity from prior clicks, retrieves time-eligible Top-1500 candidates, writes the standard format, and evaluates them. |
| [1_baseline_models/CG_collaborative_filtering_baseline.ipynb](1_baseline_models/CG_collaborative_filtering_baseline.ipynb) | Builds item-item cosine Top-K neighbours from the shared user-item matrix, scores request histories, writes Top-1500 candidates and evaluates them. |
| [1_baseline_models/CG_content_baseline.ipynb](1_baseline_models/CG_content_baseline.ipynb) | Text, entity and text/entity cosine-similarity baselines over causal request profiles. |
| [1_baseline_models/CG_rolling_popularity_baseline.ipynb](1_baseline_models/CG_rolling_popularity_baseline.ipynb) | Rolling prior-hour popularity control. Its score mode is fixed to `prior_hour_clicks`, ranking by previous-hour clicks and writing Top-1500 candidates. |
| [2_neural_models/CG_content_two_tower_mlp.ipynb](2_neural_models/CG_content_two_tower_mlp.ipynb) | Content two-tower MLP with rolling prior-hour popularity features (`content_rolling_popularity_two_tower_mlp`), trained on the full `labels_train` pool with up to four exposed negatives redrawn per positive each epoch for both BPR and BCE. |
| [3_sequential_neural_models/CG_content_two_tower_transformer.ipynb](3_sequential_neural_models/CG_content_two_tower_transformer.ipynb) | Content two-tower with a transformer over previous-session text/entity vectors, plus mean-pooled history and popularity fallbacks; trained on the full `labels_train` pool with exposed negatives redrawn per positive each epoch for BPR. |
| [4_combined_models/CG_content_two_tower_rolling_popularity.ipynb](4_combined_models/CG_content_two_tower_rolling_popularity.ipynb) | Combined 750-item rolling-popularity plus 750-item content two-tower candidate set; the content model draws exposed negatives per epoch from the full `labels_train` pool. |
| [2_neural_models/CG_gmf.ipynb](2_neural_models/CG_gmf.ipynb) | Generalised matrix factorisation over the shared matrix, trained with BPR or BCE using up to four exposed session negatives redrawn per positive each epoch. |
| [2_neural_models/CG_ncf.ipynb](2_neural_models/CG_ncf.ipynb) | Neural collaborative filtering with two embeddings, an MLP scoring head and BPR/BCE training using up to four exposed session negatives redrawn per positive each epoch. |
| [2_neural_models/CG_ncf_features.ipynb](2_neural_models/CG_ncf_features.ipynb) | NCF with frozen fit-split popularity and freshness features appended to the MLP input, trained with BPR or BCE on up to four exposed session negatives redrawn per positive each epoch. |
| [3_sequential_neural_models/CG_ncf_features_transformer.ipynb](3_sequential_neural_models/CG_ncf_features_transformer.ipynb) | Sequential NCF variant with transformer history encoding and popularity/freshness features, trained with BPR or BCE on all stored session negatives per positive. |

Generated files are written under `artifacts/<model>/` and are gitignored. Evaluation tables are written under the project-level `results/` folder.

All model notebooks run the full pipeline over the complete target split.

## Shared data and matrix utilities

`cf_models.py` is limited to code reused across models:

- `load_context` loads the article index, availability timestamps, training popularity and user count.
- `build_interaction_matrix` builds or loads the cached binary `users x article_row` matrix from training interactions.
- `resolve_device` selects CPU or CUDA consistently for the neural notebooks.

Cosine-neighbour construction, item-item selection, ranking models and their evaluation helpers live in the notebook that owns them. Do not add model-specific branches to `cf_models.py`.

## Standard candidate format

Every candidate-generation model must write the same long-format table:

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

`cutoff` is **not** a ranking metric. It only identifies nested Top-M sets so one Top-C artifact can be evaluated at several values of M. Recall ignores the order of candidates.

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
    m_values=(50, 100, 200, 500, 1000, 1500),
)
```

`recommendations_path` may point to a Parquet file, or a Polars DataFrame can be passed directly. The evaluator raises on missing impressions, invalid cutoffs, duplicate candidates and missing required columns (`impression_uid`, `news_code`, `cutoff`). Pass `impression_uids=` to evaluate a bounded subset; candidates must cover exactly that subset.

Candidate tables only need to reach the largest requested `M`. All metrics are computed at every requested `M` using that Top-M set.

### Output

Every result is one long table:

```
model, split, m, metric, value, numerator, denominator
```

Every row uses `m=<requested M>`. Call `wide_table(result)` to pivot this into one row per model/split/M with one column per metric.

| Metric | Definition |
|---|---|
| `recall_pooled` | Total retrieved clicked articles / total clicked articles. Impressions with more clicks contribute more. |
| `recall_impression_mean` | Mean of per-impression recall. Every impression with at least one click has equal weight. |
| `hit_rate` | Share of impressions with at least one retrieved click. |
| `long_tail_recall` | Recall on clicks for articles outside the previous completed hour's Top-200 by clicks (tie-broken by exposures, then `news_code`). |
| `cold_article_recall` | Recall on clicks for articles never shown before the click's hour. |
| `novelty_bits` | Mean `-log2((clicks in the previous hour + 1) / (previous-hour clicks + catalogue size))` over the recommended articles. Higher means less mainstream. |
| `catalogue_coverage` | Unique articles retrieved in the Top-M set divided by the eligible catalogue at the split's latest evaluated timestamp. |
| `personalisation` | `1 - mean Jaccard overlap` between the Top-M lists of different users. Each user is represented by their first impression; up to 10,000 sampled user pairs are used. Higher means the lists differ more between users. |

Long-tail recall, cold-article recall and novelty read exposure history from the Stage 1 candidate logs of the evaluated split and every earlier split (`train` -> `validation` -> `test`). Only exposures from hours before the click or request are used.

Every model calls `evaluate_to_csv`, which writes `results/candidate_generation_<model>_<split>.csv` and updates `results/candidate_generation_comparison_<split>.csv`.

Run the tests with:

```bash
python -m unittest tests.test_cg_evaluation -v
```

## Popularity baseline

The popularity notebook:

1. Fits article popularity from prior clicks: `clicks_train` for validation, or `clicks_train + clicks_validation` for test.
2. Orders articles by prior clicks, breaking ties by ascending `news_code`.
3. Filters the catalogue with `available_from <= timestamp`.
4. Expands the timestamp-level list to the standard impression-level format.
5. Writes Top-1500 candidates and evaluates M = 50, 100, 200, 500, 1000 and 1500.

The model is non-personalised, so it is strongest when M is large enough to cover the head of the click distribution and weaker at small M. Its result file is `results/candidate_generation_popularity_validation.csv`.

## Rolling popularity baseline

`CG_rolling_popularity_baseline.ipynb` builds a prequential popularity order from
strictly earlier completed hours. Its score mode is fixed to `prior_hour_clicks` and
is not configurable: it ranks by previous-hour clicks, tie-broken by previous-hour
exposures and then `news_code`.
It writes `artifacts/popularity_freshness_prior_hour_clicks/recommendations_<split>_top1500.parquet`.

## Validation comparison

`recall_pooled` from an earlier 500-candidate run of
`results/candidate_generation_comparison_validation.csv`:

| Model | M=50 | M=100 | M=200 | M=500 |
|---|---:|---:|---:|---:|
| `cf_item_item` | 0.0070 | 0.0208 | 0.0437 | 0.0867 |
| `popularity` | 0.0406 | 0.0925 | 0.1367 | 0.2109 |
| `content_text` | 0.0018 | 0.0038 | 0.0076 | 0.0176 |
| `content_entity` | 0.0016 | 0.0031 | 0.0056 | 0.0116 |
| `content_text_entity` | 0.0033 | 0.0060 | 0.0103 | 0.0208 |
| `popularity_freshness_prior_hour_clicks` | 0.5780 | 0.7002 | 0.8068 | 0.9230 |
| `content_popularity_two_tower_mlp` | 0.0146 | 0.0337 | 0.0613 | 0.0946 |
| `gmf` | 0.0089 | 0.0164 | 0.0295 | 0.0794 |
| `ncf` | 0.0084 | 0.0154 | 0.0320 | 0.0777 |
| `ncf_popularity` | 0.0129 | 0.0228 | 0.0487 | 0.1063 |

## Adding a new model

A new candidate-generation notebook should:

1. Reuse `load_context`, `build_interaction_matrix` and `resolve_device` from `cf_models.py` when it needs the shared matrix or device selection.
2. Keep model-specific scoring, neighbour construction and candidate selection in the notebook.
3. Generate a standard table with the same `CANDIDATE_CAP` as every other model, using `CandidateWriter` from `cg_result_writer.py`.
4. Evaluate it with the shared `CandidateEvaluator`.
5. Save its own model and candidate artifacts under `artifacts/<model>/`.

Do not add model-specific branches to `cg_evaluation.py` or `cf_models.py`.

## Verification

```bash
python -m unittest tests.test_cf_baseline -v
python -m unittest discover -s tests -p "test_cg_evaluation.py" -v
```
