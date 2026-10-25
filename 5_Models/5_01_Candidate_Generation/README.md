# 5.01 Candidate generation

Candidate generation retrieves a small set of articles from the full, time-eligible catalogue. A later re-ranker orders that set. This stage owns the shared candidate format, shared data access and matrix construction, the reusable recall evaluator, and the baseline and neural candidate generators.

## Contents

| File | What it does |
|---|---|
| [cf_models.py](cf_models.py) | Shared project context and static binary user-item matrix. Exposes `load_context`, `build_interaction_matrix` and `resolve_device`; model-specific logic stays in the notebooks. |
| [cg_evaluation.py](cg_evaluation.py) | Shared Recall@M evaluator for every candidate-generation model. Loads the split once, validates candidates, and returns pooled and per-impression recall. |
| [baseline_models/CG_popularity_baseline.ipynb](baseline_models/CG_popularity_baseline.ipynb) | Fits non-personalised popularity from prior exposures, retrieves time-eligible Top-500 candidates, writes the standard format, and evaluates them. |
| [baseline_models/CG_collaborative_filtering_baseline.ipynb](baseline_models/CG_collaborative_filtering_baseline.ipynb) | Builds item-item cosine Top-K neighbours from the shared user-item matrix, scores request histories, writes Top-500 candidates and evaluates them. |
| [baseline_models/CG_content_baseline.ipynb](baseline_models/CG_content_baseline.ipynb) | Text, entity and text/entity cosine-similarity baselines over causal request profiles. |
| [neural_models/CG_content_two_tower_mlp.ipynb](neural_models/CG_content_two_tower_mlp.ipynb) | Content two-tower MLP with frozen fit-split popularity features, trained with sampled exposed negatives. |
| [neural_models/CG_gmf.ipynb](neural_models/CG_gmf.ipynb) | Generalised matrix factorisation over the shared matrix, trained with BPR and one sampled negative per positive. |
| [neural_models/CG_ncf.ipynb](neural_models/CG_ncf.ipynb) | Neural collaborative filtering with two embeddings, an MLP scoring head and BPR training. |
| [neural_models/CG_ncf_with_popularity.ipynb](neural_models/CG_ncf_with_popularity.ipynb) | NCF with frozen fit-split popularity features appended to the MLP input. |

Generated files are written under `artifacts/<model>/` and are gitignored. Evaluation tables are written under the project-level `results/` folder.

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
- No padding row, null key or unknown article is allowed.

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
    m_values=(50, 100, 200, 500),
)
```

`recommendations_path` may point to a Parquet file, or a Polars DataFrame can be passed directly. The evaluator raises on missing impressions, invalid cutoffs, duplicate candidates, unknown articles or ineligible articles.

### Metrics

| Column | Definition |
|---|---|
| `recall_pooled` | Total retrieved clicked articles / total clicked articles. Impressions with more clicks contribute more. |
| `recall_impression_mean` | Mean of per-impression recall. Every impression has equal weight. |
| `hit_rate` | Share of impressions with at least one retrieved click. |
| `impressions` | Number of evaluated impressions. |
| `relevant_clicks` | Total known clicked articles in the recall denominator. |
| `retrieved_relevant` | Clicked articles retrieved inside the candidate set. |
| `mean_candidates` | Mean candidate-set size at that M. |
| `mean_eligible_pool` | Mean number of articles eligible at request time. |
| `eligible_click_coverage` | Share of known clicks whose article was eligible at request time. |

The denominator uses all known clicks. Ineligible clicks are reported through `eligible_click_coverage` rather than removed from the metric.

## Popularity baseline

The popularity notebook:

1. Fits article popularity from prior exposures: `exposures_train` for validation, or `exposures_train + exposures_validation` for test.
2. Orders articles by prior exposures, breaking ties by ascending `news_code`.
3. Filters the catalogue with `available_from <= timestamp`.
4. Expands the timestamp-level list to the standard impression-level format.
5. Writes Top-500 candidates and evaluates M = 50, 100, 200 and 500.

The model is non-personalised, so it is strongest when M is large enough to cover the head of the exposure distribution and weaker at small M. Its result file is `results/candidate_generation_popularity_validation.csv`.

## Validation comparison

`recall_pooled` from `results/candidate_generation_comparison_validation.csv`:

| Model | M=50 | M=100 | M=200 | M=500 |
|---|---:|---:|---:|---:|
| `cf_item_item` | 0.0070 | 0.0208 | 0.0437 | 0.0867 |
| `popularity` | 0.0028 | 0.0136 | 0.0463 | 0.1789 |
| `content_text` | 0.0018 | 0.0038 | 0.0076 | 0.0176 |
| `content_entity` | 0.0016 | 0.0031 | 0.0056 | 0.0116 |
| `content_text_entity` | 0.0033 | 0.0060 | 0.0103 | 0.0208 |
| `content_popularity_two_tower_mlp` | 0.0146 | 0.0337 | 0.0613 | 0.0946 |
| `gmf` | 0.0089 | 0.0164 | 0.0295 | 0.0794 |
| `ncf` | 0.0084 | 0.0154 | 0.0320 | 0.0777 |
| `ncf_popularity` | 0.0129 | 0.0228 | 0.0487 | 0.1063 |

## Adding a new model

A new candidate-generation notebook should:

1. Reuse `load_context`, `build_interaction_matrix` and `resolve_device` from `cf_models.py` when it needs the shared matrix or device selection.
2. Keep model-specific scoring, neighbour construction and candidate selection in the notebook.
3. Generate a standard table with the same `CANDIDATE_CAP` as every other model.
4. Evaluate it with the shared `CandidateEvaluator`.
5. Save its own model, manifest and candidate artifacts under `artifacts/<model>/`.

Do not add model-specific branches to `cg_evaluation.py` or `cf_models.py`.

## Verification

```bash
python -m unittest tests.test_cf_baseline -v
python -m unittest discover -s tests -p "test_cg_evaluation.py" -v
```
