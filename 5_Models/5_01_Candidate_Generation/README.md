# 5.01 Candidate generation

Candidate generation retrieves a small set of articles from the full, time-eligible catalogue. A later re-ranker orders that set. This stage owns the shared candidate format, the reusable recall evaluator and the popularity baseline.

| File | What it does |
|---|---|
| [cg_evaluation.py](cg_evaluation.py) | Shared Recall@M evaluator for every candidate-generation model. Loads the split once, validates candidates, and returns pooled and per-impression recall. |
| [baseline_models/CG_popularity_baseline.ipynb](baseline_models/CG_popularity_baseline.ipynb) | Fits non-personalised popularity, retrieves time-eligible Top-500 candidates, writes the standard format, and evaluates them. |
| [neural_models/CG_content_two_tower_mlp.ipynb](neural_models/CG_content_two_tower_mlp.ipynb) | Trains a content two-tower MLP on causal request histories, scores the time-eligible catalogue, writes Top-500 candidates and evaluates them with the shared evaluator. |

Generated files are written under `artifacts/` and are gitignored. Evaluation tables are written under the project-level `results/` folder.

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

1. Fits article popularity from `clicks_train` for validation, or `clicks_train + clicks_validation` for test.
2. Orders articles by prior clicks, breaking ties by ascending `news_code`.
3. Filters the catalogue with `available_from <= timestamp`.
4. Expands the timestamp-level list to the standard impression-level format.
5. Writes Top-500 candidates and evaluates M = 50, 100, 200 and 500.

Current validation results:

| M | `recall_pooled` | `recall_impression_mean` |
|---:|---:|---:|
| 50 | 0.0406 | 0.0459 |
| 100 | 0.0925 | 0.1037 |
| 200 | 0.1367 | 0.1468 |
| 500 | 0.2109 | 0.2157 |

The model cannot retrieve the roughly 14.4% of validation clicks whose articles were never clicked in train. Content-based candidate generation is the next baseline.

## Adding a new model

A new candidate-generation notebook only needs to:

1. Generate a standard table with the same `CANDIDATE_CAP` as every other model.
2. Evaluate it with the shared `CandidateEvaluator`.
3. Save its own model and candidate artifacts under `artifacts/<model>/`.

Do not add model-specific branches to `cg_evaluation.py`.

## Verification

```bash
python -m unittest discover -s tests -p "test_cg_evaluation.py" -v
```

The evaluation table for popularity is saved as `results/candidate_generation_popularity_validation.csv`.
