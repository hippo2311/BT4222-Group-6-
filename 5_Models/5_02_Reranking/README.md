# 5.02 Reranking

Reranking orders the articles in an impression so the ones a user will click come first.
Every notebook here trains (or defines) one model family, scores **every candidate in every
complete logged slate**, and evaluates those scores with the shared `rr_evaluation.py`.

The notebooks read prepared data from `DATA_ROOT` (default
`/Users/numkoos/Downloads/v3-mindsmall`) and write checkpoints, score files and metric
tables there. Change `ENVIRONMENT`, `PROJECT_ROOT` and `DATA_ROOT` in section 1.2 of a
notebook to run it elsewhere (for example on Colab).

## Notebooks

| Notebook | Model | Experiments |
|---|---|---|
| [1_Baseline/RR_popularity_baseline.ipynb](1_Baseline/RR_popularity_baseline.ipynb) | Causal smoothed article CTR | Fixed rule; validation and test |
| [1_Baseline/RR_content_similarity_baseline.ipynb](1_Baseline/RR_content_similarity_baseline.ipynb) | Cosine to the user's text / entity history profile | Text, entity and 3 fusion weights |
| [2_MLP/RR_MLP.ipynb](2_MLP/RR_MLP.ipynb) | Tabular MLP | BCE vs BPR (3 negative samplers) vs BCE–BPR hybrid |
| [3_Attention/RR_candidate_attention.ipynb](3_Attention/RR_candidate_attention.ipynb) | Text-only candidate→history attention | 1 vs 4 attention heads |
| [3_Attention/RR_NRMS.ipynb](3_Attention/RR_NRMS.ipynb) | Text-only NRMS | Heads × dropout grid, then loss ablation on the winner |
| [3_Attention/RR_hybrid_attention_mlp.ipynb](3_Attention/RR_hybrid_attention_mlp.ipynb) | Candidate attention + tabular MLP | 27-trial dropout / weight decay / width grid |
| [3_Attention/RR_hybrid_nrms_mlp.ipynb](3_Attention/RR_hybrid_nrms_mlp.ipynb) | NRMS user encoder + tabular MLP | Single configuration |
| [3_Attention/RR_hybrid_sequence_mlp.ipynb](3_Attention/RR_hybrid_sequence_mlp.ipynb) | LSTM/GRU user encoder + tabular MLP | `RNN_TYPE` × `OBJECTIVE` (BCE+BPR or LambdaNDCG) |
| [RR_evaluate_generated_candidates.ipynb](RR_evaluate_generated_candidates.ipynb) | Sequence reranker on generated candidates | Generator vs reranker vs blend weights |

Run the popularity notebook first: later notebooks use its validation result as the
baseline for observed-CTR lift.

Every model notebook follows the same layout:

1. **Setup**: packages, paths, ranking-feature previews
2. **Experiment configuration**: all tunable constants in one cell
3. **Training data**: sampling, preprocessing, causal histories
4. **Model**: one markdown explanation and one class per component
5. **Training**
6. **Validation**: select the trial with the best validation nDCG@10
7. **Test**: evaluate only the validation-selected trial
8. **Recommendations**: rank one user's latest logged slate with the saved scores

Model, data-preparation and training code lives in the notebooks. `rr_evaluation.py` is
the only shared module.

## Shared evaluation module

| Function | Purpose |
|---|---|
| `score_and_evaluate(predict, split=..., ...)` | Streams a split's feature partitions through a notebook's `predict(frame)`, saves `<split>_scores.parquet`, evaluates it and optionally writes `reranking_<model>_<split>.csv` |
| `RerankingEvaluator` | Validates a complete scored slate and reports MRR, nDCG@K, Recall@K, click retention, engaged-impression rate and observed CTR |
| `compare_results(results, baseline_model=...)` | Concatenates result tables or CSVs, adds observed-CTR lift and sorts |
| `rank_logged_slate(user_code, score_file, ...)` | Shows one user's latest logged slate ranked by a score file |
| `evaluate_retrieved_candidates(candidates, ...)` | Known-click metrics and candidate recall for generated candidate lists |

### Scored-slate contract

A score file has one row per labelled `(impression_uid, news_code)` pair in the split,
with a finite `score` column (larger is better). The evaluator rejects missing,
duplicate or extra pairs, so models are always compared on complete logged slates.
Tied scores are broken by ascending `news_code`.

`recall_at_K` is an impression mean; `click_retention_at_K` pools every known click.
Observed CTR and its lift are offline logged-slate proxies, not online CTR estimates.

### Adding a metric

Metrics are decorator-registered per-slate functions. Return `mean(...)` for an
impression average, `ratio(...)` for a pooled numerator and denominator, or
`total(...)` for a count:

```python
from rr_evaluation import mean, register_metric

@register_metric("top_ranked_click")
def top_ranked_click(context):
    return {"top_ranked_click": mean(context.ranked_labels[0])}
```

Every registered metric runs by default; pass `metric_names=` to
`RerankingEvaluator.evaluate` to select a subset.

## Logged slates versus generated candidates

The model notebooks measure *conditional ranking quality on logged slates*, the only
setting with complete labels. `RR_evaluate_generated_candidates.ipynb` reranks the
candidate generator's output instead; there, unshown articles have unknown relevance and
count as "not a known click", so always report candidate recall alongside its ranking
metrics. Retrieval itself is evaluated at Recall@M by
`5_01_Candidate_Generation/cg_evaluation.py`.

## Split discipline

Tune and select on validation only. Each notebook evaluates its validation-selected
trial on test once, for reporting; do not change a grid or setting in response to test.
