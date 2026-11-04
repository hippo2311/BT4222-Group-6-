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
| [1_Baseline/RR_popularity_baseline.ipynb](1_Baseline/RR_popularity_baseline.ipynb) | Causal smoothed article CTR | Fixed rule (no tuning); validation and test |
| [1_Baseline/RR_content_similarity_baseline.ipynb](1_Baseline/RR_content_similarity_baseline.ipynb) | Cosine to the user's causal text / entity history profile | 5 rules: text, entity, fusion with text weight 0.25 / 0.50 / 0.75 |
| [2_MLP/RR_MLP.ipynb](2_MLP/RR_MLP.ipynb) | Tabular MLP (256 → 64) | BCE vs BPR with popularity-weighted negatives (α = 0, 0.5, 1.0), then BCE–BPR hybrids (BPR weight 0.25 / 0.50 / 0.75) |
| [3_Attention/RR_1_candidate_attention.ipynb](3_Attention/RR_1_candidate_attention.ipynb) | Candidate→history multi-head attention (+ optional tabular) | 12-trial grid: heads {1, 4} × dropout {0.1, 0.2, 0.3} × scorer {128-64, 256-128-64} |
| [3_Attention/RR_2_nrms.ipynb](3_Attention/RR_2_nrms.ipynb) | NRMS self-attention user encoder (+ optional tabular) | 9-trial grid: heads {1, 2, 4} × attention dropout {0, 0.1, 0.2}; then the MLP-style loss / negative-sampling ablation on the winner |
| [3_Attention/RR_3_rnn.ipynb](3_Attention/RR_3_rnn.ipynb) | LSTM/GRU over the chronological click history (+ optional tabular) | One run per `RNN_TYPE` (`lstm`/`gru`) × `OBJECTIVE` (`bce_bpr` or `lambda_ndcg`@5), trained on all retained training candidates |
| [3_Attention/RR_4_branch_interaction.ipynb](3_Attention/RR_4_branch_interaction.ipynb) | One switchable model: news (text, metadata, title) × user (self-attention, position, candidate pooling, short-term GRU) × tabular | Stage 1: 13 architectures with listwise softmax; stage 2: 8 losses on the chosen architecture; stage 3: full training set with early stopping |

The first three `3_Attention` notebooks differ mainly in the **user encoder**. Each has a
`USE_TABULAR` switch: `True` feeds the causal tabular features to the scorer alongside the
candidate and user vectors, `False` drops that branch (text + click history only). Run
each notebook once per setting; its final section compares the two, which shows how much
the tabular features add on top of the learned user representation.

### Branch interaction study (`RR_4`)

`RR_4_branch_interaction.ipynb` combines those encoders in one network with every branch
switchable, and measures how they interact:

- **2×2, tabular × user encoder**: `text`, `tabular+text`, `text+user`, `tabular+text+user`
  (plus a `tabular_only` reference)
- **Ladder**: `+metadata` → `+position` → `+candidate_pooling` → `full`
- **Leave-one-out**: `full-tabular`, `full-metadata`, `full-position`, `full-candidate_pooling`
- The trainable title encoder is off by default (`RUN_TITLE_ENCODER = True` adds
  `full+title_encoder`)

Stage 1 trains every architecture on the same 10k training impressions and ranks them on
a fixed 20k-impression validation subset (`score_and_evaluate(..., impressions=...)`);
the top 3 are rescored on full validation. `FINAL_ARCHITECTURE` is pinned to `+metadata`,
which ties the best full-validation nDCG@10 (0.4499 vs 0.4500) while keeping the
personalised news and user branches. Stage 2 compares `softmax`, `lambda_ndcg`, `bce`,
`bpr` and four weighted combinations on that architecture. Stage 3 retrains the chosen
architecture and loss on every training impression (up to 20 epochs, patience 3 on a 10%
holdout's nDCG@10), and only that model is scored on test. Every trial is checkpointed,
so a stopped run resumes with `RESUME = True`.

![Final RR_4 architecture](3_Attention/figures/rr4_final_architecture.png)

The figure shows the final model: `+metadata` (category, subcategory and entity
embeddings added to the news encoder) with the NRMS user encoder and the tabular branch,
trained with 0.5 × listwise softmax + 0.5 × LambdaNDCG@10.

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

Model, data-preparation and training code lives in the notebooks. The shared modules are
`rr_evaluation.py` (scoring and metrics) and `experiment_naming.py` (self-describing
artifact names, used by `RR_3_rnn.ipynb`).

## Shared evaluation module

| Function | Purpose |
|---|---|
| `score_and_evaluate(predict, split=..., ...)` | Streams a split's feature partitions through a notebook's `predict(frame)`, saves `<split>_scores.parquet`, evaluates it and optionally writes `reranking_<model>_<split>.csv`. With `impressions=`, only that subset is scored and files use a `<split>_subset` prefix so they are never mistaken for full-split results |
| `RerankingEvaluator(..., impressions=None)` | Validates a complete scored slate (optionally restricted to a fixed impression subset) and reports MRR, nDCG@K, Recall@K, click retention, engaged-impression rate and observed CTR |
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
setting with complete labels.
[`5_03_Pipeline_Evaluation`](../5_03_Pipeline_Evaluation/README.md) reranks the
candidate generator's output instead; there, unshown articles have unknown relevance and
count as "not a known click", so always report candidate recall alongside its ranking
metrics. Retrieval itself is evaluated at Recall@M by
`5_01_Candidate_Generation/cg_evaluation.py`.

## Split discipline

Tune and select on validation only. Each notebook evaluates its validation-selected
trial on test once, for reporting; do not change a grid or setting in response to test.
