# 5.03 Pipeline evaluation

Evaluates candidate generation and reranking **together**, end to end, the way they run in
deployment. The pipeline notebooks are self-sufficient: they build candidates, rerank and
score the final Top-K in one pass without writing intermediate candidate files.

| Notebook | Purpose |
|---|---|
| [PE_content_reranker_baseline_pipeline.ipynb](PE_content_reranker_baseline_pipeline.ipynb) | Single end-to-end notebook: rebuild the baseline combined generator (200 popularity + 250 content) in memory, rerank with the content-similarity baseline, and evaluate the final Top-K |
| [PE_popularity_reranker_baseline_pipeline.ipynb](PE_popularity_reranker_baseline_pipeline.ipynb) | Same combined generator, reranked by the causal smoothed prior article CTR baseline, and evaluated on the final Top-K |
| [PE_attention_reranker_niche_pipeline.ipynb](PE_attention_reranker_niche_pipeline.ipynb) | Single end-to-end notebook: train the niche candidate generator and the `RR_4` `+metadata` reranker inside, generate candidates in memory, and evaluate the reranked final Top-K |
| [results.ipynb](results.ipynb) | Compare the three pipelines on the test split: load every `pipeline_*.csv` from `results/`, tabulate accuracy and reach metrics, and plot them |

All metrics are known-click metrics: unshown retrieved articles have unknown relevance, and
clicks the retrievers missed count as misses. Candidate recall is the ceiling for every
ranking metric.

## Baseline pipelines

Two single end-to-end notebooks rebuild the same baseline candidate generator in memory and
differ only in the reranker. Neither reads or writes intermediate candidate files.

- **Candidate generation**: rebuilds the combined generator from stage 1, 3 and 4 artifacts:
  200 rolling prior-hour popularity candidates plus 250 content text/entity cosine candidates
  per request (450 total), with request history removed.
- **Rerankers**:
  - `PE_content_reranker_baseline_pipeline.ipynb` applies the content-similarity rule. The run
    uses the fixed `content_fusion_text_0.50` rule; text, entity and the other fusion weights
    are still defined and can be compared by setting `SELECTED_TRIAL = None`.
  - `PE_popularity_reranker_baseline_pipeline.ipynb` applies the parameter-free causal
    popularity rule from `RR_popularity_baseline.ipynb`: each candidate is scored by
    `smoothed_prior_article_ctr`, with ties broken by ascending `news_code`.

The reranked list is evaluated at every cutoff in `K_VALUES = (5, 10, 30)`: every metric in
`cg_evaluation.py` at `M = k`, plus MRR@k and nDCG@k. The saved final Top-K list uses
`K = max(K_VALUES) = 30`.

The content rule is fixed to `content_fusion_text_0.50` by `SELECTED_TRIAL`, so validation and
test runs need no selection step. Set `SELECTED_TRIAL = None` to compare all five content rules
on validation and save the winner to
`results/pipeline_baseline_content_popularity_combined_content_similarity_selection_validation.json`;
a test run then reuses that selection instead of re-selecting. The popularity rule has no
settings to tune.

Only the final Top-K, metrics, config and replay cache are written; candidate lists and scores
stay in memory. Outputs are:

- Content reranker:
  `results/pipeline_baseline_content_popularity_combined_content_similarity_<split>_<impressions>.csv`,
  `5_03_Pipeline_Evaluation/artifacts/baseline_content_popularity_combined_content_similarity_<split>_top<K>_<impressions>.parquet`,
  the config under
  `5_03_Pipeline_Evaluation/artifacts/baseline_content_popularity_combined_content_similarity/`.
- Popularity reranker:
  `results/pipeline_baseline_content_popularity_combined_popularity_reranker_<split>_<impressions>.csv`,
  `5_03_Pipeline_Evaluation/artifacts/baseline_content_popularity_combined_popularity_reranker_<split>_top<K>_<impressions>.parquet`,
  the config under
  `5_03_Pipeline_Evaluation/artifacts/baseline_content_popularity_combined_popularity_reranker/`.

Both write their causal-feature caches under
`5_03_Pipeline_Evaluation/artifacts/replayed_features/`.

## Attention reranker pipeline

`PE_attention_reranker_niche_pipeline.ipynb` is a single end-to-end notebook: it trains the
niche candidate generator and the `RR_4` `+metadata` reranker inside the notebook, generates
candidates in memory, and evaluates the final Top-K.

- **Candidate generation**: the niche two-tower model from
  `CG_niche_content_two_tower_sequential_combined.ipynb` is trained (or loaded from
  `candidate_generator/model.pt`) and builds 450 candidates per request: 200 rolling
  prior-hour popularity + 250 niche.
- **Reranking**: the `RR_4` `+metadata` model (`softmax+lambda_ndcg`) is trained (or loaded
  from `reranker/model.pt`) on the generator's training candidates.

Training uses the generator's train candidates joined to `labels_train`; a generated candidate
that was never shown has no label and is filled with `clicked = 0` for the listwise loss.
Evaluation still treats unshown candidates as unknown, so clicks the generator missed stay in
the denominator and count as misses. `TRAIN_IMPRESSIONS` and `EVAL_IMPRESSIONS` control runtime.

The final Top-K lists (`K_VALUES = (5, 10, 30)`, so `K = 30`) are evaluated with every metric
in `cg_evaluation.py` at `M = K`, plus MRR@K and nDCG@K, matching the baseline pipelines.

Only models, the reranker preprocessor, the causal-feature replay cache, the final Top-K and
the metric tables are written. Candidate lists and scores stay in memory. Outputs are
`results/pipeline_attention_reranker_niche_generated_<split>_<impressions>.csv`,
`5_03_Pipeline_Evaluation/artifacts/attention_reranker_niche_generated_<split>_top<K>_<impressions>.parquet`,
and the model/cache folders under `5_03_Pipeline_Evaluation/artifacts/attention_reranker_niche/`.

For a test run, set `TARGET_SPLIT = "test"` and rerun. Checkpoints are reused when present and
compatible; delete the relevant `model.pt` to force retraining. The standalone CG and RR4
notebooks remain unchanged and can be run independently.

## Comparing the pipelines

`results.ipynb` loads every metrics CSV matching `results/pipeline_*.csv`, so all pipelines are
compared on the same impressions and cutoffs. It reports accuracy (`ndcg`, `mrr`,
`recall_pooled`, `recall_impression_mean`, `hit_rate`) and reach/diversity
(`long_tail_recall`, `cold_article_recall`, `novelty_bits`, `catalogue_coverage`,
`personalisation`) as tables and plots, plus a written comparison of the three pipelines on the
test split. It does not write any artifacts; run the three pipeline notebooks first.

## Running it

All pipeline notebooks are self-sufficient. The attention notebook trains the candidate
generator and RR4 in one run; the two baseline notebooks rebuild their deterministic candidate
generator in memory. `TRAIN_IMPRESSIONS` (attention only) and `EVAL_IMPRESSIONS` control the
sampled impressions; `None` uses every impression. The first full run is long because it
replays the causal timeline for the generated candidates. The feature replay cache and, for
the attention notebook, the two model checkpoints are reused on later runs, so reruns are
much faster.

The standalone candidate-generation and RR4 notebooks remain available for independent
experiments; neither pipeline notebook requires them to be executed first.

### Running on Google Colab

Each pipeline notebook exposes an `ENVIRONMENT` switch in section 1.2. Set it to `"colab"` to
mount Google Drive, read the project code from `/content/drive/MyDrive/BT4222-Group-6-`, and
read and write data, artifacts and results under `/content/drive/MyDrive/v3-mindsmall`. Local
runs keep `ENVIRONMENT = "local"` with the existing `NEWSREC_ROOT` / `NEWSREC_DATA_ROOT`
overrides.

Outputs are the final Top-K parquet under `5_03_Pipeline_Evaluation/artifacts/` and the metric
CSV/JSON under `results/`. The attention notebook also writes model checkpoints and the
preprocessor under `5_03_Pipeline_Evaluation/artifacts/attention_reranker_niche/`; the content
reranker baseline writes its config under
`5_03_Pipeline_Evaluation/artifacts/baseline_content_popularity_combined_content_similarity/`,
and the popularity reranker baseline writes its config under
`5_03_Pipeline_Evaluation/artifacts/baseline_content_popularity_combined_popularity_reranker/`.
