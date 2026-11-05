# 5.03 Pipeline evaluation

Evaluates candidate generation and reranking **together**, the way they run in deployment,
at the candidate budget chosen in candidate generation.

| Notebook | Purpose |
|---|---|
| [PE_two_stage_pipeline.ipynb](PE_two_stage_pipeline.ipynb) | Rerank generated validation candidates with the final `RR_4` model and evaluate every popularity × content budget |
| [PE_final_pipeline.ipynb](PE_final_pipeline.ipynb) | Run the final system: the 500-candidate generator output (popularity top 200 + content top 300) reranked by the final `RR_4` model, evaluated on all retrieved and on shown-only candidates |
| [PE_baseline_pipeline.ipynb](PE_baseline_pipeline.ipynb) | Run the content-similarity baseline reranker over the combined generator's Top-500 candidates, evaluated at the final Top-K with every `cg_evaluation.py` metric plus MRR@K and nDCG@K |
| [PE_attention_reranker_pipeline.ipynb](PE_attention_reranker_pipeline.ipynb) | Run the combined generator (popularity top 200 + content top 300) through the `RR_5` attention reranker and evaluate the final Top-K with the same metric table as the baseline pipeline |

## Candidate budgets

The candidate file (`recommendations_validation_top1250.parquet`) holds 1,250 candidates per
validation impression: ranks 1–500 from the popularity model and 501–1250 from the content
model, with no article repeated. A budget `(P, C)` keeps popularity ranks `1..P` and content
ranks `501..500+C`, so `M = P + C` candidates reach the reranker.

Candidates are scored once at the largest budget. The reranker scores each candidate
independently, so every smaller budget is a filter on the same scores rather than a new run.

## What is reported

For each budget, two orderings of the same candidates:

- **Generator only**: the two retrievers' rankings interleaved (popularity 1, content 1, …)
- **Generator + reranker**: ordered by the reranker score

Metrics come from `rr_evaluation.evaluate_retrieved_candidates`: candidate recall, MRR,
nDCG@5/10 and Recall@10. They are known-click metrics: unshown retrieved articles have
unknown relevance, and clicks the retrievers missed count as misses. Candidate recall is
the ceiling for every ranking metric.

The budget itself (`CHOSEN_POPULARITY = 200`, `CHOSEN_CONTENT = 300`) is chosen in
candidate generation, on recall@M against catalogue coverage. Raw pipeline nDCG@10 is
biased towards small, popularity-only budgets: unshown articles can never count as clicks,
and the logging policy mostly showed popular ones. So the notebook reports its argmax as a
diagnostic only. At the chosen budget it compares the generator order, the free reranker
and quota merges, both on all retrieved candidates and on shown candidates only.

## Baseline pipeline

`PE_baseline_pipeline.ipynb` evaluates the content-similarity baseline reranker end to end:

- **Candidate generation**: `CG_content_two_tower_rolling_popularity.ipynb` writes 500
  candidates per impression (`cutoff` 1-200 popularity, 201-500 content).
- **Reranking**: the content-similarity baseline, with all five rules compared (text, entity,
  and fusion with text weight 0.25 / 0.50 / 0.75).

The final Top-K list (`K = 10` by default) is evaluated with every metric in
`cg_evaluation.py` at `M = K`, plus MRR@K and nDCG@K. The best content rule is selected on
validation nDCG@K and saved to
`results/pipeline_<candidate_tag>_content_similarity_selection_validation.json`; a test run
reuses that selection instead of re-selecting.

Outputs are `results/pipeline_<candidate_tag>_content_similarity_<split>_<impressions>.csv`
and
`5_03_Pipeline_Evaluation/artifacts/<candidate_tag>_content_similarity_<split>_top<K>_<impressions>.parquet`.
Replayed content features are cached under
`5_03_Pipeline_Evaluation/artifacts/replayed_features/` with a candidate-source-specific name.
A test run also needs `recommendations_test_top500.parquet` from the combined generator
notebook.

## Attention reranker pipeline

`PE_attention_reranker_pipeline.ipynb` evaluates the combined candidate generator with the
`RR_5_generated_candidates` attention reranker:

- **Candidate generation**: `CG_content_two_tower_rolling_popularity.ipynb` writes 500
  candidates per impression (`cutoff` 1-200 popularity, 201-500 content).
- **Reranking**: the `+metadata`, `softmax+lambda_ndcg` reranker trained on the generator's
  training slates in `RR_5_generated_candidates.ipynb`.

The final Top-K list (`K = 10` by default) is evaluated with every metric in
`cg_evaluation.py` at `M = K`, plus MRR@K and nDCG@K. `generator_order` is kept as the
reference row, matching `PE_baseline_pipeline.ipynb`.

The notebook reuses `generated_<split>_scores.parquet` from the RR_5 artifact folder when it
covers the selected candidates. If that file is missing or incomplete, it loads the RR_5
checkpoint and preprocessor, replays the causal features (reusing an existing cache when
available) and scores the candidates itself, so the notebook can run on its own from the CG
candidate file.

`RR5_RUN_TAG` selects which RR_5 run to consume. `None` reads the legacy
`branch_interaction_generated` folder (the 5,000-training-impression run). To change the
sample size, set `TRAIN_IMPRESSIONS` and `RUN_TAG` in `RR_5_generated_candidates.ipynb`,
rerun it, then set the same tag in `RR5_RUN_TAG` here. Tagged runs keep separate checkpoints,
preprocessors, replay caches and score files, so sample sizes can be compared without
overwriting each other.

Outputs are `results/pipeline_attention_reranker_generated_<split>_<impressions>.csv` and
`5_03_Pipeline_Evaluation/artifacts/attention_reranker_generated_<split>_top<K>_<impressions>.parquet`.
A metadata JSON with the source paths and headline metrics is written next to the CSV.

For a test run, run the CG notebook with `TARGET_SPLIT = "test"`, then either run the RR_5
test section or let this notebook's fallback score the test candidates. The fallback replays
test features, which is the slow path.

## Running it

Scoring replays the causal timeline to build features for every (impression, candidate)
pair. `EVAL_IMPRESSIONS = 10_000` (about 12.5M pairs at the full 500 + 750 budget) keeps
it laptop-sized; `None` scores all 31,624 impressions (about 40M pairs). The score file is
reused on later runs.

Outputs: score files under `<DATA_ROOT>/5_Models/5_03_Pipeline_Evaluation/artifacts/`, and the
budget grid in `<DATA_ROOT>/results/pipeline_budget_grid_<impressions>.csv`.

`PE_final_pipeline.ipynb` reads `recommendations_validation_top500.parquet` and writes
`<DATA_ROOT>/results/pipeline_final_top500_<impressions>.csv`. Each candidate's score
doesn't depend on the rest of the list, so pairs already scored by
`PE_two_stage_pipeline.ipynb` with the same reranker are reused (`REUSE_SCORES`), and only
the remaining pairs are replayed.
