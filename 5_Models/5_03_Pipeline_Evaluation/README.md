# 5.03 Pipeline evaluation

Evaluates candidate generation and reranking **together**, the way they run in deployment,
at the candidate budget chosen in candidate generation.

| Notebook | Purpose |
|---|---|
| [PE_two_stage_pipeline.ipynb](PE_two_stage_pipeline.ipynb) | Rerank generated validation candidates with the final `RR_4` model and evaluate every popularity × content budget |
| [PE_final_pipeline.ipynb](PE_final_pipeline.ipynb) | Run the final system: the 500-candidate generator output (popularity top 200 + content top 300) reranked by the final `RR_4` model, evaluated on all retrieved and on shown-only candidates |

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
