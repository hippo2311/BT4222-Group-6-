# Feature handoff review — 20 October 2026

## Current status: v2 full build validated

The full v2 run is complete and ready for offline model building. All 40,363,532 feature rows,
50 model inputs, 2,609,219 profile references, source labels, eligibility and complete validation/test
slates passed checks. Eleven independently reconstructed requests include a long history and the
shared boundary second. All 67 tests passed. The current logistic smoke check achieved AUC 0.70319 /
NDCG@10 0.42518 versus popularity 0.65045 / 0.36476 on all validation slates.
No further full rebuild is required for this handoff. The review history below explains the changes.

## Integration with main (`632067e`)

Kept the new representation diagnostics: per-field/split vocabulary/OOV coverage,
title/abstract/fused neighbours, article fallback flags, and bounded-memory
slate/click/history coverage. Also kept implementation-aware representation cache
invalidation, distinct-history validation before deduplication, the clearer
`history_over_cap` name, external `NEWSREC_CODE` support, and upstream regression tests.

Retained the local full-history text/entity/interest pooling, 50-feature default,
categorical schema, importable notebook methods, force/range controls and ETA.
Full history references reconstruct interest centroids without storing duplicate
centroids per request; the reader also supports upstream centroid-only references.
The final test count and real smoke outcome are recorded in the local
`results/feature_review_status.json`. The v2 full rebuild has now been completed and audited.

The completed v1 full run succeeded: 40,363,532 feature rows, 520,464 timestamp
batches, 9,127.7 seconds for feature construction, and no recorded notebook errors.
The following changes introduced **v2**, whose full build has now passed validation.

| Review point | Decision |
|---|---|
| Friday absent from training; weak calendar effects | Exclude `weekday` and `day_part` by default. Keep calculations available for controlled ablations. |
| Almost constant startup/boundary flags | Exclude `population_startup` and `age_left_censored` by default. |
| Notebook functions unavailable to teammates | `newsrec.ranking` imports the notebook's tagged methods and selections. Calculations remain visible/editable in the notebook. No build/example cells execute on import. Restart Python after notebook edits. |
| Retrieved candidates not exercised | Regression tests score genuinely history-only/unexposed articles and an unseen category, reject future articles, and pass finite features through a trained logistic ranker. A real history-only preview is included in notebook verification. |
| Censoring | The old flag identifies the exact log boundary only. Document this and keep it out of default model inputs. All ages are exposure-observation ages, not publication ages. A 24–48-hour window would be an unverified heuristic. |
| Capped-history pooling | Text, entity and category-interest vectors use the full supplied snapshot plus every allowed earlier click, preserving repeats. Sequence references alone are capped; recent views retain their independent item/time limits. |
| Unverified snapshot ordering | Sequence references use a reproducible random subset instead of privileging the highest article IDs. No snapshot-recency claim is made. |
| Profile reconstruction | Save `full_history_rows` as well as capped rows/masks. These reproduce full pooled features exactly; memory/storage costs increase and must be measured on the new full run. |
| Hardcoded CTR support | `Settings(article_ctr_min_exposures=100)` controls `article_low_support`, independently of half-life support. |
| Categorical inputs | `categorical_columns` and per-feature metadata identify category/subcategory and any re-enabled calendar buckets. Use train-fitted embeddings/one-hot encoding; do not scale category IDs as ordered numbers. |
| Causal edge cases | Tests cover the shared train/validation boundary second, same-time/future label mutations, histories over 50 items, cap-invariant pooling, frozen evaluation, and exact saved-reference reconstruction. |

## Model smoke check on completed v1 artifacts

Train-only preprocessing and logistic fitting used a reproducible uniform sample
of **100,000 rows from the sampled training dataset**. Evaluation used **all
334,913 validation impressions / 13,192,987 candidate rows**, with complete slates.
No test labels were used. The four excluded default features above were omitted;
the underlying v1 profiles were still capped, so these are not v2 results.

| Model | Mean impression AUC | Mean NDCG@10 |
|---|---:|---:|
| Smoothed prior article CTR | 0.65045 | 0.36476 |
| Logistic smoke model | 0.70295 | 0.42479 |

Tied scores receive average/expected credit. These results show ranking signal;
they are neither a leakage proof nor a tuned baseline comparison. The local
`results/model_smoke_check_v1.json` preserves that historical check; `results/model_smoke_check.json` now records the current v2 check.

## Reproduction commands (not required again for the current handoff)

```bash
python run_all.py --from-stage 3 --to-stage 4 --force
PYTHONPATH=src python -m newsrec.sanity --max-train-rows 100000
```

The v2 full-run counts, coverage, quality and cost have now been checked. The current
`results/feature_readiness_audit.json` applies to v2; the v1 audit is archived separately.
Smoke builds use a separate directory and do not replace the completed full run.

Retrieval Recall@M, trained re-ranking, cold-start quality and the connected
two-stage pipeline remain downstream work. Plan group ablations for overlapping
cosines, multi-interest, half-life and request context. Slate context stays opt-in:
compute it on the actual retrieved slate and measure its distribution shift.

Prequential evaluation assumes prompt earlier feedback; retain frozen-mode
sensitivity checks. Logged exposure bias remains. Sampled training probabilities
are not calibrated CTR: future calibration work needs unsampled chronological
validation and must account for slate-dependent negative inclusion probabilities.
