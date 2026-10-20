# Project methods journal

This journal records the approaches discussed, implemented and checked during data preparation, representation building and feature engineering. Discussion alone does not mean an approach was implemented or validated. The final section describes the current adopted approach and completed full run, as of **20 October 2026**.

## 1. Data acquisition and split workflow

The initial recovery workflow downloaded original MIND-large releases and split the original training impressions chronologically: earliest 85% for training, remaining 15% for validation, and the labelled dev release for testing.

Because the team already had agreed splits on Google Drive, downloading and splitting everything again was unnecessary for ordinary team use. A separate download notebook was considered; the adopted workflow instead puts a missing-input Drive fallback at the start of data preparation. Existing local files are reused. If Drive is unavailable, the notebook directs the user to stage 0.

Stage 0 was consolidated into `download_split_process.ipynb`: download, extraction and splitting in one notebook. Cached archives and extracted raw files can be reused. Historical download checks on 17 October encountered HTTP 401/409 responses, so source URLs/access remain an external dependency of the recovery path. These observations are not a claim about current availability.

Agreed split files are preserved unless rebuilding is explicitly requested. A deterministic timestamp sort can assign same-second boundary rows differently from another splitting implementation; changing the split requires rebuilding downstream datasets. The current team data has training and validation requests sharing the boundary second, so feature replay groups those requests together before applying feedback.

Data preparation retains valid rows, empty histories, repeat exposures and missing article fields. It validates supplied-history consistency, parses logged labels, and creates stable cross-split keys and integer codes. Catalogue-wide descriptive counts remain EDA information; they must not substitute for strictly prior event statistics in ranking.

## 2. What EDA suggested

The EDA notebooks examined impression/slate structure, users and histories, and article/temporal behaviour. Their findings informed candidate features rather than establishing that those features improve ranking.

| Observation | Method considered or implication |
|---|---|
| Slate length is strongly associated with observed CTR | Exclude current `slate_len`; compare ranking quality per impression. |
| Logged position is shuffled file order | Exclude `position`; do not treat it as a display rank. |
| Labels exist for shown pairs only | Sample exposed negatives; never invent negative labels for unshown retrieved articles. |
| Many negatives per click | Use four negative draws per positive as the initial setting; larger counts remain a sensitivity experiment. |
| Re-shown, previously passed-over articles have lower observed CTR | Include prior pair exposures, prior outcomes and exposure gaps. |
| Slate composition effects weaken after controlling for slate size | Keep slate-relative features optional. |
| Many users have few impressions but useful supplied histories | Combine content with supplied histories and explicit cold-start/missing-profile fallbacks. |
| Supplied histories are static in the validated data | Append only genuine earlier observed clicks, preserving repeats. |
| Category affinity and previously seen categories show stronger engagement | Include category/subcategory affinity and `seen_category`. |
| Popularity is concentrated and low-support CTR is unstable | Use prior counts, expanding smoothed CTR and support flags. |
| Engagement changes with observed age | Model exposure-observation age jointly with popularity; investigate past-only decay estimates. |
| Many held-out articles were unexposed during training | Support time-eligible content retrieval and unexposed-article fallbacks. |
| Time-of-day and visit-gap effects are comparatively weak | Treat request context as an ablation candidate; calendar buckets were later disabled by default. |

The original EDA recommendation of a 50-item history cap was subsequently refined: the cap now applies to sequence references, not pooled user interests.

## 3. Where calculations should live

Three code arrangements were explored:

1. **Shared Python implementations with thin notebooks.** This reduced repetition and made code importable, but important calculations were less visible to readers assessing the notebooks.
2. **Fully self-contained notebooks.** Moving the implementation inline improved visibility, but repeated validation, file handling and cache code made the notebooks long.
3. **Visible notebook methods with generic shared infrastructure.** This is the adopted arrangement: core representation/feature calculations and worked examples are in notebook cells; common checks, replay, caching and storage remain in `src/newsrec/`.

Feature selection and custom formulas are editable through `ENABLED_GROUPS`, `DISABLED_FEATURES` and `EXTRA_FEATURES`. The earlier separate feature module was removed so the notebook owns its feature catalogue and calculations.

To support downstream Python consumers without hiding the notebook implementation, `newsrec.ranking` loads trusted tagged definitions and selections. It does not execute setup, download, example or build cells. Python must be restarted after notebook edits. Causal profile methods are repeated in the representation and feature notebooks and have a shared fallback; tests compare their definitions to guard against divergence.

Notebook readability was also revised: feature groups and the model contract are explained separately, redundant setup/check sections were shortened, and shared worked examples demonstrate the methods. Optional examples remain in an appendix.

## 4. Article representation methods

### Text and title-token inputs

The implemented text baseline fits TF-IDF vocabulary/IDF and truncated SVD using **training titles only**. Title and abstract are transformed into the same frozen coordinates, normalised, and fused with a configurable title weight. If only one field has a usable vector, that field supplies the fallback. The default text dimension is 64.

Title token IDs, masks and a training-fitted vocabulary are saved for later NRMS work. Category/subcategory mappings are also training-fitted, with an unknown fallback for held-out categories.

Using a title-fitted space for both fields makes their coordinates comparable, but abstracts can contain out-of-vocabulary content. Field/split vocabulary diagnostics, zero-vector checks and title/abstract/fused neighbour comparisons were added to make this limitation visible. Fitting an alternative abstract vocabulary or a different text encoder remains future experimentation, not a validated change.

This is an offline training-window fit. Held-out text does not fit the model, but this is not a claim that the vocabulary was refitted using only the prefix before every training impression.

### Entities and missing-vector coverage

The entity method parses title/abstract annotations, identifies usable canonical pretrained entity vectors from the training release, pools article entities and normalises the result. Text and entity coordinate systems stay separate; missing vectors are represented by zeros and validity/fallback information.

Coverage diagnostics distinguish entity lookup coverage, article-level vector availability and event-weighted availability. A later integration from `main` added article fallback tables and bounded-memory counts for slate exposures, clicked exposures and supplied-history entries. Click labels are used for these descriptive coverage reports, not representation fitting.

### Observable eligibility

Article eligibility uses the earliest observable supplied-history discovery or slate appearance, filtered by `available_from <= request_time`. Test-user histories are not backdated to the training window. This catalogue is a proxy based on what can be observed in the logs, not a publication-time catalogue.

## 5. User profiles and history caps

Static supplied-history profiles pool all valid supplied-history vectors. They contain no logged feedback and are suitable as a supplied-history baseline or as inputs to later user clustering. Only training users may fit future cluster/scaler models.

The initial causal implementation maintained full history counts but pooled some vectors from the capped retained history. With unverified snapshot order, deterministic article-row sorting could favour higher IDs in that retained subset. Category interests could also disappear when their supporting articles left the sequence cap.

Two remedies were considered during review/integration:

- Maintain incremental per-category vector sums and save category centroids in each request reference.
- Pool from full causal history and persist full history row references, alongside the capped sequence.

The adopted approach is the second. Text, entity and category-interest vectors use the full supplied snapshot plus every allowed earlier click, preserving repeated clicks. The sequence cap no longer decides which articles define pooled interests. Snapshot sequence references use a reproducible random subset when order is unverified, without claiming recency.

The recent view uses timestamped earlier clicks only, with independent item/time limits. `long_text` remains the supplied-history-only baseline. Full row references support exact reconstruction without also storing duplicate interest centroids. The reader supports the upstream centroid-only reference format for compatibility. Memory and storage grow with full history and are measured in the completed run.

## 6. Causal ranking features and evaluation modes

The feature pipeline uses **begin → score all requests at timestamp t → finish**. Current and same-second feedback is excluded from scoring. Global/article/pair counts and user click histories update only after all same-time requests have been scored.

The default is prequential evaluation: earlier validation/test feedback can affect later requests. This assumes prompt feedback availability; actual click-arrival delays are unknown. A frozen-target mode provides a sensitivity alternative, but it freezes the target split's allowed event feedback, so its exposure-versus-click semantics should be considered when interpreting comparisons.

Candidate feature groups include content, user history, category/text/entity matching, repeat exposure, smoothed popularity, observed freshness, multi-interest matching, half-life and request context. Missing measurements use explicit validity information rather than silently interpreting zero as an observed similarity or age.

Half-life compares disjoint completed six-hour windows using Beta(1,1)-smoothed CTR. Both windows need configurable exposure support, and a finite positive decay estimate requires recent CTR below early CTR. Unsupported estimates have validity/reason flags. Staleness uses the estimate with a six-hour grace rule. These quantities describe observed exposure/engagement decay, not publication age or true expiry.

## 7. Review-driven feature revisions: v1 to v2

The earlier default schema had 54 model features. Review identified extrapolation, interpretation and engineering concerns.

| Concern | Adopted revision |
|---|---|
| Friday is absent from training; calendar buckets have weak supporting evidence | Disable `weekday` and `day_part` by default; retain their calculations for controlled ablations. |
| `population_startup` contributes mainly at the beginning of replay | Exclude it from the default model allowlist. |
| `age_left_censored` flags only exposure at the exact log boundary | Document its narrow meaning and exclude it by default. A 24–48-hour boundary would remain a heuristic, not verified publication censoring. |
| Category IDs could be treated as arbitrary numeric magnitudes | Add `categorical_columns` and preprocessing metadata for embeddings/one-hot encoding. |
| Article CTR support threshold was hardcoded | Add independent `article_ctr_min_exposures`, default 100. |
| Stripping labels from logged pairs did not exercise genuinely retrieved items | Add history-only/unexposed candidate tests, unseen-category scoring and future-item rejection. |
| Cap-dependent pooling and reconstruction | Use full causal pooling, persist full history rows, and test histories over 50 items. |
| Later consumers could not import notebook functions directly | Add the trusted notebook-definition adapter while preserving visible calculations. |

Request-gap and first-visit features remain enabled, but their incremental value needs ablation. Slate-relative median advantage remains opt-in because it depends on the actual candidate set and can shift between logged and retrieved slates. Subtracting a common median alone does not change within-slate order.

Multi-interest and several cosine measures may overlap; half-life is supported only for a subset of rows. These are reasons for controlled ablations, not evidence that a feature group has already improved performance. User-interest clusters and graph embeddings remain deferred work.

## 8. Training-negative sampling and calibration

The adopted sampler takes deterministic exposed-negative draws per positive, normally four without replacement within each draw, and retains the unique union of selected pairs plus all positives. Seeds depend on the configured seed and impression/positive identity. Shuffling input rows must not change the result.

Sampling happens after computing features/state from allowed complete logged event history. Validation/test slates are not negatively sampled. Unshown retrieved items receive no invented labels.

The sampled training positive rate is about 22%, while held-out exposure CTR is about 4%. Consequently, a ranking score is not automatically a calibrated click probability. Calibration on an appropriate unsampled chronological validation slice, slate-dependent inverse-selection weighting and larger negative-count sensitivity checks were discussed; they have not been implemented as a calibration study.

## 9. Execution, saved outputs and team integration

Earlier terminal execution produced timestamped notebook copies under `runs/`. The adopted runner instead saves outputs directly into source notebooks after completed cells. Saved Parquet/NumPy data survives editor restarts; notebook variables do not.

`REBUILD=False` validates/reuses compatible outputs and builds missing outputs. Inclusive stage ranges select what runs; `--force` rebuilds selected stages without permanently changing notebook preferences. Re-splitting remains a separate explicit action. Formula-only notebook changes require rebuilding even if their column names remain unchanged.

Direct feature-notebook execution defaults to a 1,000-timestamp smoke run through `FULL_RUN=False`. Terminal execution defaults to full features, with an explicit smoke override. Smoke and full feature directories are separate.

A cell heartbeat was added to distinguish long-running work from a stalled-looking terminal. Feature replay then gained timestamp progress, an approximate remaining time and a local finish estimate, blending timestamp and candidate-volume progress. This is a heuristic; workload changes and final checks can make early estimates optimistic. Interrupted builds restart because the SQLite pair store alone is not a complete state checkpoint.

Integration with `main` commit `632067e` retained representation quality/coverage reports, shared-implementation cache invalidation, distinct-history validation before deduplication, the `history_over_cap` description, external code/data-root support and additional tests. Full causal pooling, the selected feature schema and the notebook import interface were preserved when resolving overlaps.

## 10. Validation methods explored and completed

Validation grew from examples and input guards to three complementary levels:

1. **Automated regression tests:** training-only fits, same-time/future-label mutations, row-order invariance, frozen evaluation, deterministic sampling, long-history pooling, reference reconstruction, retrieved candidates, source-notebook output saving and execution controls.
2. **Full artifact checks:** input/output hashes, schema agreement, all saved feature values, key uniqueness, label/source alignment, complete held-out slates, and user/time/article eligibility.
3. **Independent calculation checks:** reconstruction from raw processed history/event prefixes for stratified requests and edge cases. This is sample-based formula verification, not an independent recomputation of every feature row.

A quick logistic-versus-popularity diagnostic was implemented to test usable ranking signal. A LightGBM check was suggested during review but was not implemented. The diagnostic logistic model uses a reproducible uniform sample of 100,000 rows from the sampled training dataset, train-only scaling/one-hot encoding, and all complete validation slates. AUC/NDCG handle score ties with average/expected credit. Test labels are not used in this comparison.

| Artifact version | Evaluation mode | Popularity AUC / NDCG@10 | Logistic AUC / NDCG@10 |
|---|---|---|---|
| v1, before full causal pooling | Prequential | 0.65045 / 0.36476 | 0.70295 / 0.42479 |
| v2, current completed full build | Prequential | 0.65045 / 0.36476 | 0.70319 / 0.42518 |

Both logistic checks used the same bounded-training/full-validation design. The small difference is not evidence of a statistically established improvement or an ablation result. These checks show ranking signal; they do not establish production calibration, retrieval quality or absence of every possible leakage mechanism.

Historical reports are retained separately from current results. The external review brief was consolidated into this journal so that explored methods and adopted decisions are recorded together.

## 11. Follow-up critique and remaining work

The follow-up review accepted readiness for offline model development on logged slates. The remaining concerns have different scopes:

| Item | What is completed | What remains |
|---|---|---|
| Observation-age censoring | The narrow boundary flag is excluded and publication-age uncertainty is documented. | `article_age_hours` starts at first observed exposure. For articles already live before the window, it understates their publication age; excluding the flag does not recover that missing age. Keep this limitation in the final report. |
| Retrieved candidates | Unit tests cover history-only/unexposed items, unseen categories and future-item rejection. A real-data preview also scores eligible history-only articles using constructed pairs. | Run the contract against actual candidate-generator output, especially zero-exposure and missing-vector cases, when retrieval exists. The preview is not an end-to-end retrieval evaluation. |
| Ranking signal | Logistic fitting on 100,000 sampled training rows and scoring all complete validation slates is complete. | LightGBM, larger-training-sample sensitivity and controlled group ablations have not been run. The tiny v1/v2 metric difference does not establish that full causal pooling improves ranking. |
| Verification scope | Every saved feature row passed structural/value/key/label/eligibility checks; profile-reference checks also cover all saved requests. | Independent raw-history formula reconstruction covers 11 requests. Do not call this independent formula verification of every row or proof of no leakage. |
| Notebook imports | The import adapter and definition-consistency tests are implemented and included in the 67-test suite. | Restart Python after notebook edits and keep the consistency tests running; duplicated definitions and tagged-cell loading remain maintenance risks. |

For model development, record **prequential or frozen mode in every model-result table**, together with training sample size and evaluation scope. Sampled-training scores must not be described as calibrated click probabilities. Neither documentation clarification requires a feature rebuild.

The next experiments are a LightGBM baseline and feature-group ablations on the same validation protocol, followed by real retrieved-candidate integration and end-to-end evaluation once candidate generation is available. These are pending experiments, not completed results.

## Current approach and validated state — 20 October 2026

**Adopted schema:** `ranking-features-v2-full-causal-pooling`. **Status:** ready for offline model development under the declared prequential feedback protocol. No additional full rebuild is required unless inputs, settings, feature selections or calculations change.

| Current component | Adopted method |
|---|---|
| Text | Training-title-fitted TF-IDF/SVD; frozen title/abstract transforms; 64-dimensional fused vectors and fallback |
| Entities | Canonical pretrained training-release lookup; normalised article pooling with missing-vector flags |
| Users | Supplied histories plus allowed earlier clicks; full causal pooled vectors/interests; capped sequence references; timestamped recent view |
| Eligibility | Earliest observable history/slate discovery filtered at request time |
| Ranking inputs | 50 allowlisted features, explicit categorical metadata and separate diagnostics/labels |
| State protocol | Group same-time requests; score before feedback; default prequential mode |
| Sampling | Per-positive exposed-negative draws; all positives retained; complete validation/test slates |
| Implementation | Visible notebook calculations, generic shared infrastructure and importable notebook methods |

The 50 selected columns comprise content (9), history (9), match (11), repeat (6), popularity (4), freshness (2), multi-interest (4), half-life (3) and request context (2). Calendar/startup/boundary flags are disabled by default; slate context is opt-in.

| Split | Feature rows | Positive labels |
|---|---:|---:|
| Training, sampled | 13,084,988 | 2,868,998 |
| Validation, complete slates | 13,192,987 | 514,658 |
| Test, complete slates | 14,085,557 | 574,845 |
| Total | 40,363,532 | 3,958,501 |

The full build processed **520,464 timestamps** through **2019-11-15 23:59:43**. Recorded feature-construction time was about **2h 48m**, sampled peak process RSS **4.82 GB**, and Parquet artifacts **2.68 GB**. These figures do not predict model-training memory requirements.

All saved rows passed finite/type/range/mask, unique-key, source-label, owner/time and eligibility checks. Validation/test slates were complete. Eleven independently reconstructed requests included a history over 50 items and the shared train/validation boundary second. All **2,609,219 request references**, full-history references, title-token inputs and supplied-profile arrays passed validation. **67 automated tests passed**, with no recorded notebook errors.

The fresh v2 logistic diagnostic achieved **impression AUC 0.70319 and NDCG@10 0.42518** against popularity **0.65045 / 0.36476** across all **334,913 validation impressions / 13,192,987 candidate rows**.

Current provenance and evidence:

- `results/feature_readiness_audit.json`: current full-row/profile audit and independent reconstruction results.
- `results/feature_review_status.json`: current handoff status and manifest identity.
- `results/model_smoke_check.json`: current v2 diagnostic model result.
- `results/feature_readiness_audit_v1.json` and `results/model_smoke_check_v1.json`: historical evidence.
- `4_Feature_Engineering/artifacts/prequential-all/`: complete current features, labels, references, schema and reports.

Current full manifest SHA-256:

```text
dc87a817e9c88903c6b7a985dd75e3d55ade379bd18aa9465a692d7a8370f1c0
```

Model owners should use only `model_columns`, encode `categorical_columns`, join labels on both `impression_uid` and `news_code`, fit preprocessing on training data, and evaluate complete held-out slates. The prequential protocol assumes earlier feedback is promptly available; observation-time age/eligibility, logged exposure bias and sampled-probability calibration remain limitations. Retrieval Recall@M, group ablations, cold-start model quality, clusters/graph embeddings and the connected end-to-end recommender are the next work, not completed results.
