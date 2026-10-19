# 3. Article and user representations

Run these notebooks in order after data preparation.

| Notebook | What it does | Main outputs in `artifacts/` |
|---|---|---|
| [3_01_news_embeddings.ipynb](3_01_news_embeddings.ipynb) | Fits training-only TF-IDF/SVD and frozen title/abstract transformations; builds fused article text vectors, category mappings and title-token inputs; reports vocabulary coverage and neighbours. | `article_index.parquet`, `title64.npy`, `abstract64.npy`, `text64.npy`, title-token IDs/masks/vocabulary, text diagnostics and manifest |
| [3_02_entity_embeddings.ipynb](3_02_entity_embeddings.ipynb) | Parses article entities, pools valid canonical pretrained entity vectors, records missing-vector fallbacks, and measures article/event-weighted coverage. | `article_entity100.npy`, entity mappings/links/quality, `article_fallback.parquet`, `usage_coverage.csv` and manifest |
| [3_03_user_profiles.ipynb](3_03_user_profiles.ipynb) | Establishes observation-time article eligibility, pools supplied histories, and demonstrates causal text/entity/recent and category-interest profiles using earlier feedback. | `article_availability.parquet`, `supplied_*_profiles.npy`, supplied-profile index, manifests and profile checks |

All arrays share the article-row mapping; row 0 is padding. Text and entity vectors use separate spaces. Eligibility is based on observable discovery, not publication time. Static supplied profiles contain no logged feedback; causal pooled profiles use full prior history, with capped sequence references and a separate recent view.

Execution, cache controls and data locations are described in the [project README](../README.md).

`REBUILD=False` loads saved outputs after checking shared implementation, settings, inputs and output hashes.
Missing outputs or changed shared code trigger a rebuild.
Set `REBUILD=True` after changing calculations/settings, then rerun dependent notebooks. Incompatible/stale files
raise a clear error rather than silently reusing them. Keep representation settings consistent between notebooks.

Outputs live in `3_Representations/artifacts/`:

- `article_index.parquet`: news ID to dense array row; row 0 is reserved for padding.
- `title64.npy`, `abstract64.npy`, `text64.npy`: text vectors; zero vectors have validity masks.
- `article_entity100.npy`, entity index/quality/links: a separate entity coordinate system.
- Title token IDs, masks and vocabulary: inputs for the later NRMS model.
- `article_availability.parquet`: earliest observable discovery, not publication time.
- `supplied_*_profiles.npy` and profile index: supplied-history-only user profiles for baselines/clustering.
- Manifests: settings, source/output hashes and versions.
- `text_quality.csv`, `text_neighbors.csv`: per-field/split vocabulary coverage and multi-query view comparisons.
- `article_fallback.parquet`: explicit text/entity availability for each article.
- `usage_coverage.csv`: event-weighted slate/click/history coverage, scanned once per split in batches.

Causal profiles use supplied history once plus earlier logged clicks. Current/same-second feedback is excluded.
Ranking histories are capped; recent profiles use timestamped clicks only. Stage 4 writes per-request history
references for retrieval/rankers. Entity and text spaces remain separate; missing vectors retain fallback flags.
Category-interest centroids use full causal click history via per-category vector sums; the sequence cap does
not remove dominant interests. Request references persist these centroids for exact reconstruction.
`long_text` remains a supplied-history baseline. `history_over_cap` does not imply truncation of static profiles.
The synthetic check in notebook 3_03 verifies delayed feedback and dominant interests beyond the cap.

Coverage and cost reports are written under `results/`. Saved files survive closing VS Code; Python variables do not.
Later notebooks can load these files with `pd.read_parquet(...)` or `np.load(...)` without rebuilding representations.

Edit the method cells to change calculations, then set `REBUILD=True` and rerun affected/downstream notebooks.
The module defaults remain available to later Python consumers; notebook builds explicitly use their visible methods.
The causal profile methods are repeated in `3_03` and `4_01`; keep both copies consistent when changing that method.
Restart the kernel after editing imported helpers.

## Improvements and rationale

### 3_01 — Text embeddings

| Step | Code added or changed | Reason | Output |
|---:|---|---|---|
| 1 | `tfidf.fit_transform(titles[train])` | Fit the vocabulary and IDF on training articles only to prevent validation/test leakage. | Frozen training-only TF-IDF model |
| 2 | `svd.fit(train_tfidf)` | Learn dimensional reduction from training data only. | Frozen 64-dimensional SVD space |
| 3 | `tfidf.transform(titles)` and `tfidf.transform(abstracts)` | Encode every split in the same frozen coordinate system. | `title64.npy` and `abstract64.npy` |
| 4 | `normalise(0.7 * title + 0.3 * abstract)` | Combine title and abstract information into one compact representation. | `text64.npy` |
| 5 | Fall back to title or abstract when one vector is missing. | Avoid a zero fused vector when one valid text field is available. | Improved text-vector coverage |
| 6 | Count `empty_text`, `zero_tfidf`, and `oov_fraction` by field and split. | Identify articles that the training vocabulary cannot represent well. | `text_quality.csv` |
| 7 | `view_neighbors(ROOT, queries=3, count=3)` | Inspect several articles instead of judging quality from one example. | `text_neighbors.csv` |
| 8 | Compare `title64`, `abstract64`, and `text64` neighbours. | Check whether abstracts improve semantic similarity or introduce noise. | View, rank, titles, categories, and cosine scores |
| 9 | Check dimensions, finite values, padding, and vector norms. | Detect invalid arrays before retrieval or feature engineering. | Validation assertions and representation summary |
| 10 | Display `representation_cost.csv`. | Report wall time, memory use, CPU information, and output size. | Computational-cost table |

### 3_02 — Entity embeddings

| Step | Code added or changed | Reason | Output |
|---:|---|---|---|
| 1 | Parse entity JSON into `missing`, `empty`, `invalid`, and `present` states. | Distinguish absent annotations from malformed data. | Status columns in `entity_quality.parquet` |
| 2 | `sorted(set(title_ids) \| set(abstract_ids))` | Prevent a Wikidata entity from being counted twice. | Deduplicated entities per article |
| 3 | Use training-release entity embeddings only. | Keep one consistent entity coordinate system. | `entity_vectors.npy` |
| 4 | Mean-pool matched vectors and apply L2 normalisation. | Produce one fixed-size vector per article for cosine similarity. | `article_entity100.npy` |
| 5 | Check shape, `float32`, finite values, padding, and unit norms. | Confirm that the matrix satisfies the downstream contract. | Validation assertions |
| 6 | Create `has_text` and `has_entity` flags. | Make missing representations explicit to downstream models. | Availability flags per article |
| 7 | Define `text_and_entity`, `text_only`, `entity_only`, and `neither` fallbacks. | Specify how to handle articles with missing vectors. | `article_fallback.parquet` |
| 8 | Report coverage for slate, click, and supplied-history entries. | Measure missing vectors where they directly affect recommendations. | `usage_coverage.csv` |
| 9 | Separate entity-ID coverage from article-vector coverage. | Avoid reporting coverage with the wrong denominator. | Coverage tables with explicit cohorts |
| 10 | Read parquet logs in batches. | Audit full logs without loading them entirely into memory. | Bounded-memory coverage scan |

### 3_03 — User profiles

| Step | Code added or changed | Reason | Output |
|---:|---|---|---|
| 1 | Pool all valid supplied-history text and entity vectors. | Build static profiles without using logged feedback labels. | `supplied_text_profiles.npy` and `supplied_entity_profiles.npy` |
| 2 | Validate every distinct `(user_code, history)` pair before pooling. | Detect histories that change unexpectedly across rows or splits. | Clear error for conflicting histories |
| 3 | Rename `history_truncated` to `history_over_cap`. | Clarify that the static profile still uses the complete supplied history. | Correct flag in `supplied_profile_index.parquet` |
| 4 | Maintain `interest_sums[category]` for each user. | Preserve complete causal interest information without storing every old event. | Full-history category vector sums |
| 5 | Build interest centroids from `interest_sums`. | Prevent a dominant interest from disappearing after its articles leave the capped sequence. | Correct multi-interest vectors |
| 6 | Apply `history_cap` only to `history_rows`. | Keep a bounded NRMS sequence while retaining complete category interests. | Fixed-length history rows and masks |
| 7 | Keep `long_text` as the supplied-history-only profile. | Separate the static baseline from profiles updated using observed feedback. | Stable long-term baseline vector |
| 8 | Use `begin()` to create pre-feedback state for timestamp `t`. | Ensure requests at the same timestamp use the same prior information. | Frozen same-time profiles |
| 9 | Use `profile()` without consuming current labels. | Prevent the current click target from entering its own features. | Leakage-safe request profiles |
| 10 | Use `finish()` only after the timestamp batch is scored. | Allow feedback to affect later timestamps only. | Prequential causal updates |
| 11 | Run `check_temporal_profiles(TemporalState, SETTINGS)`. | Test dominant interests beyond the cap, same-time isolation, and delayed feedback. | Synthetic causal-check table |
| 12 | Save `interest_vectors` in request-profile references. | Let Stage 4 reconstruct exact full-history interest vectors. | Interest vectors in request-profile parquet files |
| 13 | Add backward-compatible profile reconstruction. | Read older references while using exact saved centroids in new runs. | Correct reconstructed profiles |
| 14 | Display coverage and computational-cost reports. | Verify profile availability and runtime requirements. | Coverage and cost tables |

### Additional reliability improvements

| Code change | Reason |
|---|---|
| Check `implementation_hashes` in `saved_stage_ready()`. | Rebuild cached artifacts when shared representation code changes. |
| Add synthetic end-to-end tests for all three notebooks. | Verify OOV handling, invalid entity JSON, missing vectors, empty or conflicting histories, causal updates, cache invalidation, and Stage 4 compatibility. |
| Clear old notebook outputs. | Avoid presenting results from the previous implementation as current results. |
| Use concise English comments and input/output descriptions. | Make each calculation easier to understand, review, and reproduce. |
