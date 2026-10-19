# 3. Article and user representations

Run in order after data preparation:

1. `3_01_news_embeddings.ipynb`: training-only TF-IDF/SVD, title/abstract vectors and NRMS title tokens.
2. `3_02_entity_embeddings.ipynb`: canonical training entity lookup, pooled article vectors and coverage.
3. `3_03_user_profiles.ipynb`: article availability, supplied-history profiles and causal profile examples.

Select the project's `.venv` kernel and run top to bottom. Core methods are actual notebook code: training-only
TF-IDF/SVD and vocabularies, entity pooling, article discovery, supplied-history pooling and causal profile updates.
Build calls pass these notebook functions to shared file/cache/audit helpers in `src/newsrec/`.
No builder or separate Python run is needed.
Defaults are in `src/newsrec/common.py`; override them with `Settings(...)` in a notebook if needed.
`DATA_ROOT` can point to another folder holding the saved datasets.

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
