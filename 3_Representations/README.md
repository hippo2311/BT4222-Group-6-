# 3. Article and user representations

Run in order after data preparation:

1. `3_01_news_embeddings.ipynb`: training-only TF-IDF/SVD, title/abstract vectors and NRMS title tokens.
2. `3_02_entity_embeddings.ipynb`: canonical training entity lookup, pooled article vectors and coverage.
3. `3_03_user_profiles.ipynb`: article availability, supplied-history profiles and causal profile examples.

Select the project's `.venv` kernel and run top to bottom. The short notebooks show settings, execution and results;
shared algorithms/checks live in `src/newsrec/`. Imports run automatically; no builder or separate Python run is needed.
Defaults are in `src/newsrec/common.py`; override them with `Settings(...)` in a notebook if needed.
`DATA_ROOT` can point to another folder holding the saved datasets.

`REBUILD=False` loads saved outputs after checking settings, inputs and output hashes. Missing outputs are built.
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

Causal profiles use supplied history once plus earlier logged clicks. Current/same-second feedback is excluded.
Ranking histories are capped; recent profiles use timestamped clicks only. Stage 4 writes per-request history
references for retrieval/rankers. Entity and text spaces remain separate; missing vectors retain fallback flags.

Coverage and cost reports are written under `results/`. Saved files survive closing VS Code; Python variables do not.
Later notebooks can load these files with `pd.read_parquet(...)` or `np.load(...)` without rebuilding representations.

Edit representation logic in `src/newsrec/representations.py`, user-state logic in `temporal.py`, and orchestration
in `pipeline.py`. Restart the kernel after editing imported code, then rebuild affected outputs.
