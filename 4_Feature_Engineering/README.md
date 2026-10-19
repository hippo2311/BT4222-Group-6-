# 4. Ranking feature engineering

Run `4_01_build_ranking_features.ipynb` after the three representation notebooks. Core calculations live in its cells:
causal profiles, past-only half-life, ranking-feature formulas and deterministic negative sampling. Build calls pass
these visible functions to shared storage/replay helpers. Generic key/eligibility/cache checks stay in `src/newsrec/`.
Edit method cells and rebuild affected outputs; do not run the Python helper files separately.

Feature definitions and selection now live entirely in the notebook; `src/newsrec/features.py` has been removed.
Edit `ENABLED_GROUPS`, `DISABLED_FEATURES`, and `EXTRA_FEATURES` near the top to choose or add model inputs.
Custom entries contain their explanation and `compute(frame, cfg)` calculation. Selected columns are passed to
the shared builder and saved schema; an incompatible cached schema is rejected. Set `REBUILD=True` after changing
selection/calculations. Unselected built-in values may remain in the wide output for diagnostics/ablations.

The notebook introduces the model contract by feature group, then pairs each method with a small executable
example. Synthetic teaching inputs stay in memory; the later coverage, value, sampling and cost tables use the
actual saved run. Formula extraction is checked against the existing implementation across replay timestamps.

The notebook defaults to a 1,000-timestamp chronological smoke run. Set `MAX_TIMESTAMPS=None` for a full run.
The terminal runner defaults to full features; `--features-smoke 1000` selects the smaller run.

`REBUILD=False` reuses an existing complete, compatible feature run. Set `REBUILD=True` after changing calculations
or inputs; `--overwrite-features` also rebuilds the selected run. Changed representation dependencies or settings
cause a validation error instead of silently mixing incompatible data.

Features include history/category affinity, text/entity matching, previous user/article exposures, smoothed
popularity, observed age, multi-interest matching and supported half-life. Request context is separately tagged;
median-relative slate advantage is opt-in. Historical state uses all earlier events, before training-negative sampling.
Same-second train/validation rows are scored before feedback updates. Logged validation/test slates remain complete.

Outputs under `4_Feature_Engineering/artifacts/<run>/`:

- `ranking_features_<split>/`: partitioned features; use only the schema's model columns.
- `labels_<split>/`: separate impression/article labels.
- `request_profiles_<split>/`: history rows, masks and interest/recent references for later models.
- `training_negative_groups/` and sampling summary: shared deterministic exposed-negative draws.
- Feature schema, coverage, quality, cost, manifest and completion status.

`src/newsrec/io.py` provides `open_feature_run` to validate saved runs and open Arrow datasets without loading everything
into RAM. Alternatively, read selected Parquet columns with pandas/Polars. All saved outputs survive VS Code restarts.
The SQLite pair store is temporary build state, removed after success; an interrupted build restarts from the beginning.

Age is not publication age. Half-life uses supported completed windows and validity flags; no hard expiry filter is
applied. Full MIND runtime/RAM still needs measurement; synthetic tests do not establish recommendation quality.

Default settings live in `src/newsrec/common.py`; notebook `Settings(...)` overrides are optional.
Restart the kernel after changing shared Python code, then rebuild affected outputs.
