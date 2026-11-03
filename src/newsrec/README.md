# src/newsrec — shared pipeline package

The notebooks in stages 3/4 own the **calculations** (visible in their cells); this package
owns the **plumbing** around them: chronological replay, storage, cache validation,
reporting and diagnostics. Notebooks import from here; nothing in here is run separately.

The single source of truth for feature calculations is
`4_Feature_Engineering/4_01_build_ranking_features.ipynb` — `ranking.py` loads its tagged
cells, so the terminal runner (`run_all.py`) and the tests use exactly the code reviewers
see in the notebook.

## Files

| File | What it does |
|---|---|
| `common.py` | Configuration and shared primitives. The frozen `Settings` dataclass holds every tunable (embedding dims, TF-IDF params, `history_cap=50`, recent window 24 h / 10 items, `interest_count=3`, `ctr_strength=20`, `negative_count=4`, `training_negative_sampling="epoch"` (full saved training pool; `"fixed"` restores static sampling), `update_mode` prequential/frozen, `context_features`, batch sizes) with range validation and `fingerprint()` hashes used to detect stale cached outputs. Also: `normalise`/`normalised_mean` (L2 pooling), `sha256`/`write_json`, manifest/cost/report writers, `require_inputs`/`saved_stage_ready` (reuse-or-rebuild gate), `prepare_feature_pairs` (keyed-input contract: unique keys, single timestamp, eligibility), and `validate_feature_schema` (rejects duplicates, empty selections, model columns colliding with protected keys/targets/diagnostics). |
| `temporal.py` | The causal, no-leakage state machine. `PairStore` is a disk-backed SQLite store of per-(user, article) exposure/click history. `User` holds one user's supplied-history snapshot, capped deque of observed clicks, category counters and version. `TemporalState` is the replay state: `begin(t)` seeds/validates a timestamp batch, `profile(user, t)` builds full causal pooled text/entity and category-interest vectors, supplied-only/recent views, and capped sequence references, `half_life(rows, t)` estimates past-only decay from completed 6-hour windows, and `finish(t, events)` applies feedback **after** scoring. Also holds the article-level arrays (exposures, clicks, first-seen, hourly CTR) behind popularity, freshness and half-life features. |
| `representations.py` | Builds and loads the stage-3 artifacts. Text space: training-only TF-IDF + TruncatedSVD (`fit_text_space`), NRMS title tokens (`encode_title_tokens`), category encodings. Entities: lookup parsing and article pooling (`build_entities`). `build_news` assembles the article index and vector arrays; `build_availability` records earliest observable discovery time (not publication). `Bundle` is the read-only object the feature build consumes (index, vector arrays, quality/coverage, `rows()`/`lookup`). `representation_summary` and `text_neighbors` provide coverage reporting and label-free neighbour inspection. |
| `pipeline.py` | The replay engine and dataset build. `timestamp_groups` streams Parquet in bounded batches grouped by timestamp; `replay_batches` heap-merges the impression/candidate streams across all splits into one chronological timestamp at a time; `PartitionWriter` writes partitioned Parquet outputs; `time_horizon` gives the replay (start, end). `build_supplied_profiles`/`pool_supplied_history` produce supplied-history-only profiles for baselines/clustering. `build_ranking_dataset` is the main loop (begin → build features → write → finish per timestamp), wiring in the state factory, feature builder and negative sampler passed from the notebook. `verify_feature_parity` and `profile_preview` support the inference-parity and profile-inspection checks. |
| `io.py` | Saved-run entry points. `feature_run` is what the notebook calls: decides build-vs-reuse, runs `build_ranking_dataset`, writes manifest/status. `open_feature_run` validates a saved run (hashes, schema, completeness) and opens it as Arrow datasets without loading everything into RAM. `reconstruct_profile` rebuilds a causal profile from a saved `request_profiles` reference; `feature_sample` peeks at a few rows. |
| `audits.py` | Label-free representation diagnostics. `usage_coverage` scans article text/entity availability across slate entries, clicked entries and supplied-history entries in bounded batches (feeds the coverage reports); `view_neighbors` inspects nearest neighbours without click labels; `check_temporal_profiles` verifies causal profile behavior on a small state. |
| `sanity.py` | A bounded smoke check, not a baseline. `python -m newsrec.sanity` fits a logistic model on a bounded training sample and compares it with smoothed prior-CTR popularity on complete validation slates using impression AUC and NDCG@10 to confirm the pipeline produces rankable output. Explicitly not a tuned baseline, calibration study or leakage proof. |
| `ranking.py` | The notebook-as-source bridge. `load_ranking_methods()` parses `4_01_build_ranking_features.ipynb` and executes only the tagged cells (`feature_catalogue`, `feature_controls`, `method`, `feature_schema`) into a clean namespace, exposing `CausalProfiles`, `TemporalState`, `build_features`, `feature_schema`, `sample_training_groups`. Only tagged definitions are loaded — never setup, examples, outputs or dataset builds. Restart Python after editing the notebook. |
| `progress.py` | Display-only timing: `duration()` formatter and `ReplayProgress` for the elapsed-time updates during long cells. Never used to compute features. |
| `__init__.py` | Package docstring only. |

## Conventions

- **Causality:** state at timestamp `t` contains only events strictly before `t`; feedback is
  applied in `finish()` after all requests at `t` are scored.
- **Cache safety:** outputs are reused only when settings fingerprints, input hashes and
  schema versions match; incompatible files raise instead of silently mixing.
- **Memory:** all large reads/writes stream in bounded Parquet batches
  (`parquet_batch_rows` / `output_batch_rows`); the pair store is on disk.
- **Reports:** manifests, cost measurements and coverage/quality CSVs are written under
  `results/` and the stage artifact folders via `common.py` helpers.

`training.py` provides `iter_training_epoch(run_dir, epoch)`: streams full training features/labels and returns sampled frames plus per-positive groups. Pass a new epoch number each time for a fresh reproducible draw. Legacy sampled artifacts must be rebuilt. See the feature-engineering README for usage.
