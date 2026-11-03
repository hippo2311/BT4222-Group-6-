# 4. Ranking feature engineering

| Notebook | What it does |
|---|---|
| [4_01_build_ranking_features.ipynb](4_01_build_ranking_features.ipynb) | Selects model inputs, defines causal profile/feature calculations, replays chronological logged events, defines epoch negative sampling, and writes/audits ranking datasets for baselines and re-rankers. |

The notebook covers content, history, user–article matching, repeat exposure, popularity, observed freshness, multi-interest, supported half-life and request context. Slate-relative context is optional. Editable `ENABLED_GROUPS`, `DISABLED_FEATURES` and `EXTRA_FEATURES` determine the saved model allowlist.

Requests at the same timestamp are scored before feedback updates. Pooled vectors use full causal history; sequence inputs are capped. By default, `Settings(training_negative_sampling="epoch")` saves all training candidates. The training iterator draws up to four exposed negatives per positive from the same impression afresh each epoch. Validation/test slates remain complete. `Settings(training_negative_sampling="fixed")` restores the previous fixed sample. Important calculations and worked examples are visible in the notebook; shared helpers handle checks, replay and storage.

| Output under `artifacts/<run>/` | Content |
|---|---|
| `ranking_features_<split>/` | Partitioned feature tables and diagnostics |
| `labels_<split>/` | Separate labels keyed by impression and article |
| `request_profiles_<split>/` | Full history rows, capped sequence rows/masks, recent rows and interest references |
| `training_negative_groups/` | Epoch-zero reference draw; dynamic training regenerates groups each epoch |
| `feature_schema.json` | Selected `model_columns`, `categorical_columns`, definitions and exclusions |
| Coverage, quality, sampling and cost reports | Representation fallback, feature ranges, sampling retention and measured resource use |
| Manifest and run status | Input/output provenance and completion state |

The full prequential run is `artifacts/prequential-all/`; smoke runs use separate folders. Model consumers must use the schema's allowlist, encode category IDs appropriately, and join labels on both keys. `newsrec.io.open_feature_run` validates/opens datasets, and `newsrec.ranking` exposes the notebook methods to Python consumers.

Execution and rebuild instructions are in the [project README](../README.md). Method comparisons, validation results and the current approach are in [journal.md](../journal.md).

Rebuild old fixed-sample features after preparing the intended MIND-small splits:

```bash
python run_all.py --from-stage 4 --to-stage 4 --force
```

Existing sampled files cannot supply discarded negatives. The new settings fingerprint requires a rebuild; representation embeddings do not need refitting solely for this sampling change. More training feature rows and disk use are expected.

An epoch-based ranker can consume the full pool with:

```python
from newsrec.io import open_feature_run
from newsrec.training import iter_training_epoch

opened = open_feature_run(ROOT, run_dir)
model_columns = opened["schema"]["model_columns"]
for epoch in range(num_epochs):
    for frame, groups in iter_training_epoch(run_dir, epoch):
        X, y = frame[model_columns], frame["clicked"]
        # Run your model's training step here.
        # groups contains positive_news_code and negative_news_codes per impression.
```

Each yielded frame contains whole sampled impressions, all positives, and the unique union of selected negatives. Grouped losses should use the returned memberships: negatives shared by positives appear once in the frame. The same seed/epoch reproduces the draw; subsequent epochs draw afresh and can overlap. The iterator reads training partitions in bounded chunks, including slates crossing file boundaries. It does not fit preprocessing or shuffle batches. Select only model columns for the model input and fit preprocessing on training data.

No epoch-based ranker exists in this repository yet; this implements its sampling/data-loading interface. The logistic sanity check still uses a bounded row sample, and standard LightGBM training needs a separate fitting strategy. Saved groups are an epoch-zero reference, not the full negative pool. Calibration and negative coverage are not guaranteed by resampling.
