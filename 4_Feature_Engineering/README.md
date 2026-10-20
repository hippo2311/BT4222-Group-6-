# 4. Ranking feature engineering

| Notebook | What it does |
|---|---|
| [4_01_build_ranking_features.ipynb](4_01_build_ranking_features.ipynb) | Selects model inputs, defines causal profile/feature calculations, replays chronological logged events, samples training negatives, and writes/audits ranking datasets for baselines and re-rankers. |

The notebook covers content, history, user–article matching, repeat exposure, popularity, observed freshness, multi-interest, supported half-life and request context. Slate-relative context is optional. Editable `ENABLED_GROUPS`, `DISABLED_FEATURES` and `EXTRA_FEATURES` determine the saved model allowlist.

Requests at the same timestamp are scored before feedback updates. Pooled vectors use full causal history; sequence inputs are capped. Sampling affects written training examples, while validation/test slates remain complete. Important calculations and worked examples are visible in the notebook; shared helpers handle checks, replay and storage.

| Output under `artifacts/<run>/` | Content |
|---|---|
| `ranking_features_<split>/` | Partitioned feature tables and diagnostics |
| `labels_<split>/` | Separate labels keyed by impression and article |
| `request_profiles_<split>/` | Full history rows, capped sequence rows/masks, recent rows and interest references |
| `training_negative_groups/` | Reproducible per-positive exposed-negative draws |
| `feature_schema.json` | Selected `model_columns`, `categorical_columns`, definitions and exclusions |
| Coverage, quality, sampling and cost reports | Representation fallback, feature ranges, sampling retention and measured resource use |
| Manifest and run status | Input/output provenance and completion state |

The full prequential run is `artifacts/prequential-all/`; smoke runs use separate folders. Model consumers must use the schema's allowlist, encode category IDs appropriately, and join labels on both keys. `newsrec.io.open_feature_run` validates/opens datasets, and `newsrec.ranking` exposes the notebook methods to Python consumers.

Execution and rebuild instructions are in the [project README](../README.md). Method comparisons, validation results and the current approach are in [journal.md](../journal.md).
