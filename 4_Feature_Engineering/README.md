# 4. Ranking feature engineering

[4_01_build_ranking_features.ipynb](4_01_build_ranking_features.ipynb) combines the prepared MIND-small data and representations into ranking features, labels and user-history references. It builds features chronologically, scoring requests at the same timestamp before applying their feedback.

The completed full-dataset outputs are included in the model-ready bundle. Follow the [project README](../README.md) to download them and continue to stage 5.

## Rebuild

After preparing the data and running the three [representation notebooks](../3_Representations/README.md), set `NEWSREC_ROOT` to the data root and run the feature notebook. Set `FULL_RUN = True` in its first code cell to build the complete dataset; the default is a 1,000-timestamp smoke run. Set `REBUILD = True` only when intentionally replacing a saved run.

The full run is saved under `4_Feature_Engineering/artifacts/prequential-all/`:

| Output | Contents |
|---|---|
| `ranking_features_<split>/` | Feature tables for train, validation and test. |
| `labels_<split>/` | Click labels, joined to features by impression and article keys. |
| `request_profiles_<split>/` | History and interest references for each request. |
| `feature_schema.json` | Model input columns, categorical columns and feature definitions. |
| `training_negative_groups/` | Reference negative sample for epoch zero. |
| Reports and manifests | Coverage, quality, sampling, cost and provenance. |

## Use in a model

Use `model_columns` from the schema and encode categorical columns appropriately. The training output retains all candidates; `newsrec.training.iter_training_epoch(run_dir, epoch)` draws up to four exposed negatives per positive within the same impression each epoch. Its returned groups describe the positive–negative memberships. Validation and test keep complete slates. The [stage 5 READMEs](../README.md#project-structure) describe the models and evaluation.
