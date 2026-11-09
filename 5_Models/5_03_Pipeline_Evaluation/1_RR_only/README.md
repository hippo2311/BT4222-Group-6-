# 1. Reranker-only evaluation (logged MIND slates)

[PE_RR_logged_slates.ipynb](PE_RR_logged_slates.ipynb) reorders the exact slates MIND logged and
compares three rerankers on the test split:

| Model | Type |
|---|---|
| `popularity_smoothed_prior_article_ctr` | Rule baseline: causal smoothed article CTR |
| `content_entity_cosine` | Rule baseline: entity cosine to the causal click profile |
| `attention_reranker` | RR4 `+metadata`, `softmax+lambda_ndcg`, trained in the notebook |

The notebook:

1. checks every input exists;
2. inspects the slates, click histories and article vectors;
3. loads the trained RR4 reranker (`TRAIN_RERANKER = False`, the default), or trains it the same way
   as stage 3 of `RR_4_branch_interaction.ipynb` (`TRAIN_RERANKER = True`);
4. scores all models with `rr_evaluation.py`;
5. compares RR4 with the metrics the RR_4 notebook recorded;
6. breaks the results down by the user's prior click count;
7. measures how much RR4's ranking depends on its tabular features: permutation importance per
   feature group and per single feature (test nDCG@10 lost when it is shuffled), and SHAP values;
8. runs model diagnostics: the training curve, the validation-to-test gap (RR4 and popularity
   are also scored on the other split), nDCG@10 by slate size, where new and older clicked articles
   are ranked, and what the clicks RR4 misses have in common.

It runs on Colab or locally without edits; see the [5_03 README](../README.md#on-google-colab).

## Inputs (under the project folder)

- `4_Feature_Engineering/artifacts/prequential-all/`: `feature_schema.json`, and
  `ranking_features_*`, `labels_*` and `request_profiles_*` for `train`, `validation` and `test`
  (section 9.2 scores the split that is not the target; it is skipped if that split is missing)
- `3_Representations/artifacts/`: `article_index.parquet`, `text64.npy`, `article_entity100.npy`
- `5_Models/5_02_Reranking/rr_evaluation.py`

## Outputs

- With `TRAIN_RERANKER = True`: `model.pt` and `preprocessor.joblib` in the save path
  (`submission/3.Model_Artifacts/Train_Reranker` on Colab; locally
  `5_Models/5_03_Pipeline_Evaluation/artifacts/reranker_logged_slates/reranker/`)
- `5_Models/5_03_Pipeline_Evaluation/artifacts/reranker_logged_slates/scores/<model>_<split>_scores.parquet`
- `results/rr_only_logged_slates_<split>.csv`, `rr_only_logged_slates_<split>_by_history.csv`
  and `rr_only_logged_slates_<split>.json`
- `results/rr_only_feature_permutation_<split>.csv` (one row per feature group),
  `results/rr_only_feature_permutation_single_<split>.csv` (one row per tabular feature) and
  `results/rr_only_feature_shap_<split>.csv` (one row per tabular feature)
- `results/rr_only_diagnostics_<split>_<table>.csv`: one file per diagnostic table
