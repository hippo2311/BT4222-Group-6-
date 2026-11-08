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
3. trains RR4 the same way as stage 3 of `RR_4_branch_interaction.ipynb`, or loads the saved
   checkpoint;
4. scores all models with `rr_evaluation.py`;
5. compares RR4 with the metrics the RR_4 notebook recorded;
6. breaks the results down by the user's prior click count.
7. measures how much RR4's ranking depends on its tabular features: permutation importance per
   feature group and per single feature (test nDCG@10 lost when it is shuffled), and SHAP values.

It runs on Colab or locally without edits; see the [5_03 README](../README.md#on-google-colab).

## Inputs (under the project folder)

- `4_Feature_Engineering/artifacts/prequential-all/`: `feature_schema.json`, and
  `ranking_features_*`, `labels_*` and `request_profiles_*` for `train` and the target split
- `3_Representations/artifacts/`: `article_index.parquet`, `text64.npy`, `article_entity100.npy`
- `5_Models/5_02_Reranking/rr_evaluation.py`

## Outputs

- `5_Models/5_03_Pipeline_Evaluation/artifacts/reranker_logged_slates/reranker/model.pt` and
  `preprocessor.joblib`, reused on the next run while the settings match
- `5_Models/5_03_Pipeline_Evaluation/artifacts/reranker_logged_slates/scores/<model>_<split>_scores.parquet`
- `results/rr_only_logged_slates_<split>.csv`, `rr_only_logged_slates_<split>_by_history.csv`
  and `rr_only_logged_slates_<split>.json`
- `results/rr_only_feature_permutation_<split>.csv` (one row per feature group),
  `results/rr_only_feature_permutation_single_<split>.csv` (one row per tabular feature) and
  `results/rr_only_feature_shap_<split>.csv` (one row per tabular feature)
