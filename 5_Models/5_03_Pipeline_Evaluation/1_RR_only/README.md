# 1. Reranker-only evaluation (logged MIND slates)

[PE_RR_logged_slates.ipynb](PE_RR_logged_slates.ipynb) reorders the exact slates MIND logged
and compares three rerankers on validation and test:

| Model | Type |
|---|---|
| `popularity_smoothed_prior_article_ctr` | Rule baseline (causal smoothed article CTR) |
| `content_entity_cosine` | Rule baseline (entity cosine to the causal click profile) |
| `branch_plus_metadata__softmax_plus_lambda_ndcg__full_data` | Selected RR_4 model, loaded from its checkpoint |

Nothing is trained. The notebook checks every input exists, inspects the slates, histories and
article vectors, scores all models with `rr_evaluation.py`, verifies the selected model
reproduces its RR_4 metrics, and breaks results down by the user's prior click count.

## Inputs (under `DATA_ROOT`)

Set `ENVIRONMENT = "colab"` to read these from `MyDrive/v3-mindsmall`; the layout is the same
as local.

- `4_Feature_Engineering/artifacts/prequential-all/`: `feature_schema.json`,
  `ranking_features_{validation,test}/`, `labels_{validation,test}/`,
  `request_profiles_{validation,test}/`
- `3_Representations/artifacts/`: `article_index.parquet`, `text64.npy`, `article_entity100.npy`
- `5_Models/5_02_Reranking/artifacts/branch_interaction/`: `preprocessor.joblib`,
  `branch_plus_metadata__softmax_plus_lambda_ndcg__full_data/model.pt`

The repository (`PROJECT_ROOT`) supplies `5_Models/5_02_Reranking/rr_evaluation.py`.

## Outputs

- Score files: `DATA_ROOT/5_Models/5_03_Pipeline_Evaluation/artifacts/1_RR_only/<model>/<split>_scores.parquet`
- Metrics: `DATA_ROOT/results/pe_rr_only_overall.csv`, `pe_rr_only_by_history.csv`,
  `pe_rr_only_metadata.json`

A full local run takes under two minutes on an M-series Mac (MPS).
