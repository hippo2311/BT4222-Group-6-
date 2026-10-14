# BT4222 Group 6 — Personalised news recommendation on Microsoft MIND

Full-catalogue, two-stage recommender (candidate generation Top-M, re-ranking Top-K) built on MIND-large.
Interim report: 4 Oct 2026. This repository follows the task plan of the final report.

## Layout

| Folder | Task | Owner | Status |
|---|---|---|---|
| `0_Data_Acquisition/` | 1. data acquisition and temporal split | Member 3 | done |
| `1_Data_Preparation/` | 2. data cleaning and validation (`clean_and_validate_data.ipynb`, `data_quality_report.csv`) | Member 3 | done |
| `2_EDA/` | 3. exploratory data analysis (`impression_analysis`, `user_analysis`, `article_temporal_analysis`, `figures/`, `tables/`) | Member 3 | done |
| `3_Representations/` | 4. article and user representations | | to come |
| `4_Basic_Models/` | 5. baseline models | | to come |
| `5_Candidate_Generation/` | 6. candidate generation (Recall@M) | | to come |
| `6_Feature_Engineering/`, `7_Re_Ranking/` | 7. re-ranking | | to come |
| `8_Advanced_Features/`, `9_Analysis/` | 8. advanced features and ablation | | to come |
| `10_End_to_End/` | 9. end-to-end evaluation and documentation | | to come |
| `notebook_builders/` | scripts that generate the notebooks of tasks 2 and 3 | Member 3 | |

## Data setup (not in git)

Place the agreed split folders at the repository root, exactly as produced by `0_Data_Acquisition/split_process.ipynb`:

```
train/        behaviors_train.parquet, news_train.parquet, entity_embedding.vec, relation_embedding.vec
validation/   behaviors_validation.parquet, news_validation.parquet
test/         behaviors_test.parquet, news_test.parquet, entity_embedding.vec, relation_embedding.vec
```

They are gitignored (about 1 GB). The notebooks also find them on Google Drive under `MyDrive/BT4222` when run in Colab.

## Run order

```
jupyter nbconvert --to notebook --execute --inplace 1_Data_Preparation/clean_and_validate_data.ipynb   # ~1 min, writes 1_Data_Preparation/processed/
jupyter nbconvert --to notebook --execute --inplace 2_EDA/impression_analysis.ipynb                     # 1-3 min each
jupyter nbconvert --to notebook --execute --inplace 2_EDA/user_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace 2_EDA/article_temporal_analysis.ipynb
```

`1_Data_Preparation/processed/` (typed impressions, exploded candidates, news catalogue with per-split exposure facts,
user codes; 456 MB) is the shared input for every later task and is regenerated rather than committed.
Every number quoted in the report comes from `2_EDA/manifest_*.json` or `2_EDA/tables/`.

Requirements: Python 3.12, polars >= 1.30, pandas, numpy, pyarrow, scipy, matplotlib, jupyter. 16 GB RAM is enough.
