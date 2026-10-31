# 1. Data preparation — MIND-small

Run [1. data_preparation.ipynb](<1. data_preparation.ipynb>) from top to bottom in Google Colab. It downloads `MINDsmall_train` from the `arashnic/mind-news-dataset` Kaggle dataset.

## What it does

- Inspects columns, missing values, duplicates, impression labels, article references, and data types.
- Parses reading histories and candidate click labels while keeping the original text. Missing histories become empty lists; missing abstracts stay missing.
- Converts `N123` to `news_code=123` and assigns stable `user_code` values, reserving `0` for padding or unknown users.
- Splits impressions by timestamp: **train** before 13 Nov 2019, **validation** on 13 Nov, and **test** from 14 Nov onward. All three come from `MINDsmall_train`; the test folder is not the official dev release.
- Saves one row per impression and one row per shown candidate for EDA and modelling.

## Outputs

| Local path | Content |
|---|---|
| `train/`, `validation/`, `test/` | Split behavior and news Parquet files, plus one-row-per-candidate `behaviors_<split>_exploded.parquet` |
| `train/`, `test/` | Entity and relation embedding files |
| `1_Data_Preparation/processed/` | `users.parquet`, `news_catalogue.parquet`, and `impressions_<split>.parquet` / `candidates_<split>.parquet` |
| `1_Data_Preparation/mappings/` | JSON maps from original user and news IDs to integer codes |
| `0_Data_Acquisition/split_manifest.json` | Dataset source, split rule, and row counts |

The final Colab cell uploads these files to `Group6_MINDsmall` on the shared Drive. Its top level contains only `train/`, `validation/`, and `test/`; shared files go inside `train/`. Locally, `processed/` stays in place because the existing EDA and model code reads that path.

The notebook retains basic validation and checks saved candidate row and click totals. It does not produce the old MIND-large `data_quality_report.csv`.
