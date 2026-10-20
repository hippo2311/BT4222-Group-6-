# 1. Data preparation

| Notebook | What it does |
|---|---|
| [clean_and_validate_data.ipynb](clean_and_validate_data.ipynb) | Loads local split inputs or downloads missing files from the team's Drive; validates histories, candidate labels, keys, news fields, entity coverage and temporal integrity; writes typed processed datasets. |

The notebook preserves valid impressions, empty histories, repeat exposures and missing article fields. It creates stable cross-split impression keys and integer user/article codes, and sorts event data chronologically.

| Output | Content |
|---|---|
| `data_quality_report.csv` | Check results and diagnostic details |
| `processed/users.parquet` | User IDs and integer codes |
| `processed/impressions_<split>.parquet` | Requests, timestamps, supplied histories and slate summaries |
| `processed/candidates_<split>.parquet` | Logged impression/article pairs, shuffled position and click labels |
| `processed/news_catalogue.parquet` | Article metadata, split membership and descriptive exposure statistics |

Catalogue exposure summaries describe their recorded windows; time-dependent model features must instead use the past-only replay. Logged `position` is shuffled file order, not a display-rank feature.

Execution instructions and input-download recovery are in the [project README](../README.md).
