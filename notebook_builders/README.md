# Notebook builders

These optional scripts generate the preparation/EDA notebooks. Running a builder overwrites its target notebook; normal execution uses the existing notebooks directly.

| Script | What it generates or checks |
|---|---|
| `build_prep.py` | `1_Data_Preparation/clean_and_validate_data.ipynb` |
| `build_impression.py` | `2_EDA/impression_analysis.ipynb` |
| `build_user.py` | `2_EDA/user_analysis.ipynb` |
| `build_article.py` | `2_EDA/article_temporal_analysis.ipynb` |
| `eda_common.py` | Shared EDA notebook-generation helpers |
| `verify_split.py` | Standalone split-verification checks |

Stages 3/4 are edited directly in their notebooks and have no builders. See the [project README](../README.md) for the execution workflow.
