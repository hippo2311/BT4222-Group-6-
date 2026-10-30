# Notebook builders

These optional scripts generate the older MIND-large preparation and EDA notebooks. `build_prep.py` recreates the retired `clean_and_validate_data.ipynb`; do not run it for the current MIND-small preparation notebook. Running any builder overwrites its target notebook.

| Script | What it generates or checks |
|---|---|
| `build_prep.py` | `1_Data_Preparation/clean_and_validate_data.ipynb` |
| `build_impression.py` | `2_EDA/impression_analysis.ipynb` |
| `build_user.py` | `2_EDA/user_analysis.ipynb` |
| `build_article.py` | `2_EDA/article_temporal_analysis.ipynb` |
| `eda_common.py` | Shared EDA notebook-generation helpers |
| `verify_split.py` | Standalone split-verification checks |

Stages 3/4 are edited directly in their notebooks and have no builders. See the [project README](../README.md) for the execution workflow.
