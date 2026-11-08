# Notebook builders

These optional scripts generate notebooks. The three EDA builders produce the current **MIND-small** EDA notebooks in `2_EDA/`; `build_prep.py` and `verify_split.py` are legacy **MIND-large** tools. Running any builder overwrites its target notebook.

| Script | What it generates or checks |
|---|---|
| `build_impression.py` | `2_EDA/impression_analysis.ipynb` (MIND-small) |
| `build_user.py` | `2_EDA/user_analysis.ipynb` (MIND-small) |
| `build_article.py` | `2_EDA/article_temporal_analysis.ipynb` (MIND-small) |
| `eda_common.py` | Shared EDA notebook-generation helpers |
| `build_prep.py` | Legacy: recreates the retired MIND-large `1_Data_Preparation/clean_and_validate_data.ipynb`. Do not run it for the current MIND-small preparation notebook. |
| `verify_split.py` | Legacy: standalone checks for the old MIND-large train/validation/test split |

Stages 3/4 are edited directly in their notebooks and have no builders. See the [project README](../README.md) for the execution workflow.
