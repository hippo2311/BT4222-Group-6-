# 3. Article and user representations

These notebooks turn the prepared MIND-small data into article vectors and user profiles for feature engineering and modeling. The completed outputs are included in the model-ready bundle; see the [project README](../README.md) to download them.

To rebuild, prepare the data first, set `NEWSREC_ROOT` to the directory containing `1_Data_Preparation/processed/` and the split folders, and run these notebooks in order:

| Notebook | Output |
|---|---|
| [3_01_news_embeddings.ipynb](3_01_news_embeddings.ipynb) | Article index, text vectors, categories and title tokens. Text transformations are fitted on training data. |
| [3_02_entity_embeddings.ipynb](3_02_entity_embeddings.ipynb) | Article entity vectors, mappings and coverage reports. |
| [3_03_user_profiles.ipynb](3_03_user_profiles.ipynb) | Article availability and supplied-history profiles, with causal profile examples. |

Outputs are saved to `3_Representations/artifacts/` under the data root. The notebooks can also use their `DATA_ROOT` setting; explicitly setting the root avoids their local fallback path. Set `REBUILD = True` when you deliberately need to replace cached outputs. Article arrays use the same row mapping, with row 0 reserved for padding. Continue to [feature engineering](../4_Feature_Engineering/README.md).
