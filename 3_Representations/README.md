# 3. Article and user representations

Run these notebooks in order after data preparation.

| Notebook | What it does | Main outputs in `artifacts/` |
|---|---|---|
| [3_01_news_embeddings.ipynb](3_01_news_embeddings.ipynb) | Fits training-only TF-IDF/SVD and frozen title/abstract transformations; builds fused article text vectors, category mappings and title-token inputs; reports vocabulary coverage and neighbours. | `article_index.parquet`, `title64.npy`, `abstract64.npy`, `text64.npy`, title-token IDs/masks/vocabulary, text diagnostics and manifest |
| [3_02_entity_embeddings.ipynb](3_02_entity_embeddings.ipynb) | Parses article entities, pools valid canonical pretrained entity vectors, records missing-vector fallbacks, and measures article/event-weighted coverage. | `article_entity100.npy`, entity mappings/links/quality, `article_fallback.parquet`, `usage_coverage.csv` and manifest |
| [3_03_user_profiles.ipynb](3_03_user_profiles.ipynb) | Establishes observation-time article eligibility, pools supplied histories, and demonstrates causal text/entity/recent and category-interest profiles using earlier feedback. | `article_availability.parquet`, `supplied_*_profiles.npy`, supplied-profile index, manifests and profile checks |

All arrays share the article-row mapping; row 0 is padding. Text and entity vectors use separate spaces. Eligibility is based on observable discovery, not publication time. Static supplied profiles contain no logged feedback; causal pooled profiles use full prior history, with capped sequence references and a separate recent view.

Execution, cache controls and data locations are described in the [project README](../README.md).
