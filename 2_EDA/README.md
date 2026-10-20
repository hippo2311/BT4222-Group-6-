# 2. Exploratory data analysis

These notebooks read the processed datasets from stage 1. Descriptive analyses use training data; held-out splits are used where cross-split cold start is measured.

| Notebook | What it does | Outputs |
|---|---|---|
| [impression_analysis.ipynb](impression_analysis.ipynb) | Analyses slate lengths, clicks per impression, logged position, exposure/label semantics, negative sampling, repeat exposures and slate composition. | `figures/imp_*.png`, supporting CSV tables and `manifest_impression_analysis.json` |
| [user_analysis.ipynb](user_analysis.ipynb) | Analyses activity, supplied histories, cold-start users, causal history augmentation, category affinity, novelty and short-/long-term interests. | `figures/usr_*.png`, supporting CSV tables and `manifest_user_analysis.json` |
| [article_temporal_analysis.ipynb](article_temporal_analysis.ipynb) | Analyses catalogue structure, popularity, category CTR, observed freshness/lifespan, cold-start articles, time of day and visit gaps. | `figures/art_*.png`, supporting CSV tables and `manifest_article_temporal_analysis.json` |

Each notebook runs independently after data preparation. Figure data is saved under `tables/`; manifests contain summary measurements. Results and their implications for modelling are recorded in [journal.md](../journal.md).

Execution instructions are in the [project README](../README.md).
