# 0. Data acquisition

| Notebook | What it does |
|---|---|
| [download_split_process.ipynb](download_split_process.ipynb) | Downloads the original MIND-large train and labelled dev archives, reuses cached archives/extracted files, and creates local chronological train/validation/test splits. |

The earliest 85% of train-release impressions form training; the remaining 15% form validation. The labelled dev release supplies test data. Existing complete splits are reused unless rebuilding is requested.

Outputs are `behaviors_<split>.parquet` and `news_<split>.parquet` in the root `train/`, `validation/` and `test/` folders. Train/test entity vectors stay alongside those inputs; validation uses training vectors. ZIP archives are cached in `downloads/`, with split provenance recorded in `split_manifest.json`.

Execution instructions are in the [project README](../README.md).
