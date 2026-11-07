# BT4222 Group 6 — News Recommendation

This project builds and evaluates a news recommendation pipeline using **MIND-small**. We have already prepared the data, article and user representations, and ranking features, so you can start with the models in stage 5.

## Project structure

| Stage | Purpose |
|---|---|
| [1_Data_Preparation](1_Data_Preparation/README.md) | Download, clean and split MIND-small. |
| [2_EDA](2_EDA/README.md) | Explore the data. |
| [3_Representations](3_Representations/README.md) | Build article vectors and user profiles. |
| [4_Feature_Engineering](4_Feature_Engineering/README.md) | Build the features and labels used for ranking. |
| [5_Models](5_Models/) | Develop [candidate generators](5_Models/5_01_Candidate_Generation/README.md), [rerankers](5_Models/5_02_Reranking/README.md), and [evaluate the pipeline](5_Models/5_03_Pipeline_Evaluation/README.md). |
| [src/newsrec](src/newsrec/README.md) | Shared pipeline and model utilities. |

## Get started with the existing artifacts

From the repository root, set up Python 3.12 and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The `model-ready` bundle contains the processed MIND-small data, representations, and completed ranking features. [artifact_bundles.json](artifact_bundles.json) pins the `v3-mindsmall` version, Google Drive folder, and manifest checksum. Download and verify it with:

```bash
python download_artifacts.py --bundle model-ready
python download_artifacts.py --bundle model-ready --verify-only
```

The script restores files into `1_Data_Preparation/processed/`, `3_Representations/artifacts/`, and `4_Feature_Engineering/artifacts/prequential-all/`. It checks hashes and refuses to overwrite conflicting files. The package was restored successfully from a local copy, but its upload to Drive and remote download have not been confirmed. If the download reports missing files or access errors, ask the bundle owner to upload or share the complete `v3-mindsmall` folder. If you have that folder locally, install it with:

```bash
python download_artifacts.py --bundle model-ready --local-package /path/to/v3-mindsmall
```

The folder must contain `bundle_manifest.json`, `processed.zip`, `representations.zip`, and `ranking_features.zip`. The bundle does not contain the raw split folders.

Then start with [candidate generation](5_Models/5_01_Candidate_Generation/README.md), continue to [reranking](5_Models/5_02_Reranking/README.md), and use [pipeline evaluation](5_Models/5_03_Pipeline_Evaluation/README.md) to evaluate the combined system. Check each notebook's data paths before running; the reranking notebooks in particular require `DATA_ROOT` and `PROJECT_ROOT` to be set for your machine. The feature schema lists the model input columns; see the [feature README](4_Feature_Engineering/README.md) for training and label details.

## Reproduce the artifacts

1. Run the [data preparation notebook](<1_Data_Preparation/1. data_preparation.ipynb>) in Colab. Place its local `train/`, `validation/`, `test/`, and `1_Data_Preparation/processed/` outputs under the same data root. The [shared prepared splits](https://drive.google.com/drive/folders/1UAEC9VLoR5-HLciNpWBSwLahOZgthnsq) are also available, but the Drive upload layout alone does not recreate the local `processed/` directory.
2. Set `NEWSREC_ROOT` to that data root, then run the three [representation notebooks](3_Representations/README.md) in order.
3. Run the [feature engineering notebook](4_Feature_Engineering/README.md), setting `FULL_RUN = True` for the complete dataset. Outputs are written under the same data root.

For example, when the data root is the checkout, launch Jupyter after `export NEWSREC_ROOT="$PWD"`. The preparation notebook requires Colab; the older stage-1 `run_all.py` flow targets a different preparation notebook. The MIND-small test split is the final day of `MINDsmall_train`, not the official dev release. Large data and generated artifacts are excluded from Git.
