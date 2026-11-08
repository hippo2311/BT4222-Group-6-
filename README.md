# BT4222 Group 6 — News Recommendation

**Submission folder (data, models and code): <https://drive.google.com/drive/folders/1jf5KcEc1p39kkPRNLugL7xH7E4TNgBbI>**

| Folder in `submission/` | Contents |
|---|---|
| `1.MIND_split/` | Shortcut to the MIND-small `train/`, `validation/` and `test/` splits produced by [stage 1](1_Data_Preparation/README.md) |
| `2.Feature_Engineering/` | The prepared model-ready data as zips: `processed.zip` (stage 1), `representations.zip` (stage 3), `ranking_features.zip` (stage 4) and `bundle_manifest.json` with their checksums |
| `3.Model_Artifacts/MSFT_Reranker/` | The trained RR4 attention reranker (`model.pt`, `preprocessor.joblib`) used by the reranker-only evaluation |
| `3.Model_Artifacts/E2E/` | Checkpoints for the final end-to-end system (`candidate_generator/model.pt`, `reranker/model.pt`, `reranker/preprocessor.joblib`); when empty, notebook 4 trains them |
| `4.Code/` | This repository |

This project builds and evaluates a news recommendation pipeline using **MIND-small**. We have already prepared the data, article and user representations, and ranking features, so you can start with the models in stage 5.

## Start here: run the final notebooks on Google Colab

The `submission` folder holds the code **and** the prepared data, so nothing has to be downloaded or rebuilt. Stages 1–4 have already been run.

1. Open the submission link above. In Google Drive, right-click the `submission` folder, then choose **Organise > Add shortcut > My Drive**. Editor access is needed, because the notebooks unpack the data and save their models and results into `4.Code`.
2. Open each notebook below from `submission/4.Code` with **Open with > Google Colaboratory**. For notebooks 1 and 4, choose **Runtime > Change runtime type > T4 GPU**.
3. Run each one with **Runtime > Run all**, in this order. The notebooks detect Colab and find the folders automatically, so no paths need editing.

On the first run, a notebook unpacks the zips in `2.Feature_Engineering` into `4.Code` (a few minutes); later runs reuse them. Notebooks 1 and 4 copy the checkpoints in `3.Model_Artifacts` into their model folder before deciding whether to train, so the submitted files are never overwritten.

Each notebook has one **Configure** cell at the top (section 0), and it is the only cell meant to be edited. It sets where to run (`RUN_ENV`: `"auto"`, `"colab"` or `"local"`), the project folder, the data path (`COLAB_DATA_PATH` / `LOCAL_DATA_PATH`), the model path (`MODEL_PATH`) and, for notebooks 1 and 4, the submitted checkpoints (`SUBMITTED_MODEL_PATH`). With the defaults, Colab uses `/content/drive/MyDrive/submission/4.Code` for code, data and outputs. By default it also evaluates the test split, trains on the whole training set and evaluates every test impression. Setting `TRAIN_IMPRESSIONS` or `EVAL_IMPRESSIONS` to a number such as `5_000` gives a quicker run on a sample. `REUSE_SAVED_MODELS = True` loads a model saved by an earlier run when its settings match.

| # | Notebook | Purpose |
|---|---|---|
| 1 | [5_Models/5_03_Pipeline_Evaluation/1_RR_only/PE_RR_logged_slates.ipynb](5_Models/5_03_Pipeline_Evaluation/1_RR_only/PE_RR_logged_slates.ipynb) | Reranker only: rule baselines vs the trained RR4 attention reranker on the logged MIND slates |
| 2 | [5_Models/5_03_Pipeline_Evaluation/2_CG_RR_E2E/PE_popularity_reranker_baseline_pipeline.ipynb](5_Models/5_03_Pipeline_Evaluation/2_CG_RR_E2E/PE_popularity_reranker_baseline_pipeline.ipynb) | End to end: baseline generator + popularity reranker |
| 3 | [5_Models/5_03_Pipeline_Evaluation/2_CG_RR_E2E/PE_content_reranker_baseline_pipeline.ipynb](5_Models/5_03_Pipeline_Evaluation/2_CG_RR_E2E/PE_content_reranker_baseline_pipeline.ipynb) | End to end: baseline generator + content-similarity reranker |
| 4 | [5_Models/5_03_Pipeline_Evaluation/2_CG_RR_E2E/PE_attention_reranker_niche_pipeline.ipynb](5_Models/5_03_Pipeline_Evaluation/2_CG_RR_E2E/PE_attention_reranker_niche_pipeline.ipynb) | End to end, final system: niche two-tower generator + RR4 attention reranker |
| 5 | [5_Models/5_03_Pipeline_Evaluation/results.ipynb](5_Models/5_03_Pipeline_Evaluation/results.ipynb) | Comparison tables and plots across all of the above |

The other notebooks in stages 1–5 are the development work behind these (data preparation, EDA, representations, features, and model selection in `5_01` and `5_02`). They don't need to be run. [5_03_Pipeline_Evaluation/README.md](5_Models/5_03_Pipeline_Evaluation/README.md) explains what each final notebook measures.

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

The folder must contain `bundle_manifest.json`, `processed.zip`, `representations.zip`, and `ranking_features.zip`; the submission folder's `2.Feature_Engineering/` is exactly this, so download it and pass its path to `--local-package`. The bundle does not contain the raw split folders.

Then start with [candidate generation](5_Models/5_01_Candidate_Generation/README.md), continue to [reranking](5_Models/5_02_Reranking/README.md), and use [pipeline evaluation](5_Models/5_03_Pipeline_Evaluation/README.md) to evaluate the combined system. Each stage reads its data root from a different environment variable, and all of them default to the checkout: `NEWSREC_ROOT` for stages 3–4 and candidate generation, `DATA_ROOT` for reranking, and `NEWSREC_DATA_ROOT` (or `LOCAL_DATA_PATH` in the Configure cell) for pipeline evaluation. Note that the pipeline-evaluation notebooks ship with `LOCAL_DATA_PATH` set to one team member's folder; set it to `None` or your own path when running locally. The feature schema lists the model input columns; see the [feature README](4_Feature_Engineering/README.md) for training and label details.

## Reproduce the artifacts

1. Run the [data preparation notebook](<1_Data_Preparation/1. data_preparation.ipynb>) in Colab. Place its local `train/`, `validation/`, `test/`, and `1_Data_Preparation/processed/` outputs under the same data root. The [shared prepared splits](https://drive.google.com/drive/folders/1UAEC9VLoR5-HLciNpWBSwLahOZgthnsq) are also available, but the Drive upload layout alone does not recreate the local `processed/` directory.
2. Set `NEWSREC_ROOT` to that data root, then run the three [representation notebooks](3_Representations/README.md) in order.
3. Run the [feature engineering notebook](4_Feature_Engineering/README.md), setting `FULL_RUN = True` for the complete dataset. Outputs are written under the same data root.

For example, when the data root is the checkout, launch Jupyter after `export NEWSREC_ROOT="$PWD"`. The preparation notebook requires Colab. `run_all.py` still lists the retired MIND-large notebooks (`0_Data_Acquisition/download_split_process.ipynb` and `1_Data_Preparation/clean_and_validate_data.ipynb`) for stages 0–1, so use it only for stages 2–4 (`python run_all.py --from-stage 2 --to-stage 4`). The MIND-small test split is the final day of `MINDsmall_train`, not the official dev release. Large data and generated artifacts are excluded from Git.
