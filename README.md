# BT4222 Group 6 — News Recommendation

This repository contains our MIND news recommendation work. The current data preparation notebook uses **MIND-small**. Some EDA, representation, and model notebooks were developed with **MIND-large**; rerun and check their results before reporting them for MIND-small.

## Prepare MIND-small

1. Open [1. data_preparation.ipynb](<1_Data_Preparation/1. data_preparation.ipynb>) in Google Colab.
2. Run the cells in order. The notebook downloads `MINDsmall_train` from Kaggle, inspects and parses it, assigns integer IDs, and splits impressions by date.
3. Run the final cell to upload the outputs to the shared Google Drive folder. It creates `Group6_MINDsmall/train`, `validation`, and `test`.

The `test` split is the last day of **MINDsmall_train**, not the official MIND-small dev release. See [data preparation details](1_Data_Preparation/README.md) for the split dates and file layout.

## Use the prepared data

The notebook writes local `train/`, `validation/`, and `test/` folders, plus `1_Data_Preparation/processed/`. The EDA and downstream code read the processed impression, candidate, user, and news tables. The split folders preserve the corresponding behavior, news, and embedding files.

| Stage | Location |
|---|---|
| Data preparation | [1_Data_Preparation](1_Data_Preparation/README.md) |
| Exploratory analysis | [2_EDA](2_EDA/README.md) |
| News and user representations | [3_Representations](3_Representations/README.md) |
| Ranking features | [4_Feature_Engineering](4_Feature_Engineering/README.md) |
| Candidate generation baselines | [5_Models/5_01_Candidate_Generation](5_Models/5_01_Candidate_Generation/README.md) |

## Local setup

Use Python 3.12 from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The preparation notebook imports `google.colab`, so run it in Colab. The older `run_all.py` and notebook builder for stage 1 still target the former MIND-large preparation flow; do not use them to regenerate the MIND-small notebook. Existing MIND-large model artifacts and their download instructions are documented in [ARTIFACTS.md](ARTIFACTS.md).

Raw and processed datasets are excluded from Git. Commit notebooks and code; share large data through the team's Drive folder.

## MIND Small data location

The shared prepared splits are in [Group6_MINDsmall](https://drive.google.com/drive/folders/1UAEC9VLoR5-HLciNpWBSwLahOZgthnsq). On this machine the prepared MIND-small data, representations and feature build live at the checkout root. EDA, representation, feature and candidate-generation notebooks default to the checkout root; set `NEWSREC_ROOT` to another prepared data root when needed. These defaults require the data to be present locally; they do not download the Drive folder.

`run_all.py` also uses the checkout root. The existing `model-ready` bundle configuration is for the older dataset and must not be used as a Small bundle until a new Small bundle has been published and pinned.
