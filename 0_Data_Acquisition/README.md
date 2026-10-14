# 0. Data acquisition and temporal split

BT4222 Group 6 · final project · task 1. Owner: Member 3 (Anh Minh Hoang). Split notebook written by the team on 27 Sep 2026.

## Source data

| Release | Files | Rows | Where it came from |
|---|---|---|---|
| MIND-large **train** | `behaviors.tsv`, `news.tsv`, `entity_embedding.vec`, `relation_embedding.vec` | 2,232,748 impressions, 101,527 articles | https://msnews.github.io/ (MINDlarge_train.zip) |
| MIND-large **dev** | same four files | 376,471 impressions, 72,023 articles | https://msnews.github.io/ (MINDlarge_dev.zip) |

The MIND-large test release has no labels and is not used. Download links also sit in `load_mind_dataset.ipynb` (project root).
Raw TSV files have no header. Column order: `behaviors.tsv` = `impression_id, user_id, time, history, impressions`;
`news.tsv` = `news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities`.

## Split procedure (`split_process.ipynb`)

1. Read the train release's `behaviors.tsv`, parse `time` with the format `%m/%d/%Y %I:%M:%S %p`, sort by time.
2. Cut at row `int(0.85 * 2,232,748) = 1,897,835`: rows before the cut are **train**, the rest **validation**.
3. For each of the two parts, keep the `news.tsv` rows referenced by that part's histories or impression slates
   (`news_train.parquet`, `news_validation.parquet`).
4. Read the dev release as **test** (`behaviors_test.parquet`), and its own `news.tsv` as `news_test.parquet`.

Resulting folders (project root):

| Folder | Behaviours | Window (log clock) | News rows |
|---|---|---|---|
| `train/` | 1,897,835 impressions, 667,880 users | 2019-11-09 00:00:00 to 2019-11-14 07:30:28 | 98,671 |
| `validation/` | 334,913 impressions, 235,963 users | 2019-11-14 07:30:28 to 2019-11-14 23:59:59 | 71,991 |
| `test/` | 376,471 impressions, 255,990 users | 2019-11-15 00:00:00 to 2019-11-15 23:59:43 | 72,023 |

`train/` also holds `behaviors_train_exploded.parquet` (one row per candidate; its `recommendation_position` is the
shuffled file order, not a display rank). The entity and relation embedding files sit in `train/` (train release) and
`test/` (dev release); the validation split shares the train release's files.

## Verification (see `1_Data_Preparation/clean_and_validate_data.ipynb`, section 1)

* Row counts match the arithmetic above exactly; train + validation = the full train release.
* Windows are ordered: last train timestamp <= first validation timestamp < first test timestamp.
* The cut falls inside one second (07:30:28 on 14 Nov): 4 same-second rows are in train and 9 in validation.
  `polars.sort()` is not stable by default, so re-running the notebook may swap these 13 rows. The saved parquet files
  are the reference copy; a stable re-sort of them reproduces the saved assignment with zero mismatches.
* `session_id` restarts at 1 in the dev release, so it is not unique across splits. Use `(split, session_id)` or the
  `impression_uid` key added by the data-preparation notebook.
* The test file is not sorted by time (the dev release order is arbitrary); the processed copies are sorted.
* Each news file is exactly the set of articles referenced by its split; no referenced id is missing.

## Reproducing

```
# from the project root, with the raw releases unpacked into train/ (train release) and test/ (dev release)
jupyter nbconvert --to notebook --execute --inplace 0_Data_Acquisition/split_process.ipynb
jupyter nbconvert --to notebook --execute --inplace 1_Data_Preparation/clean_and_validate_data.ipynb
```

The second command re-checks the split, writes the typed processed files under `1_Data_Preparation/processed/`
and the check list `1_Data_Preparation/data_quality_report.csv`.
