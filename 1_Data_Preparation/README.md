# 1. Data cleaning and validation

BT4222 Group 6 · final project · task 2. Owner: Member 3 (Anh Minh Hoang).

`clean_and_validate_data.ipynb` reads the three split folders (`train/`, `validation/`, `test/`), re-checks the split, parses
and validates every history and impression list, checks the links between behaviour and news records, audits news-field quality
and entity-embedding coverage, runs the temporal-integrity checks, and writes typed processed files. Run time about 1 minute.

## Start here

`python run_all.py` runs stages 1–4 by default. You can also open this notebook directly.
The first cells reuse local inputs, or fetch missing split Parquets/entity vectors from the shared Drive,
then continue cleaning. Completed local inputs work without Drive access; existing files are preserved.
If Drive has a permissions/quota/missing-file problem, the error points to
`0_Data_Acquisition/download_split_process.ipynb` (`python run_all.py --from-stage 0 --to-stage 0`).
That recovery route uses original releases/local ZIPs, independently of Drive.

## Outputs

| File | Content |
|---|---|
| `data_quality_report.csv` | one row per check (178 rows: 111 PASS, 66 INFO, 1 WARN, 0 FAIL): area, split, check, value, expectation, status, detail |
| `processed/users.parquet` | `user_id`, `user_code` (750,434 users across the three splits) |
| `processed/impressions_<split>.parquet` | one row per impression, sorted by time: `impression_uid, split, session_id, user_id, user_code, timestamp, history, history_len, slate_len, n_clicks, n_negatives` |
| `processed/candidates_<split>.parquet` | one row per (impression, candidate): `impression_uid, user_code, timestamp, position, news_code, clicked` (70.3 M / 13.2 M / 14.1 M rows) |
| `processed/news_catalogue.parquet` | union of the three news files (104,151 articles) with split flags, text and entity counts, and per-split first/last exposure, exposure and click counts |

Keys: `impression_uid = split_code × 10,000,000 + session_id` (session ids restart at 1 in the dev release);
`news_code` = digits of `news_id`; `user_code` = rank of `user_id` over all users.

## Cleaning decisions (no row is dropped)

1. Keep all impressions; every row passes the grammar, label and duplicate checks.
2. Keep empty histories (`history_len = 0`): the cold-start users the system must serve.
3. Keep candidates already in the user's history (0.3 % of candidates; a legitimate repeat-exposure signal).
4. Keep null abstracts (5.3 %) as null; the representation notebook decides the fallback.
5. Keep `[]` entity lists as valid empty lists (about 25 % of articles per field).
6. Sort by `(timestamp, session_id)`; the dev release is not time-ordered.
7. Add stable cross-split keys and integer codes (above).
8. Explode impression strings; `position` is the shuffled file order and is **not** a model input.
9. Store per-split article exposure facts on the catalogue for convenience; they are retrospective and must only be used for
   events after `first_exposure_<split>`.

## Temporal-integrity results

* Windows are ordered: train 9 Nov 00:00 → 14 Nov 07:30:28, validation → 14 Nov 23:59:59, test = 15 Nov.
* The cut falls inside one second (4 rows in train, 9 in validation at 07:30:28); a stable re-sort reproduces the saved files exactly.
* Histories are static: 0 users whose history changes between impressions or between splits; test histories equal train histories
  for the 205,303 shared users, so in-window clicks are **not** in the test-day history and may be appended time-causally.
* Each news file is exactly the set of articles referenced by its split; no referenced id is missing; no duplicate ids.
* 18.4 % of validation users and 19.8 % of test users never appear in train; 47 % of test-day slate articles were shown in train
  (30 % of test exposures), the rest are new.
* Timezone of the timestamps is undocumented; hour-of-day features use the log clock.
