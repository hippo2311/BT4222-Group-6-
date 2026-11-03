# MIND-small epoch-sampling validation — 3 November 2026

The full MIND-small build and clean bundle restore passed. The upload release is **v3-mindsmall**. The existing MIND-large inputs and v2 package were preserved.

Data came from `arashnic/mind-news-dataset`, version 2, `MINDsmall_train`. The checked-in preparation transformations ran locally, omitting Colab authentication and Drive upload. All three representation notebooks and the feature notebook then ran on the complete isolated dataset. Executed notebooks and logs are under `runs/mindsmall-epoch/`.

| Split | Impressions | Saved candidate rows | Clicked candidates |
|---|---:|---:|---:|
| Train | 95,071 | 3,389,342 | 141,558 |
| Validation | 31,624 | 1,231,673 | 47,961 |
| Test | 30,270 | 1,222,429 | 46,825 |

Training is 9–12 November 2019, validation is 13 November, and test is 14 November. These are chronological splits of the training release, not the official dev release.

All 3,247,784 exposed training negatives were retained. The saved training pool is about 5.3 times the old sampled training row count. Training must use `newsrec.training.iter_training_epoch(run_dir, epoch)` to draw negatives afresh; simply fitting the full saved table uses all negatives.

| Epoch | Selected pointwise rows | Positive groups | Groups with a different negative draw from epoch 0 |
|---|---:|---:|---:|
| 0 | 642,102 | 141,558 | 0 |
| 1 | 642,193 | 141,558 | 123,394 |
| 2 | 642,340 | 141,558 | 123,413 |

Repeating epoch 1 reproduced the same rows and group memberships. Epoch 0 exactly matched the saved reference groups and the previous 642,102-row fixed draw. Across epochs 0–2, source-level sampling selected 1,257,551 distinct impression/article pairs. Overlap between epochs is allowed; a small negative pool can give the same draw every time.

Checks passed for every saved source label and candidate count, unique impression/article keys, finite values in all 50 model columns, article eligibility, complete validation/test slates, positive retention, same-impression exposed negatives, and artifact checksums. The package was restored into a clean temporary folder, both feature and representation loaders opened successfully, an epoch batch was read, and repeat installation reused verified files.

Feature replay took **29.20 minutes**, over 125,501 timestamps. Measured peak process RSS was **2.86 GB**. Audited epoch iteration took approximately 22–23 seconds per pass on this machine. No blocking errors were found. One existing Polars deprecation warning concerns the future `explode(empty_as_null=...)` default; it did not interrupt execution.

The upload folder contains only `processed.zip`, `representations.zip`, `ranking_features.zip`, and `bundle_manifest.json`, totalling **498.2 MB**. A convenience outer archive is `results/artifact_bundles/v3-mindsmall.zip`; extract it before uploading the version folder into `MIND/model-ready/`. Keep the three inner ZIPs zipped. Drive upload/download has not been performed for this release.

Manifest SHA-256: `52cd084bde2147425f988aafcdb45b7ec92135d0ce39f314ac5e2020907a6578`.

Evidence: `runs/mindsmall-epoch/validation_report.json`, `runs/mindsmall-epoch/audit.log`, `results/artifact_bundles/v3-mindsmall_restore_test.json`, and the bundled `results/feature_readiness_audit.json`. No epoch-based ranker was trained; these checks validate feature construction, sampling, and transfer, rather than ranking-quality improvement.
