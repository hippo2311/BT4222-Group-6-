# Shared model-ready data

The current release is **v3-mindsmall**, containing matching processed MIND-small data, representations, full training candidate features, and complete validation/test features. It passed full-data audits and a clean local restore. See [validation results](4_Feature_Engineering/mindsmall_epoch_validation.md).

## Owner: upload v3-mindsmall

The upload folder is:

```text
results/artifact_bundles/v3-mindsmall/
    processed.zip
    representations.zip
    ranking_features.zip
    bundle_manifest.json
```

Total upload is about **498 MB**. The convenience archive `results/artifact_bundles/v3-mindsmall.zip` contains that folder; extract it first. Upload the **whole v3-mindsmall folder** into the team's shared `MIND/model-ready/` folder. Keep the three inner ZIPs zipped and all four filenames unchanged.

Expected Drive layout:

```text
MIND/model-ready/v3-mindsmall/
    processed.zip
    representations.zip
    ranking_features.zip
    bundle_manifest.json
```

`artifact_bundles.json` pins this version, its remote path, and its manifest checksum. The default download will work after the owner finishes uploading. No v3 Drive download has been tested yet. The earlier `model-ready-v2-audited` MIND-large package remains unchanged.

## Model developers

After the upload, use the latest compatible project code:

```bash
python download_artifacts.py --bundle model-ready
python download_artifacts.py --bundle model-ready --verify-only
```

This restores `1_Data_Preparation/processed/`, `3_Representations/artifacts/`, `4_Feature_Engineering/artifacts/prequential-all/`, and readiness evidence. It does not download raw splits or execute notebooks. Existing matching files are reused. Conflicting old files are rejected before extraction; move an old MIND-large dataset aside before installing this release.

If Drive access fails, download the version folder manually and run:

```bash
python download_artifacts.py --bundle model-ready --local-package /path/to/v3-mindsmall
```

For another Drive location, point directly at the version folder:

```bash
python download_artifacts.py --bundle model-ready --drive-folder 'https://drive.google.com/drive/folders/VERSION_FOLDER_ID'
```

Use the bundled schema's `model_columns` and join labels on both keys. The training table contains **all 3,389,342 candidates**. An epoch-based ranker should call `newsrec.training.iter_training_epoch(run_dir, epoch)` with a new epoch number each pass to select up to four exposed negatives per positive. `training_negative_groups/` is only an epoch-zero reference. Validation and test are evaluated on their complete slates. See [training usage](4_Feature_Engineering/README.md).

Checksums verify archive contents and code compatibility. Downloads and extraction are resumable. Do not follow installation with `run_all.py --force` unless deliberately rebuilding the datasets.

## Preparing a later release

```bash
python publish_artifacts.py --bundle model-ready --version NEW_VERSION
```

This verifies and packages completed local outputs without uploading or executing notebooks. Use a fresh version for changed data/code; validate its restore and update the version, remote path, and manifest checksum in `artifact_bundles.json`. Published versions are immutable. Always package matching stages 1, 3, and 4 together.
