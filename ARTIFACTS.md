# Shared model-ready data

Model developers download the completed data once, then run their model notebooks directly:

```bash
python download_artifacts.py --bundle model-ready
```

This restores `1_Data_Preparation/processed/`, `3_Representations/artifacts/` and the audited `4_Feature_Engineering/artifacts/prequential-all/`, plus selected quality/audit reports. It does not download raw splits or execute stages 0–4. Stage 0 is only for rebuilding original splits. The main bundle uses audited v2; the optimisation experiment is excluded.

Matching local files are verified and reused. Missing files are restored; different local files cause an error before extraction. Move conflicting datasets aside if you intend to replace them. Downloads and extraction are resumable by rerunning the command, with checksums required before files are accepted. Code compatibility is checked against the producer's shared Python helpers. Future notebook edits do not change the downloaded features; use the bundled schema and a compatible feature implementation for serving.

```bash
python download_artifacts.py --bundle model-ready --verify-only
# If Drive download access/quota fails, download the version folder in your browser:
python download_artifacts.py --bundle model-ready --local-package /path/to/model-ready-v2-audited
```

Do not follow the download command with `run_all.py --force`: that explicitly requests rebuilding. Notebook execution outputs are already in Git; dataset downloads do not execute or refresh notebooks.

## Owner: upload the prepared files

The completed package is already prepared at:

```text
results/artifact_bundles/model-ready-v2-audited/
    processed.zip
    representations.zip
    ranking_features.zip
    bundle_manifest.json
```

Total upload: approximately **3.75 GB**. These are the existing saved datasets; no rebuild was performed.

1. Open the team's shared **MIND** Drive folder.
2. Open/create a folder named **model-ready**.
3. Choose **New → Folder upload**, then select the local **model-ready-v2-audited** folder above.
4. Wait for all four files to finish uploading. Keep the ZIPs zipped and keep the folder/file names unchanged.

Expected layout:

```text
MIND/
    train/
    validation/
    test/
    model-ready/
        model-ready-v2-audited/
            processed.zip
            representations.zip
            ranking_features.zip
            bundle_manifest.json
```

The default downloader reads that path from the existing shared MIND link. It uses the folder's existing access permissions; it does not change sharing. Teammates need the latest project code containing the downloader and `artifact_bundles.json`. Live Drive verification passed on 22 October 2026: a fresh isolated project downloaded all four files with the default command, verified every archive/file checksum, restored all 40,363,532 feature rows and opened the feature/representation loaders. Repeat local verification also passed. No stages 0–4 ran. Evidence: `results/artifact_bundles/drive_download_test.json`.

For a different location, point directly at the **version folder**:

```bash
python download_artifacts.py --bundle model-ready --drive-folder 'https://drive.google.com/drive/folders/VERSION_FOLDER_ID'
```

## Owner: preparing a later version

```bash
python publish_artifacts.py --bundle model-ready
```

This only verifies/packages the completed local outputs and prints the folder to upload; it never uploads, downloads raw data or executes a notebook. An unchanged package retains its pinned manifest. Changed data/code require a new `--version VERSION_NAME`; validate that release and update `artifact_bundles.json` with its version, remote path and manifest SHA-256 before distributing it. Do not replace the four files of a published version in place.

No notebook upload cells are needed: publish the complete stage 1/3/4 bundle together so consumers get matching versions.
