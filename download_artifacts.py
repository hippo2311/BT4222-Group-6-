"""Download and verify the model-ready dataset, then start model work directly."""

from __future__ import annotations

import argparse
from pathlib import Path

from _artifact_bundle import (
    PROJECT,
    VERSION,
    digest,
    install_bundle,
    matches,
    read_json,
    validate_manifest,
)


def download_bundle(
    root=PROJECT, *, local_package=None, drive_folder=None, verify_only=False
):
    root = Path(root).resolve()
    config = read_json(PROJECT / "artifact_bundles.json")["model-ready"]
    expected_manifest = config["manifest_sha256"]
    version = config.get("version", VERSION)
    cache = root / "results/artifact_bundles/downloads" / version
    cache.mkdir(parents=True, exist_ok=True)
    remote = None

    def remote_entries():
        nonlocal remote
        if remote is None:
            import gdown

            print("Reading shared Google Drive bundle listing...", flush=True)
            entries = gdown.download_folder(
                url=drive_folder or config["drive_folder"],
                output=str(cache),
                skip_download=True,
                quiet=True,
                use_cookies=False,
                timeout=30,
            )
            prefix = "" if drive_folder else config["remote_prefix"].strip("/") + "/"
            remote = {}
            for entry in entries:
                path = str(entry.path).replace("\\", "/")
                if path.startswith(prefix):
                    name = path[len(prefix) :]
                    if name in remote:
                        raise ValueError(f"Duplicate Drive bundle filename: {name}")
                    remote[name] = entry.id
        return remote

    def fetch(name, expected):
        import gdown

        entries = remote_entries()
        if name not in entries:
            raise FileNotFoundError(
                f"Shared bundle is missing {name}. Upload the complete version folder into MIND/model-ready/. No notebook build was started."
            )
        target = cache / name
        part = target.with_name(target.name + ".download")
        print("DOWNLOAD", name, flush=True)
        result = gdown.download(
            id=entries[name],
            output=str(part),
            quiet=False,
            use_cookies=False,
            resume=True,
            retries=3,
            timeout=30,
        )
        if result is None or digest(part) != expected:
            part.unlink(missing_ok=True)
            raise ValueError(
                f"Download checksum failed: {name}; retry the same command"
            )
        part.replace(target)
        return target

    local = Path(local_package).resolve() if local_package else None
    manifest_path = (
        (local / "bundle_manifest.json") if local else cache / "bundle_manifest.json"
    )
    if not manifest_path.exists():
        # The producer can verify its own already-packaged data without re-downloading.
        producer = (
            PROJECT / "results/artifact_bundles" / version / "bundle_manifest.json"
        )
        if not local and producer.exists() and digest(producer) == expected_manifest:
            manifest_path = producer
        elif verify_only or local:
            raise FileNotFoundError(
                "No pinned local bundle manifest; download the bundle first"
            )
        else:
            manifest_path = fetch("bundle_manifest.json", expected_manifest)
    if digest(manifest_path) != expected_manifest:
        raise ValueError(
            "Bundle manifest differs from the pinned version in artifact_bundles.json"
        )
    manifest = read_json(manifest_path)
    validate_manifest(manifest, root)

    def get_archive(name, spec):
        candidates = (
            [local / name] if local else [manifest_path.parent / name, cache / name]
        )
        for path in candidates:
            if matches(path, spec):
                return path
        if verify_only:
            raise FileNotFoundError(
                f"Missing local dataset files; --verify-only will not download {name}"
            )
        if local:
            raise ValueError(f"Local package archive is missing or corrupt: {name}")
        return fetch(name, spec["sha256"])

    if verify_only:
        from _artifact_bundle import missing_files

        if any(missing_files(root, manifest).values()):
            raise FileNotFoundError(
                "Local model-ready data is incomplete; run without --verify-only"
            )
    return install_bundle(root, manifest_path, get_archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", choices=["model-ready"], required=True)
    parser.add_argument(
        "--root", type=Path, default=PROJECT, help="Local project/data root"
    )
    parser.add_argument(
        "--local-package",
        type=Path,
        help="Restore a manually downloaded version folder instead of contacting Drive",
    )
    parser.add_argument(
        "--drive-folder",
        help="Override with a Drive URL pointing directly to the version folder",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check local datasets without downloading or extracting",
    )
    args = parser.parse_args()
    try:
        download_bundle(
            args.root,
            local_package=args.local_package,
            drive_folder=args.drive_folder,
            verify_only=args.verify_only,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports Drive/network errors without triggering builds
        parser.exit(
            1,
            f"Artifact download stopped: {exc}\nNo training or feature rebuild was started. Existing datasets are preserved.\n",
        )


if __name__ == "__main__":
    main()
