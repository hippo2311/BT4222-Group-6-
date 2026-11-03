"""Transfer complete, versioned datasets without executing pipeline notebooks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parent
VERSION = "v3-mindsmall"
RUN = "4_Feature_Engineering/artifacts/prequential-all"
PREFIXES = ("1_Data_Preparation/processed/", "3_Representations/artifacts/", RUN + "/")
REPORTS = (
    "1_Data_Preparation/data_quality_report.csv",
    "results/feature_readiness_audit.json",
    "results/feature_review_status.json",
    "results/model_smoke_check.json",
    "results/notebook_consistency_check.json",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def safe_target(root, name):
    path = PurePosixPath(name)
    if "\\" in name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe bundle path: {name}")
    if not (name.startswith(PREFIXES) or name in REPORTS):
        raise ValueError(f"Unexpected bundle destination: {name}")
    target = root.joinpath(*path.parts)
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Bundle destination is a symlink: {parent}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Bundle destination escapes project: {name}")
    return target


def matches(path, info):
    return (
        path.is_file()
        and path.stat().st_size == info["bytes"]
        and digest(path) == info["sha256"]
    )


def verify_producer(root):
    """Verify saved data and dependencies; never run any build."""
    run = root / RUN
    if read_json(run / "run_status.json").get("status") != "complete":
        raise ValueError("Only a completed feature run can be shared")
    feature = read_json(run / "run_manifest.json")
    if not feature.get("full_run"):
        raise ValueError("Smoke features cannot be published as model-ready")
    for path in [
        run / "run_manifest.json",
        *(root / "3_Representations/artifacts").glob("*_manifest.json"),
    ]:
        meta = read_json(path)
        for kind in ("sources", "outputs"):
            for name, expected in meta.get(kind, {}).items():
                if digest(root / name) != expected:
                    raise ValueError(f"Saved artifact/dependency changed: {name}")
    audit = read_json(root / "results/feature_readiness_audit.json")
    # Keep the complete audit in the bundle; an archive is not a new model audit.
    if not audit:
        raise ValueError("Missing readiness evidence")
    return feature


def package_bundle(root=PROJECT, output=None, *, version=VERSION):
    root = Path(root).resolve()
    output = Path(output or root / "results/artifact_bundles" / version).resolve()
    feature = verify_producer(root)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        name: sorted(p for p in (root / folder).rglob("*") if p.is_file())
        for name, folder in zip(
            ("processed", "representations", "ranking_features"),
            ("1_Data_Preparation/processed", "3_Representations/artifacts", RUN),
        )
    }
    files["processed"] += [root / p for p in REPORTS if (root / p).is_file()]
    manifest = {
        "format": 1,
        "bundle": "model-ready",
        "version": version,
        "feature_run": RUN,
        "producer_git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "feature_manifest_sha256": digest(root / RUN / "run_manifest.json"),
        "rows_written": feature["rows_written"],
        "config_hash": feature["config_hash"],
        "compatible_code": {
            str(p.relative_to(root)): digest(p)
            for p in sorted((root / "src/newsrec").glob("*.py"))
        },
        "archives": {},
    }
    previous = output / "bundle_manifest.json"
    existing = read_json(previous) if previous.exists() else None
    if existing and any(
        existing.get(k) != manifest[k]
        for k in (
            "version",
            "compatible_code",
            "feature_manifest_sha256",
            "config_hash",
        )
    ):
        raise ValueError(
            "Published bundle versions are immutable. Package a new --version instead."
        )
    for name, paths in files.items():
        target = output / f"{name}.zip"
        inventory = {}
        for path in paths:
            relative = path.relative_to(root).as_posix()
            safe_target(root, relative)
            if path.is_symlink():
                raise ValueError(f"Refusing symlink source: {relative}")
            inventory[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        previous = output / "bundle_manifest.json"
        old = (
            read_json(previous).get("archives", {}).get(target.name)
            if previous.exists()
            else None
        )
        if old and (old["files"] != inventory or not matches(target, old)):
            raise ValueError(
                "Published files differ from this version. Package a new --version; the existing bundle is preserved."
            )
        if old:
            print("REUSE", target, flush=True)
            manifest["archives"][target.name] = old
            continue
        print("PACK", target, flush=True)
        staging = target.with_suffix(".zip.part")
        try:
            with zipfile.ZipFile(
                staging,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=1,
                allowZip64=True,
            ) as archive:
                for path in paths:
                    archive.write(path, path.relative_to(root).as_posix())
            # Ensure the source did not change during packaging.
            for relative, info in inventory.items():
                if not matches(root / relative, info):
                    raise ValueError(f"Source changed during packaging: {relative}")
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
        manifest["archives"][target.name] = {
            "bytes": target.stat().st_size,
            "sha256": digest(target),
            "files": inventory,
        }
    if existing:
        print("REUSE pinned manifest", previous, flush=True)
        return output
    write_json(output / "bundle_manifest.json", manifest)
    return output


def validate_manifest(manifest, root, *, check_code=True):
    if (
        manifest.get("format") != 1
        or manifest.get("bundle") != "model-ready"
        or manifest.get("feature_run") != RUN
    ):
        raise ValueError("Unsupported model-ready bundle")
    if set(manifest.get("archives", {})) != {
        "processed.zip",
        "representations.zip",
        "ranking_features.zip",
    }:
        raise ValueError("Bundle must contain all three stages")
    seen = set()
    for archive in manifest["archives"].values():
        for relative in archive["files"]:
            safe_target(root, relative)
            if relative in seen:
                raise ValueError(f"Duplicate bundled destination: {relative}")
            seen.add(relative)
    if check_code:
        for relative, expected in manifest["compatible_code"].items():
            path = root / relative
            if (
                not relative.startswith("src/newsrec/")
                or not path.is_file()
                or digest(path) != expected
            ):
                raise ValueError(
                    f"Code differs from bundle producer: {relative}. Use a compatible checkout; do not rebuild automatically."
                )


def missing_files(root, manifest):
    missing = {}
    for name, archive in manifest["archives"].items():
        missing[name] = []
        for relative, info in archive["files"].items():
            target = safe_target(root, relative)
            if target.exists():
                if not matches(target, info):
                    raise ValueError(
                        f"Local file differs from shared bundle: {relative}. Move the conflicting dataset aside first; nothing will be overwritten."
                    )
            else:
                missing[name].append(relative)
    return missing


def restore_archive(root, archive_path, spec, wanted):
    if not matches(archive_path, spec):
        raise ValueError(f"Archive checksum failed: {archive_path.name}")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or set(names) != set(spec["files"]):
            raise ValueError("Archive members differ from manifest")
        for item in archive.infolist():
            target = safe_target(root, item.filename)
            if (
                stat.S_ISLNK(item.external_attr >> 16)
                or item.file_size != spec["files"][item.filename]["bytes"]
            ):
                raise ValueError("Invalid archive member")
            if item.filename not in wanted:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=target.name + ".", suffix=".part", dir=target.parent
            )
            staging = Path(name)
            try:
                with os.fdopen(fd, "wb") as stream, archive.open(item) as source:
                    shutil.copyfileobj(source, stream, length=4 * 1024 * 1024)
                if not matches(staging, spec["files"][item.filename]):
                    raise ValueError(f"Extracted checksum failed: {item.filename}")
                # Link fails if a concurrent run created the destination: never overwrite.
                os.link(staging, target)
            finally:
                staging.unlink(missing_ok=True)


def install_bundle(root, manifest_path, get_archive, *, check_code=True):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(manifest_path)
    validate_manifest(manifest, root, check_code=check_code)
    missing = missing_files(
        root, manifest
    )  # Preflight every conflict before writing anything.
    for name, wanted in missing.items():
        if not wanted:
            print("REUSE verified local", name, flush=True)
            continue
        print("RESTORE", name, f"({len(wanted)} missing files)", flush=True)
        restore_archive(
            root,
            Path(get_archive(name, manifest["archives"][name])),
            manifest["archives"][name],
            set(wanted),
        )
    if any(missing_files(root, manifest).values()):
        raise ValueError("Bundle restore incomplete; retry the same command")
    receipt = root / "results/artifact_bundles/installed_model_ready.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        receipt,
        {
            "version": manifest["version"],
            "manifest_sha256": digest(manifest_path),
            "feature_run": manifest["feature_run"],
            "status": "complete",
        },
    )
    print(
        "Model-ready data verified. Start your model notebooks; stages 0-4 do not need to run.",
        flush=True,
    )
    return manifest
