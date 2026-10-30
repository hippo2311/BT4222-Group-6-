"""Small transfer fixtures: no MIND download or notebook execution required."""

import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _artifact_bundle as bundle
import download_artifacts as download


def fixture_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    manifest = {
        "format": 1,
        "bundle": "model-ready",
        "version": bundle.VERSION,
        "feature_run": bundle.RUN,
        "compatible_code": {},
        "archives": {},
    }
    for name, prefix in zip(
        ("processed", "representations", "ranking_features"), bundle.PREFIXES
    ):
        relative = prefix + "example.parquet"
        content = (name * 3).encode()
        archive = package / (name + ".zip")
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(relative, content)
        import hashlib

        manifest["archives"][archive.name] = {
            "bytes": archive.stat().st_size,
            "sha256": bundle.digest(archive),
            "files": {
                relative: {
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            },
        }
    bundle.write_json(package / "bundle_manifest.json", manifest)
    return package, manifest


def test_restore_and_reuse_without_archive_access(tmp_path):
    package, manifest = fixture_package(tmp_path)
    target = tmp_path / "target"
    bundle.install_bundle(
        target, package / "bundle_manifest.json", lambda name, spec: package / name
    )

    def forbidden(*args):
        raise AssertionError("Already verified local files must not fetch archives")

    bundle.install_bundle(target, package / "bundle_manifest.json", forbidden)
    assert (
        bundle.read_json(
            target / "results/artifact_bundles/installed_model_ready.json"
        )["status"]
        == "complete"
    )
    missing = next(iter(manifest["archives"]["processed.zip"]["files"]))
    (target / missing).unlink()
    bundle.install_bundle(
        target, package / "bundle_manifest.json", lambda name, spec: package / name
    )
    assert (target / missing).exists()


def test_conflict_preflight_writes_nothing(tmp_path):
    package, manifest = fixture_package(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    conflict = next(iter(manifest["archives"]["representations.zip"]["files"]))
    path = target / conflict
    path.parent.mkdir(parents=True)
    path.write_text("my data")
    with pytest.raises(ValueError, match="Local file differs"):
        bundle.install_bundle(
            target, package / "bundle_manifest.json", lambda name, spec: package / name
        )
    assert path.read_text() == "my data"
    assert not (target / "1_Data_Preparation").exists()


def test_archive_corruption_rejected(tmp_path):
    package, _ = fixture_package(tmp_path)
    (package / "processed.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Archive checksum"):
        bundle.install_bundle(
            tmp_path / "target",
            package / "bundle_manifest.json",
            lambda name, spec: package / name,
        )


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/tmp/escape",
        "3_Representations/artifacts/../escape",
        "run_all.py",
        "3_Representations\\artifacts\\x",
    ],
)
def test_unsafe_destinations_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        bundle.safe_target(tmp_path, name)


def test_symlink_destination_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "target"
    (target / "3_Representations").mkdir(parents=True)
    (target / "3_Representations/artifacts").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(ValueError, match="symlink"):
        bundle.safe_target(target, "3_Representations/artifacts/x.npy")


def test_google_drive_download_then_offline_reuse(tmp_path, monkeypatch):
    package, _ = fixture_package(tmp_path)
    config = tmp_path / "config"
    config.mkdir()
    bundle.write_json(
        config / "artifact_bundles.json",
        {
            "model-ready": {
                "manifest_sha256": bundle.digest(package / "bundle_manifest.json"),
                "drive_folder": "https://drive.google.com/drive/folders/example",
                "remote_prefix": "model-ready/" + bundle.VERSION,
            }
        },
    )
    monkeypatch.setattr(download, "PROJECT", config)
    calls = []

    def listing(**kwargs):
        return [
            SimpleNamespace(
                path="model-ready/" + bundle.VERSION + "/" + p.name, id=p.name
            )
            for p in package.iterdir()
        ]

    def transfer(id, output, **kwargs):
        calls.append(id)
        shutil.copyfile(package / id, output)
        return output

    monkeypatch.setitem(
        sys.modules,
        "gdown",
        SimpleNamespace(download_folder=listing, download=transfer),
    )
    root = tmp_path / "consumer"
    download.download_bundle(root)
    assert set(calls) == {p.name for p in package.iterdir()}
    calls.clear()
    download.download_bundle(root)
    assert calls == []
    download.download_bundle(root, verify_only=True)
    assert calls == []


def test_manifest_pin_rejects_changed_package(tmp_path, monkeypatch):
    package, _ = fixture_package(tmp_path)
    bundle.write_json(
        tmp_path / "artifact_bundles.json",
        {"model-ready": {"manifest_sha256": "0" * 64}},
    )
    monkeypatch.setattr(download, "PROJECT", tmp_path)
    with pytest.raises(ValueError, match="pinned version"):
        download.download_bundle(tmp_path / "target", local_package=package)


def test_package_keeps_pinned_manifest_after_git_commit(tmp_path, monkeypatch):
    root = tmp_path / "producer"
    for prefix in bundle.PREFIXES:
        folder = root / prefix
        folder.mkdir(parents=True)
        (folder / "example.parquet").write_bytes(b"data")
    (root / "src/newsrec").mkdir(parents=True)
    (root / "src/newsrec/common.py").write_text("# stable model code")
    bundle.write_json(
        root / bundle.RUN / "run_manifest.json", {"rows_written": {"train": 1}}
    )
    monkeypatch.setattr(
        bundle,
        "verify_producer",
        lambda root: {"rows_written": {"train": 1}, "config_hash": "test"},
    )
    monkeypatch.setattr(
        bundle.subprocess, "check_output", lambda *a, **kw: "commit-a\n"
    )
    output = bundle.package_bundle(root)
    before = (output / "bundle_manifest.json").read_bytes()
    monkeypatch.setattr(
        bundle.subprocess, "check_output", lambda *a, **kw: "commit-b\n"
    )
    bundle.package_bundle(root)
    assert (output / "bundle_manifest.json").read_bytes() == before
    (root / "1_Data_Preparation/processed/example.parquet").write_bytes(b"changed data")
    with pytest.raises(ValueError, match="Published files differ"):
        bundle.package_bundle(root)
    assert (output / "bundle_manifest.json").read_bytes() == before
