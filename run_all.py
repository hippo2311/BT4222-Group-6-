"""Execute stages 0-4 in order using this Python environment's notebook kernel.

Notebook builders author notebook files. This runner executes existing notebooks.
Raw MIND train/dev files must already have been downloaded and extracted locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    (0, "0_Data_Acquisition/split_process.ipynb"),
    (1, "1_Data_Preparation/clean_and_validate_data.ipynb"),
    (2, "2_EDA/impression_analysis.ipynb"),
    (2, "2_EDA/user_analysis.ipynb"),
    (2, "2_EDA/article_temporal_analysis.ipynb"),
    (3, "3_Representations/3_01_news_embeddings.ipynb"),
    (3, "3_Representations/3_02_entity_embeddings.ipynb"),
    (3, "3_Representations/3_03_user_profiles.ipynb"),
    (4, "4_Feature_Engineering/4_01_build_ranking_features.ipynb"),
]


@contextmanager
def environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def execute_notebook(source, cwd, output, timeout=None):
    """Save executed output even after failure; do not change source notebooks."""
    source, output = Path(source), Path(output)
    nb = nbformat.read(source, as_version=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="newsrec-kernel-") as temp:
        kernels = Path(temp)
        kernel = kernels / "newsrec-runner"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Pipeline Python",
                    "language": "python",
                }
            )
        )
        manager = KernelManager(
            kernel_name="newsrec-runner",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
        )
        client = NotebookClient(
            nb, km=manager, timeout=timeout, resources={"metadata": {"path": str(cwd)}}
        )

        def progress(cell, cell_index):
            if cell.cell_type == "code":
                print(f"    cell {cell_index + 1}/{len(nb.cells)}", flush=True)

        client.on_cell_start = progress
        try:
            client.execute()
        finally:
            nbformat.write(nb, output)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)


def plan(root, start=0, end=4, rebuild_splits=False):
    split_files = [
        root / split / f"{kind}_{split}.parquet"
        for split in ("train", "validation", "test")
        for kind in ("behaviors", "news")
    ]
    present = [p.exists() for p in split_files]
    reuse = all(present) and not rebuild_splits
    steps = [
        (stage, path)
        for stage, path in NOTEBOOKS
        if start <= stage <= end and not (stage == 0 and reuse)
    ]
    return steps, reuse, present


def preflight(root, steps, present, rebuild_splits):
    errors = []
    if not steps:
        return errors
    stages = {stage for stage, _ in steps}
    if 0 in stages:
        if any(present) and not rebuild_splits:
            errors.append(
                "Only some saved splits exist. Restore the agreed set, or explicitly use --rebuild-splits to replace it."
            )
        for split in ("train", "test"):
            for name in ("behaviors.tsv", "news.tsv"):
                if not (root / split / name).exists():
                    errors.append(f"Missing raw input: {root / split / name}")
    elif 1 in stages and not all(present):
        errors.append(
            "Stage 1 requires the six saved behaviors/news split Parquet files."
        )
    if min(stages) >= 2:
        p = root / "1_Data_Preparation/processed"
        expected = [p / "news_catalogue.parquet", p / "users.parquet"]
        expected += [
            p / f"{kind}_{s}.parquet"
            for s in ("train", "validation", "test")
            for kind in ("impressions", "candidates")
        ]
        for path in expected:
            if not path.exists():
                errors.append(f"Missing prepared input: {path}")
    selected = {path for _, path in steps}
    if (
        "3_Representations/3_02_entity_embeddings.ipynb" in selected
        and not (root / "train/entity_embedding.vec").exists()
    ):
        errors.append("Missing train/entity_embedding.vec (canonical entity vectors).")
    if min(stages) == 3:
        p = root / "3_Representations/artifacts"
        prerequisites = []
        if (
            "3_Representations/3_02_entity_embeddings.ipynb" in selected
            and "3_Representations/3_01_news_embeddings.ipynb" not in selected
        ):
            prerequisites.append("news_manifest.json")
        if "3_Representations/3_03_user_profiles.ipynb" in selected:
            if "3_Representations/3_01_news_embeddings.ipynb" not in selected:
                prerequisites.append("news_manifest.json")
            if "3_Representations/3_02_entity_embeddings.ipynb" not in selected:
                prerequisites.append("entity_manifest.json")
        for name in set(prerequisites):
            if not (p / name).exists():
                errors.append(f"Missing representation prerequisite: {p / name}")
    if min(stages) == 4:
        p = root / "3_Representations/artifacts"
        for name in (
            "news_manifest.json",
            "entity_manifest.json",
            "availability_manifest.json",
        ):
            if not (p / name).exists():
                errors.append(f"Missing representation prerequisite: {p / name}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", type=int, choices=range(5), default=0)
    parser.add_argument("--to-stage", type=int, choices=range(5), default=4)
    parser.add_argument(
        "--features-smoke",
        type=int,
        metavar="TIMESTAMPS",
        help="Build only this many initial feature timestamps; default builds full features",
    )
    parser.add_argument(
        "--rebuild-splits",
        action="store_true",
        help="Explicitly regenerate stage-0 splits; default reuses an existing complete saved set",
    )
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
        help="Explicitly replace the selected existing feature-run directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution order without running notebooks",
    )
    parser.add_argument(
        "--notebook",
        choices=[path for _, path in NOTEBOOKS],
        help="Execute one notebook only; its upstream artifacts must already exist",
    )
    args = parser.parse_args(argv)
    if args.from_stage > args.to_stage:
        parser.error("--from-stage must not exceed --to-stage")
    if args.features_smoke is not None and args.features_smoke < 1:
        parser.error("--features-smoke must be positive")
    steps, reuse, present = plan(
        ROOT, args.from_stage, args.to_stage, args.rebuild_splits
    )
    if args.notebook:
        stage = next(stage for stage, path in NOTEBOOKS if path == args.notebook)
        if stage == 0 and reuse:
            steps = []
        else:
            steps = [(stage, args.notebook)]
    if args.from_stage == 0 and reuse and not args.notebook:
        print("Stage 0: reusing the agreed saved split files (no resplitting).")
    for stage, path in steps:
        print(f"Stage {stage}: {path}")
    print(
        "Feature output:",
        "full dataset"
        if args.features_smoke is None
        else f"smoke: {args.features_smoke} timestamp batches",
    )
    if args.dry_run:
        return 0
    errors = preflight(ROOT, steps, present, args.rebuild_splits)
    if errors:
        print("\nCannot start:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 1
    if not steps:
        print("Nothing to execute.")
        return 0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run = ROOT / "runs" / stamp
    run.mkdir(parents=True)
    summary = {
        "status": "running",
        "python": sys.executable,
        "reused_saved_splits": reuse,
        "features_smoke": args.features_smoke,
        "notebooks": [],
    }

    def save():
        (run / "run_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    env = {
        "NEWSREC_ROOT": str(ROOT),
        "NEWSREC_FULL_FEATURES": "1" if args.features_smoke is None else "0",
        "NEWSREC_SMOKE_TIMESTAMPS": str(args.features_smoke or 1000),
        "NEWSREC_OVERWRITE_FEATURES": "1" if args.overwrite_features else "0",
    }
    with environment(env):
        for stage, relative in steps:
            item = {"stage": stage, "notebook": relative, "status": "running"}
            summary["notebooks"].append(item)
            save()
            start = time.perf_counter()
            print(f"\nRunning {relative}", flush=True)
            try:
                execute_notebook(ROOT / relative, ROOT, run / relative)
            except Exception as exc:  # noqa: BLE001 - record any notebook execution failure
                item.update(
                    status="failed", error=str(exc), seconds=time.perf_counter() - start
                )
                summary["status"] = "failed"
                save()
                print(
                    f"Stopped at {relative}. Executed cells/error saved under {run}",
                    file=sys.stderr,
                )
                return 1
            item.update(status="complete", seconds=time.perf_counter() - start)
            save()
            print(f"Completed in {item['seconds']:.1f}s", flush=True)
    summary["status"] = "complete"
    save()
    print(f"\nPipeline complete. Executed notebook copies and log: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
