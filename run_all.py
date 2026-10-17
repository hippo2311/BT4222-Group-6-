"""Execute selected notebooks (default: stages 1-4 with local/Drive inputs).

Notebook builders author notebook files. This runner executes existing notebooks.
Stage 0 downloads original MIND releases and splits them. Stage 1 can fetch agreed Drive splits when local inputs are absent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import nbformat
import pyarrow.parquet as pq
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
HEARTBEAT_SECONDS = 30
SPLIT_NOTEBOOK = "0_Data_Acquisition/download_split_process.ipynb"
NOTEBOOKS = [
    (0, SPLIT_NOTEBOOK),
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


class LiveNotebookClient(NotebookClient):
    """Keep notebook outputs while also forwarding text to the terminal."""

    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg["msg_type"] == "stream":
            stream = sys.stderr if msg["content"]["name"] == "stderr" else sys.stdout
            print(msg["content"]["text"], end="", file=stream, flush=True)
        return result


def execute_notebook(source, cwd, output=None, timeout=None):
    """Refresh source outputs by default, saving completed cells and errors in place."""
    source = Path(source)
    output = source if output is None else Path(output)
    nb = nbformat.read(source, as_version=4)
    # A failed run must not leave old results in cells it never reached.
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        # Replace atomically so an interrupted write cannot truncate the notebook.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, suffix=".ipynb", delete=False
        ) as stream:
            staging = Path(stream.name)
        try:
            nbformat.write(nb, staging)
            staging.replace(output)
        finally:
            staging.unlink(missing_ok=True)

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
        client = LiveNotebookClient(
            nb, km=manager, timeout=timeout, resources={"metadata": {"path": str(cwd)}}
        )

        active_cell = None
        stopped = threading.Event()

        def progress(cell, cell_index):
            nonlocal active_cell
            if cell.cell_type == "code":
                active_cell = (cell_index + 1, time.perf_counter())
                print(f"    cell {cell_index + 1}/{len(nb.cells)}", flush=True)

        def completed(cell, cell_index, **kwargs):
            nonlocal active_cell
            active_cell = None
            save()

        def heartbeat():
            while not stopped.wait(HEARTBEAT_SECONDS):
                current = active_cell
                if current is not None:
                    index, start = current
                    print(
                        f"    cell {index}/{len(nb.cells)} still executing "
                        f"({time.perf_counter() - start:.0f}s elapsed)",
                        flush=True,
                    )

        client.on_cell_start = progress
        client.on_cell_executed = completed
        monitor = threading.Thread(target=heartbeat, daemon=True)
        monitor.start()
        try:
            client.execute()
        finally:
            stopped.set()
            monitor.join()
            try:
                save()
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)


def plan(root, start=0, end=4, rebuild_splits=False):
    split_files = [
        root / split / f"{kind}_{split}.parquet"
        for split in ("train", "validation", "test")
        for kind in ("behaviors", "news")
    ]
    present = [p.exists() for p in split_files]
    vectors_present = all(
        (root / split / "entity_embedding.vec").is_file() for split in ("train", "test")
    )
    marker = root / "0_Data_Acquisition/split_manifest.json"
    try:
        building = (
            marker.exists()
            and json.loads(marker.read_text()).get("status") == "building"
        )
    except ValueError:
        building = True
    readable = all(p.is_file() and p.stat().st_size > 0 for p in split_files)
    if readable:
        try:
            for path in split_files:
                pq.read_metadata(path)
        except (OSError, ValueError):
            readable = False
    vectors_present = vectors_present and all(
        (root / split / "entity_embedding.vec").stat().st_size > 0
        for split in ("train", "test")
    )
    reuse = readable and vectors_present and not building and not rebuild_splits
    steps = [
        (stage, path)
        for stage, path in NOTEBOOKS
        if start <= stage <= end and not (path == SPLIT_NOTEBOOK and reuse)
    ]
    return steps, reuse, present


def preflight(root, steps, present, rebuild_splits):
    errors = []
    if not steps:
        return errors
    stages = {stage for stage, _ in steps}
    selected = {path for _, path in steps}
    # Stages 0 and 1 produce inputs themselves: stage 0 from MIND, stage 1 from Drive.
    providing_inputs = bool(stages & {0, 1})
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
    if (
        "3_Representations/3_02_entity_embeddings.ipynb" in selected
        and not (root / "train/entity_embedding.vec").exists()
        and not providing_inputs
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
    parser.add_argument("--from-stage", type=int, choices=range(5), default=1)
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
    if args.rebuild_splits and args.notebook is None:
        args.from_stage = 0
    if args.from_stage > args.to_stage:
        parser.error("--from-stage must not exceed --to-stage")
    if args.features_smoke is not None and args.features_smoke < 1:
        parser.error("--features-smoke must be positive")
    steps, reuse, present = plan(
        ROOT, args.from_stage, args.to_stage, args.rebuild_splits
    )
    if args.notebook:
        stage = next(stage for stage, path in NOTEBOOKS if path == args.notebook)
        if args.notebook == SPLIT_NOTEBOOK and reuse:
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
    env = {
        "NEWSREC_ROOT": str(ROOT),
        "NEWSREC_REBUILD_SPLITS": "1" if args.rebuild_splits else "0",
        "NEWSREC_FULL_FEATURES": "1" if args.features_smoke is None else "0",
        "NEWSREC_SMOKE_TIMESTAMPS": str(args.features_smoke or 1000),
        "NEWSREC_OVERWRITE_FEATURES": "1" if args.overwrite_features else "0",
    }
    with environment(env):
        for stage, relative in steps:
            start = time.perf_counter()
            print(f"\nRunning {relative}", flush=True)
            try:
                execute_notebook(ROOT / relative, ROOT)
            except Exception as exc:  # noqa: BLE001 - record any notebook execution failure
                print(
                    f"Stopped at {relative}: {exc}\n"
                    "Completed cells and error output are saved in that source notebook.",
                    file=sys.stderr,
                )
                return 1
            print(f"Completed in {time.perf_counter() - start:.1f}s", flush=True)
    print("\nPipeline complete. Outputs updated directly in the source notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
