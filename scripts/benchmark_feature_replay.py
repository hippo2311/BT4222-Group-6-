"""Compare a chronological prefix with the pre-optimisation replay driver.

Run from the project root: PYTHONPATH=src .venv/bin/python scripts/benchmark_feature_replay.py
All writes go under results/optimization; source data/representations are linked read-only by convention.
"""

import argparse
import cProfile
import hashlib
import json
import pstats
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from newsrec.common import Settings
from newsrec.pipeline import build_ranking_dataset
from newsrec.ranking import (
    TemporalState,
    build_features,
    feature_schema,
    sample_training_groups,
)

REFERENCE = "e9b1ba1a93993f25e1ee714f503e7cc49b7a16b7"


def read_parts(run, name, cutoff=None, request_ids=None):
    parts = sorted((run / name).glob("*.parquet"))
    if not parts:
        return None
    dataset = ds.dataset(parts, format="parquet")
    predicate = (
        None if cutoff is None else ds.field("timestamp") <= pd.Timestamp(cutoff)
    )
    if request_ids is not None:
        predicate = ds.field("impression_uid").isin(request_ids)
    return dataset.to_table(filter=predicate).to_pandas().reset_index(drop=True)


def compare(left, right, cutoff=None):
    checks = {}
    for name in [
        *(
            f"{kind}_{split}"
            for kind in ("ranking_features", "labels", "request_profiles")
            for split in ("train", "validation", "test")
        ),
        "training_negative_groups",
    ]:
        a = read_parts(left, name)
        # Negative groups have no timestamp; restrict them to the checked request IDs.
        ids = None
        if cutoff is not None and (
            name == "training_negative_groups" or name.startswith("labels_")
        ):
            split = (
                name.removeprefix("labels_") if name.startswith("labels_") else "train"
            )
            requests = read_parts(left, f"request_profiles_{split}")
            ids = requests.impression_uid.to_list() if requests is not None else []
        b = read_parts(
            right,
            name,
            cutoff
            if not (name == "training_negative_groups" or name.startswith("labels_"))
            else None,
            ids,
        )
        if a is None or b is None:
            assert a is None and (b is None or len(b) == 0), name
            continue
        keys = [
            c for c in ("impression_uid", "news_code", "positive_news_code") if c in a
        ]
        a = a.sort_values(keys).reset_index(drop=True)
        b = b.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6, atol=1e-7)
        for key in keys + (["clicked"] if "clicked" in a else []):
            np.testing.assert_array_equal(a[key], b[key])
        checks[name] = len(a)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamps", type=int, default=3000)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile both variants; timings then include profiler overhead",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Compare existing benchmark outputs without replaying",
    )
    parser.add_argument(
        "--variant", choices=("reference", "optimized"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.timestamps < 1:
        parser.error("--timestamps must be positive")
    root = Path(__file__).resolve().parents[1]
    work = root / "results/optimization" / f"compare-{args.timestamps}"
    work.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(
        ["git", "show", f"{REFERENCE}:src/newsrec/pipeline.py"], cwd=root, text=True
    )
    reference = types.ModuleType("newsrec._reference_pipeline")
    exec(compile(source, f"{REFERENCE}:pipeline.py", "exec"), reference.__dict__)  # noqa: S102 - trusted pinned project revision
    baseline_manifest = (
        root / "4_Feature_Engineering/artifacts/prequential-all/run_manifest.json"
    )
    baseline_hash = (
        hashlib.sha256(baseline_manifest.read_bytes()).hexdigest()
        if baseline_manifest.exists()
        else None
    )
    results = {
        "reference_commit": REFERENCE,
        "timestamps": args.timestamps,
        "profiled": args.profile,
        "audited_manifest_sha256": baseline_hash,
        "variants": {},
    }
    runs = {}
    for label, builder in (
        ("reference", reference.build_ranking_dataset),
        ("optimized", build_ranking_dataset),
    ):
        if args.variant is not None and label != args.variant:
            continue
        sandbox = work / label
        for rel in ("1_Data_Preparation/processed", "3_Representations/artifacts"):
            link = sandbox / rel
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(root / rel, target_is_directory=True)
        if not args.variant and not args.reuse:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--timestamps",
                str(args.timestamps),
                "--variant",
                label,
            ]
            if args.profile:
                command.append("--profile")
            subprocess.run(command, cwd=root, check=True)
        if not args.variant:
            suffix = "-optimized" if label == "optimized" else ""
            run = (
                sandbox
                / f"4_Feature_Engineering/artifacts/prequential-all{suffix}-smoke-{args.timestamps}"
            )
            assert (
                json.loads((run / "run_status.json").read_text())["status"]
                == "complete"
            )
            runs[label] = run
            measurement = sandbox / "measurement.json"
            results["variants"][label] = (
                json.loads(measurement.read_text())
                if measurement.exists()
                else pd.read_csv(run / "feature_cost.csv").iloc[-1].to_dict()
            )
            continue
        print(
            f"\nStarting {label}: {args.timestamps:,} chronological timestamps",
            flush=True,
        )
        profiler = cProfile.Profile() if args.profile else None
        start = time.perf_counter()
        if profiler:
            profiler.enable()
        run = builder(
            sandbox,
            Settings(),
            max_timestamps=args.timestamps,
            overwrite=True,
            state_factory=TemporalState,
            feature_builder=build_features,
            negative_sampler=sample_training_groups,
            schema_builder=feature_schema,
        )
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(sandbox / "replay.prof"))
            with (sandbox / "profile.txt").open("w") as handle:
                pstats.Stats(profiler, stream=handle).strip_dirs().sort_stats(
                    "cumtime"
                ).print_stats(50)
        results["variants"][label] = {
            "end_to_end_seconds": time.perf_counter() - start,
            **pd.read_csv(run / "feature_cost.csv").iloc[-1].to_dict(),
        }
        runs[label] = run
        (sandbox / "measurement.json").write_text(
            json.dumps(results["variants"][label], indent=2) + "\n"
        )
        print(json.dumps(results["variants"][label], indent=2), flush=True)
    if args.variant:
        return
    results["parity"] = compare(runs["optimized"], runs["reference"])
    pd.testing.assert_frame_equal(
        pd.read_csv(runs["optimized"] / "coverage.csv"),
        pd.read_csv(runs["reference"] / "coverage.csv"),
    )
    results["full_log_coverage_equal"] = True
    if baseline_manifest.exists():
        cutoff = json.loads((runs["optimized"] / "run_manifest.json").read_text())[
            "last_scored_timestamp"
        ]
        results["audited_full_prefix_parity"] = compare(
            runs["optimized"], baseline_manifest.parent, cutoff
        )
        assert (
            hashlib.sha256(baseline_manifest.read_bytes()).hexdigest() == baseline_hash
        )
    old = results["variants"]["reference"]["wall_seconds"]
    new = results["variants"]["optimized"]["wall_seconds"]
    results["replay_speedup"] = old / new
    results["scope"] = (
        "Chronological prefix parity only; full v2 audit does not certify the prototype."
    )
    (work / "comparison.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
    print(
        f"\nPASS: identical selected keys/labels; feature/profile parity within tolerance. Replay speedup {old / new:.2f}x."
    )
    print("Report:", work / "comparison.json")


if __name__ == "__main__":
    main()
