from __future__ import annotations

import heapq
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .common import (
    DEFAULT_SETTINGS,
    SPLITS,
    manifest,
    measure,
    normalised_mean,
    require_inputs,
    sha256,
    upsert_report,
    validate_feature_schema,
    write_json,
)
from .progress import ReplayProgress
from .representations import Bundle, artifact_dir
from .temporal import TemporalState


def timestamp_groups(path, columns, batch_size=100000):
    """Bounded parquet reads retaining complete timestamp groups across row groups."""
    pending = None
    last = None
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=batch_size, columns=columns
    ):
        frame = batch.to_pandas()
        frame["timestamp"] = pd.to_datetime(frame.timestamp).astype("datetime64[ns]")
        if frame.timestamp.isna().any() or not frame.timestamp.is_monotonic_increasing:
            raise ValueError(f"Unsorted or missing timestamps: {path}")
        if last is not None and frame.timestamp.iloc[0] < last:
            raise ValueError(f"Parquet row groups out of order: {path}")
        last = frame.timestamp.iloc[-1]
        if pending is not None:
            frame = pd.concat([pending, frame], ignore_index=True)
        tail = frame.timestamp.iloc[-1]
        ready = frame[frame.timestamp != tail]
        pending = frame[frame.timestamp == tail]
        for t, group in ready.groupby("timestamp", sort=False):
            yield (pd.Timestamp(t), group.reset_index(drop=True))
    if pending is not None:
        yield (pd.Timestamp(pending.timestamp.iloc[0]), pending.reset_index(drop=True))


def replay_batches(root, cfg=DEFAULT_SETTINGS, splits=SPLITS):
    processed = require_inputs(root)
    streams = []
    for split in splits:
        for kind, columns in (
            ("impressions", ["impression_uid", "user_code", "timestamp", "history"]),
            (
                "candidates",
                ["impression_uid", "user_code", "timestamp", "news_code", "clicked"],
            ),
        ):
            streams.append(
                (
                    kind,
                    split,
                    iter(
                        timestamp_groups(
                            processed / f"{kind}_{split}.parquet",
                            columns,
                            cfg.parquet_batch_rows,
                        )
                    ),
                )
            )
    heap = []
    for i, (_, _, stream) in enumerate(streams):
        first = next(stream, None)
        if first is not None:
            heapq.heappush(heap, (first[0], i, first[1]))
    while heap:
        t = heap[0][0]
        requests = []
        events = []
        while heap and heap[0][0] == t:
            _, i, frame = heapq.heappop(heap)
            kind, split, stream = streams[i]
            frame = frame.copy()
            frame["split"] = split
            (requests if kind == "impressions" else events).append(frame)
            nxt = next(stream, None)
            if nxt is not None:
                heapq.heappush(heap, (nxt[0], i, nxt[1]))
        if not requests or not events:
            raise ValueError(f"Unpaired requests/events at {t}")
        req = pd.concat(requests, ignore_index=True)
        ev = pd.concat(events, ignore_index=True)
        if (
            req.impression_uid.duplicated().any()
            or ev.duplicated(["impression_uid", "news_code"]).any()
        ):
            raise ValueError("Duplicate request/pair in timestamp stream")
        if set(ev.impression_uid) != set(req.impression_uid):
            raise ValueError("Every request must have its logged candidate slate")
        owners = req.set_index("impression_uid").user_code
        if not np.array_equal(
            ev.user_code.to_numpy(), ev.impression_uid.map(owners).to_numpy()
        ):
            raise ValueError("Candidate/request user mismatch")
        yield (t, req, ev)


class PartitionWriter:
    """Write typed Parquet parts; keep only a bounded number of pending rows."""

    def __init__(self, path, rows=100000, schema=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.pending = 0
        self.count = 0
        self.part = 0
        self.threshold = rows
        self.schema = schema

    def add(self, frame):
        if not len(frame):
            return
        self.frames.append(frame)
        self.pending += len(frame)
        self.count += len(frame)
        if self.pending >= self.threshold:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        table = pa.Table.from_pandas(
            pd.concat(self.frames, ignore_index=True), preserve_index=False
        )
        if self.schema is None:
            self.schema = table.schema
        table = table.cast(self.schema)
        pq.write_table(
            table, self.path / f"part-{self.part:05d}.parquet", compression="zstd"
        )
        self.part += 1
        self.frames = []
        self.pending = 0

    def close(self):
        self.flush()


def time_horizon(root, splits=SPLITS):
    p = require_inputs(root)
    ranges = [
        pl.scan_parquet(p / f"impressions_{s}.parquet")
        .select(
            pl.col("timestamp").min().alias("min"),
            pl.col("timestamp").max().alias("max"),
        )
        .collect()
        .row(0)
        for s in splits
    ]
    return (min(x[0] for x in ranges), max(x[1] for x in ranges))


def pool_supplied_history(history, bundle):
    raw = "" if pd.isna(history) else str(history)
    codes = [int(x.removeprefix("N")) for x in raw.split()]
    rows = bundle.rows(codes)
    text, valid_text = normalised_mean(bundle.text, rows)
    entity, valid_entity = normalised_mean(bundle.entity, rows)
    return rows, text, entity, valid_text, valid_entity


def build_supplied_profiles(
    root, cfg=DEFAULT_SETTINGS, *, history_pooler=pool_supplied_history
):
    """Supplied-only matrix for baseline/cluster handoff; no in-window feedback."""
    processed = require_inputs(root)
    out = artifact_dir(root)
    bundle = Bundle(root, cfg)
    with measure(root, "supplied_profiles"):
        users = (
            pd.read_parquet(processed / "users.parquet")
            .sort_values("user_code")
            .reset_index(drop=True)
        )
        if users.user_code.duplicated().any():
            raise ValueError("Duplicate user codes")
        users["profile_row"] = np.arange(len(users), dtype=np.int32)
        user_rows = dict(zip(users.user_code, users.profile_row))
        text = np.lib.format.open_memmap(
            out / "supplied_text_profiles.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(users), cfg.text_dim),
        )
        entity = np.lib.format.open_memmap(
            out / "supplied_entity_profiles.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(users), cfg.entity_dim),
        )
        text[:] = 0
        entity[:] = 0
        quality = []
        seen = {}
        train_users = set()
        for split in SPLITS:
            for batch in pq.ParquetFile(
                processed / f"impressions_{split}.parquet"
            ).iter_batches(
                batch_size=cfg.parquet_batch_rows, columns=["user_code", "history"]
            ):
                for row in (
                    batch.to_pandas()
                    .drop_duplicates(["user_code", "history"])
                    .itertuples(index=False)
                ):
                    u = int(row.user_code)
                    if split == "train":
                        train_users.add(u)
                    raw = "" if pd.isna(row.history) else str(row.history)
                    if u in seen:
                        if seen[u] != raw:
                            raise ValueError(f"Supplied history changed: {u}")
                        continue
                    seen[u] = raw
                    rows, text_vector, entity_vector, vt, ve = history_pooler(
                        raw, bundle
                    )
                    i = user_rows[u]
                    text[i] = text_vector
                    entity[i] = entity_vector
                    quality.append(
                        {
                            "user_code": u,
                            "history_length": len(rows),
                            "valid_text_history_count": vt,
                            "valid_entity_history_count": ve,
                            "history_over_cap": len(rows) > cfg.history_cap,
                        }
                    )
        if len(seen) != len(users):
            raise ValueError("Users mapping does not match observed users")
        text.flush()
        entity.flush()
        users = users.merge(
            pd.DataFrame(quality), on="user_code", validate="one_to_one", sort=False
        )
        users["seen_in_train"] = users.user_code.isin(train_users)
        users["has_text_profile"] = np.linalg.norm(text, axis=1) > 0
        users["has_entity_profile"] = np.linalg.norm(entity, axis=1) > 0
        users.to_parquet(out / "supplied_profile_index.parquet", index=False)
        coverage = []
        for name, count in (
            ("supplied_text_profile", "valid_text_history_count"),
            ("supplied_entity_profile", "valid_entity_history_count"),
        ):
            for cohort, mask in (
                ("all_users", np.ones(len(users), bool)),
                ("training_users", users.seen_in_train.to_numpy()),
            ):
                n = int(mask.sum())
                k = int(
                    users.loc[
                        mask,
                        "has_text_profile"
                        if count == "valid_text_history_count"
                        else "has_entity_profile",
                    ].sum()
                )
                coverage.append(
                    {
                        "split": "all" if cohort == "all_users" else "train",
                        "representation": name,
                        "cohort": cohort,
                        "denominator": n,
                        "valid": k,
                        "fallback": n - k,
                        "valid_fraction": k / max(1, n),
                    }
                )
        upsert_report(
            root,
            "representation_coverage.csv",
            pd.DataFrame(coverage),
            "supplied_profiles",
        )
        files = [
            out / f
            for f in (
                "supplied_text_profiles.npy",
                "supplied_entity_profiles.npy",
                "supplied_profile_index.parquet",
            )
        ]
        sources = [processed / f"impressions_{s}.parquet" for s in SPLITS] + [
            out / "news_manifest.json",
            out / "entity_manifest.json",
        ]
        write_json(
            out / "profiles_manifest.json",
            manifest(
                root,
                cfg,
                sources,
                files,
                clustering_policy="fit scaler/clusters only on seen_in_train rows; supplied history only",
                temporal_policy="static supplied profiles are not time-updated ranking profiles",
                article_index_hash=sha256(out / "article_index.parquet"),
            ),
        )
    return users


def _profile_references(requests, state, t):
    refs = []
    for row in requests.itertuples(index=False):
        p = state.profile(row.user_code, t)
        refs.append(
            {
                "impression_uid": int(row.impression_uid),
                "user_code": int(row.user_code),
                "timestamp": t,
                "state_version": p["state_version"],
                "history_rows": p["history_rows"].tolist(),
                "full_history_rows": p["full_history_rows"].tolist(),
                "history_mask": p["history_mask"].tolist(),
                "history_length": p["history_length"],
                "retained_length": p["retained_length"],
                "interest_group_count": p["group_count"],
                "selected_interest_categories": p["selected_interest_categories"],
                "recent_rows": p["recent_rows"],
                "no_supplied_history": p["no_supplied_history"],
                "has_text": p["has_text"],
                "has_entity": p["has_entity"],
            }
        )
    return pd.DataFrame(refs)


def replay_workload(processed, splits, max_timestamps=None):
    """Count replay work for display, without reading candidate labels/history."""
    timestamps = (
        pl.concat(
            [
                pl.scan_parquet(processed / f"impressions_{s}.parquet").select(
                    "timestamp"
                )
                for s in splits
            ]
        )
        .select(pl.col("timestamp").n_unique())
        .collect()
        .item()
    )
    if max_timestamps is not None:
        return min(timestamps, max_timestamps), None
    events = sum(
        pq.ParquetFile(processed / f"candidates_{s}.parquet").metadata.num_rows
        for s in splits
    )
    return timestamps, events


def build_ranking_dataset(
    root,
    cfg=DEFAULT_SETTINGS,
    *,
    target_split=None,
    max_timestamps=None,
    overwrite=False,
    state_factory=TemporalState,
    feature_builder=None,
    negative_sampler=None,
    schema_builder=None,
):
    """Replay complete logs; retain the full training pool for epoch sampling.

    max_timestamps is a chronological smoke run starting at training start, never
    an arbitrary slice without state. Smoke output has a separate run directory.
    Frozen mode requires one target split; earlier splits still update state.
    """
    if not all(
        callable(f) for f in (feature_builder, negative_sampler, schema_builder)
    ):
        raise ValueError(
            "Pass feature_builder, negative_sampler and schema_builder from the feature notebook"
        )
    active_schema = validate_feature_schema(schema_builder(cfg))
    root = Path(root)
    if max_timestamps is not None and max_timestamps < 1:
        raise ValueError("max_timestamps must be positive or None")
    processed = require_inputs(root)
    if cfg.update_mode == "frozen" and target_split not in SPLITS:
        raise ValueError("Frozen mode requires target_split")
    if target_split is not None and target_split not in SPLITS:
        raise ValueError("Unknown target split")
    bundle = Bundle(root, cfg)
    train_users = set(
        pl.scan_parquet(processed / "impressions_train.parquet")
        .select("user_code")
        .unique()
        .collect()["user_code"]
        .to_list()
    )
    train_articles = set(
        pl.scan_parquet(processed / "candidates_train.parquet")
        .select("news_code")
        .unique()
        .collect()["news_code"]
        .to_list()
    )
    splits = (
        SPLITS if target_split is None else SPLITS[: SPLITS.index(target_split) + 1]
    )
    total_batches, total_events = replay_workload(processed, splits, max_timestamps)
    name = f"{cfg.update_mode}-{target_split or 'all'}" + (
        f"-smoke-{max_timestamps}" if max_timestamps else ""
    )
    out = root / "4_Feature_Engineering" / "artifacts" / name
    if out.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output exists: {out}; use overwrite=True to rebuild this run"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)
    write_json(out / "feature_schema.json", active_schema)
    write_json(
        out / "run_status.json",
        {"status": "running", "full_run": max_timestamps is None},
    )
    write_splits = [target_split] if target_split else list(splits)
    coverage = Counter()
    quality = {}
    generated = Counter()
    selected_counts = Counter()
    feature_writers = {
        s: PartitionWriter(out / f"ranking_features_{s}", cfg.output_batch_rows)
        for s in write_splits
    }
    target_writers = {
        s: PartitionWriter(out / f"labels_{s}", cfg.output_batch_rows)
        for s in write_splits
    }
    reference_schema = pa.schema(
        [
            ("impression_uid", pa.int64()),
            ("user_code", pa.int64()),
            ("timestamp", pa.timestamp("ns")),
            ("state_version", pa.int64()),
            ("history_rows", pa.list_(pa.int32())),
            ("full_history_rows", pa.list_(pa.int32())),
            ("history_mask", pa.list_(pa.bool_())),
            ("history_length", pa.int64()),
            ("retained_length", pa.int64()),
            ("interest_group_count", pa.int64()),
            ("selected_interest_categories", pa.list_(pa.string())),
            ("recent_rows", pa.list_(pa.int32())),
            ("no_supplied_history", pa.bool_()),
            ("has_text", pa.bool_()),
            ("has_entity", pa.bool_()),
        ]
    )
    ref_writers = {
        s: PartitionWriter(
            out / f"request_profiles_{s}", cfg.output_batch_rows, reference_schema
        )
        for s in write_splits
    }
    group_writer = PartitionWriter(
        out / "training_negative_groups",
        cfg.output_batch_rows,
        pa.schema(
            [
                ("impression_uid", pa.int64()),
                ("positive_news_code", pa.int64()),
                ("negative_news_codes", pa.list_(pa.int64())),
            ]
        ),
    )
    state = state_factory(
        bundle, *time_horizon(root, splits), cfg, pair_path=out / "pair_history.sqlite"
    )
    written = 0
    replayed_events = 0
    progress = ReplayProgress(total_batches, total_events)
    print(
        f"Replaying {total_batches:,} timestamp batches. ETA adapts to observed speed "
        "and candidate volume; final checks take additional time.",
        flush=True,
    )
    last_progress = time.perf_counter()
    cutoff = None
    model_columns = active_schema["model_columns"]
    try:
        with measure(root, "ranking_features"):
            for t, requests, events in replay_batches(root, cfg, splits):
                state.begin(requests, t)
                for split in write_splits:
                    req = requests[requests.split.eq(split)]
                    ev = events[events.split.eq(split)]
                    if not len(req):
                        continue
                    f = feature_builder(req, ev, state, bundle, cfg, "logged")
                    generated[split] += len(f)
                    clicked = (
                        ev.set_index(["impression_uid", "news_code"])
                        .clicked.reindex(
                            pd.MultiIndex.from_frame(f[["impression_uid", "news_code"]])
                        )
                        .to_numpy()
                        == 1
                    )
                    new_user = np.fromiter(
                        (int(u) not in train_users for u in f.user_code), bool, len(f)
                    )
                    new_article = np.fromiter(
                        (int(a) not in train_articles for a in f.news_code),
                        bool,
                        len(f),
                    )
                    cohorts = (
                        ("shown_exposures", np.ones(len(f), bool)),
                        ("clicked_exposures", clicked),
                        ("new_users_vs_train", new_user),
                        ("returning_users_vs_train", ~new_user),
                        ("articles_unexposed_in_train", new_article),
                        ("no_causal_history", f.no_causal_history.to_numpy() > 0),
                    )
                    for rep, col in (
                        ("text", "article_has_text"),
                        ("entity", "article_has_entity"),
                        ("text_profile", "no_text_profile"),
                        ("entity_profile", "no_entity_profile"),
                    ):
                        valid = (
                            f[col].to_numpy() > 0
                            if rep in ("text", "entity")
                            else f[col].to_numpy() == 0
                        )
                        for cohort, mask in cohorts:
                            coverage[split, rep, cohort, "n"] += int(mask.sum())
                            coverage[split, rep, cohort, "k"] += int(valid[mask].sum())
                    matrix = f[model_columns].to_numpy()
                    zero = (matrix == 0).sum(axis=0)
                    minimum = matrix.min(axis=0)
                    maximum = matrix.max(axis=0)
                    for j, col in enumerate(model_columns):
                        old = quality.setdefault(
                            (split, col),
                            {
                                "n": 0,
                                "zero": 0,
                                "min": float("inf"),
                                "max": -float("inf"),
                            },
                        )
                        old["n"] += len(f)
                        old["zero"] += int(zero[j])
                        old["min"] = min(old["min"], float(minimum[j]))
                        old["max"] = max(old["max"], float(maximum[j]))
                    ref_writers[split].add(_profile_references(req, state, t))
                    if split == "train":
                        groups, selected = negative_sampler(ev, cfg)
                        group_writer.add(groups)
                        keep = (
                            np.ones(len(f), dtype=bool)
                            if cfg.training_negative_sampling == "epoch"
                            else np.array(
                                [
                                    (int(u), int(a)) in selected
                                    for u, a in zip(f.impression_uid, f.news_code)
                                ]
                            )
                        )
                        selected_counts["all_negatives"] += int(ev.clicked.eq(0).sum())
                        kept_labels = ev.set_index(
                            ["impression_uid", "news_code"]
                        ).clicked.reindex(
                            pd.MultiIndex.from_frame(
                                f.loc[keep, ["impression_uid", "news_code"]]
                            )
                        )
                        selected_counts["kept_unique_negatives"] += int(
                            kept_labels.eq(0).sum()
                        )
                        selected_counts["positive_groups"] += len(groups)
                        selected_counts["negative_group_memberships"] += sum(
                            map(len, groups.negative_news_codes)
                        )
                        f = f.loc[keep].reset_index(drop=True)
                    # Keep each impression contiguous for bounded epoch reads.
                    f = f.sort_values(
                        ["impression_uid", "news_code"], kind="stable"
                    ).reset_index(drop=True)
                    labels = f[["impression_uid", "news_code"]].merge(
                        ev[["impression_uid", "news_code", "clicked"]],
                        validate="one_to_one",
                    )
                    feature_writers[split].add(f)
                    target_writers[split].add(labels)
                feedback = set(splits)
                if cfg.update_mode == "frozen":
                    feedback.discard(target_split)
                state.finish(requests, events, t, feedback)
                written += 1
                replayed_events += len(events)
                cutoff = t
                if written % 10000 == 0 or time.perf_counter() - last_progress >= 30:
                    print(
                        f"{progress.message(written, replayed_events)}; cutoff {t}; "
                        f"emitted {sum(w.count for w in feature_writers.values()):,} pairs",
                        flush=True,
                    )
                    last_progress = time.perf_counter()
                if max_timestamps is not None and written >= max_timestamps:
                    break
            print(progress.message(written, replayed_events), flush=True)
            for writer in [
                *feature_writers.values(),
                *target_writers.values(),
                *ref_writers.values(),
                group_writer,
            ]:
                writer.close()
        coverage_rows = []
        for split, rep, cohort, kind in sorted(coverage):
            if kind != "n":
                continue
            n = coverage[split, rep, cohort, "n"]
            k = coverage[split, rep, cohort, "k"]
            coverage_rows.append(
                {
                    "split": split,
                    "representation": rep,
                    "cohort": cohort,
                    "denominator": n,
                    "valid": k,
                    "fallback": n - k,
                    "valid_fraction": k / max(1, n),
                }
            )
        pd.DataFrame(coverage_rows).to_csv(out / "coverage.csv", index=False)
        qrows = [
            dict(
                split=s,
                feature=f,
                **values,
                zero_fraction=values["zero"] / values["n"],
                nonfinite_count=0,
            )
            for (s, f), values in quality.items()
        ]
        pd.DataFrame(qrows).to_csv(out / "feature_quality.csv", index=False)
        write_json(
            out / "sampling_summary.json",
            dict(
                selected_counts,
                training_negative_sampling=cfg.training_negative_sampling,
                saved_groups_epoch=0,
                saved_groups_role="epoch-zero reference; resample from full training labels each epoch"
                if cfg.training_negative_sampling == "epoch"
                else "fixed training groups",
                discarded_unique_negative_fraction=1
                - selected_counts["kept_unique_negatives"]
                / max(1, selected_counts["all_negatives"]),
            ),
        )
        sources = [
            processed / f"{kind}_{s}.parquet"
            for kind in ("impressions", "candidates")
            for s in splits
        ]
        sources += [
            artifact_dir(root) / f"{x}_manifest.json"
            for x in ("news", "entity", "availability")
        ]
        state.close()
        state = None
        for scratch in out.glob("pair_history.sqlite*"):
            scratch.unlink()
        costs = pd.read_csv(root / "results" / "representation_cost.csv")
        latest = costs[costs.stage.eq("ranking_features")].iloc[-1].to_dict()
        cost = pd.DataFrame(
            [
                dict(
                    latest,
                    run=name,
                    artifact_bytes=sum(
                        p.stat().st_size for p in out.rglob("*.parquet")
                    ),
                    emitted_rows=sum(w.count for w in feature_writers.values()),
                    timestamp_batches=written,
                )
            ]
        )
        cost.to_csv(out / "feature_cost.csv", index=False)
        outputs = list(out.rglob("*.parquet")) + [
            out / "feature_schema.json",
            out / "coverage.csv",
            out / "feature_quality.csv",
            out / "sampling_summary.json",
            out / "feature_cost.csv",
        ]
        write_json(
            out / "run_manifest.json",
            manifest(
                root,
                cfg,
                sources,
                outputs,
                full_run=max_timestamps is None,
                training_candidate_pool="complete"
                if cfg.training_negative_sampling == "epoch"
                else "sampled",
                saved_groups_epoch=0,
                last_scored_timestamp=cutoff,
                timestamp_batches=written,
                rows_before_sampling=dict(generated),
                rows_written={s: w.count for s, w in feature_writers.items()},
                pair_state_backend="sqlite scratch; not resumable",
                half_life_version="completed disjoint 6h windows; Beta(1,1) smoothing",
                history_order="verified sequence"
                if cfg.snapshot_order_verified
                else "deterministic snapshot subset; no recency claim",
            ),
        )
        write_json(
            out / "run_status.json",
            {
                "status": "complete",
                "full_run": max_timestamps is None,
                "last_scored_timestamp": cutoff,
            },
        )
        if max_timestamps is None:
            upsert_report(
                root, "representation_coverage.csv", pd.DataFrame(coverage_rows), name
            )
            upsert_report(root, "feature_quality.csv", pd.DataFrame(qrows), name)
            upsert_report(root, "feature_cost.csv", cost, name)
    except Exception as exc:
        write_json(
            out / "run_status.json",
            {"status": "failed", "error": str(exc), "last_finished_timestamp": cutoff},
        )
        raise
    finally:
        if state is not None:
            state.close()
    return out


def profile_preview(
    root,
    cfg=DEFAULT_SETTINGS,
    timestamp_batches=3,
    users_per_batch=2,
    *,
    state_factory=TemporalState,
):
    """Small chronological profile preview without persisting or changing artifacts."""
    from itertools import islice

    bundle = Bundle(root, cfg)
    state = state_factory(bundle, *time_horizon(root), cfg)
    examples = []
    try:
        for t, requests, events in islice(replay_batches(root, cfg), timestamp_batches):
            state.begin(requests, t)
            for request in requests.head(users_per_batch).itertuples(index=False):
                profile = state.profile(request.user_code, t)
                examples.append(
                    {
                        "timestamp": t,
                        "impression_uid": request.impression_uid,
                        "user_code": request.user_code,
                        "history_length": profile["history_length"],
                        "retained_length": profile["retained_length"],
                        "valid_interests": profile["group_count"],
                        "has_text": profile["has_text"],
                        "has_entity": profile["has_entity"],
                        "has_recent": profile["has_recent"],
                    }
                )
            state.finish(requests, events, t, {"train", "validation", "test"})
    finally:
        state.close()
    return pd.DataFrame(examples)


def verify_feature_parity(
    root,
    cfg=DEFAULT_SETTINGS,
    *,
    state_factory=TemporalState,
    feature_builder=None,
    schema_builder=None,
):
    """Same first-request features for logged pairs and their unlabelled equivalent."""
    if not callable(feature_builder) or not callable(schema_builder):
        raise TypeError("Pass the notebook feature_builder and schema_builder")
    bundle = Bundle(root, cfg)
    state = state_factory(bundle, *time_horizon(root), cfg)
    try:
        t, requests, logged = next(replay_batches(root, cfg))
        state.begin(requests, t)
        a = feature_builder(requests, logged, state, bundle, cfg, "logged")
        b = feature_builder(
            requests,
            logged[["impression_uid", "news_code"]],
            state,
            bundle,
            cfg,
            "retrieved",
        )
        columns = validate_feature_schema(schema_builder(cfg))["model_columns"]
        pd.testing.assert_frame_equal(a[columns], b[columns])
    finally:
        state.close()
