"""Small, label-free representation diagnostics and bounded-memory coverage scans."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import SPLITS


def usage_coverage(root, cfg):
    """Count missing vectors per slate entry, click and supplied-history entry.

    Click labels are used only to report coverage, never to fit representations.
    History counts include each user once per split; repeated clicks are retained.
    """
    root = Path(root)
    out = root / "3_Representations/artifacts"
    processed = root / "1_Data_Preparation/processed"
    index = pd.read_parquet(out / "article_index.parquet")
    codes = pd.Index(index.news_code)
    text = np.load(out / "text64.npy", mmap_mode="r")
    entity = np.load(out / "article_entity100.npy", mmap_mode="r")
    has_text = np.linalg.norm(text, axis=1) > 0
    has_entity = np.linalg.norm(entity, axis=1) > 0
    rows = []
    for split in SPLITS:
        counts = {name: np.zeros(5, np.int64) for name in ("slate", "click", "history")}

        def add(name, article_codes, counts=counts):
            positions = codes.get_indexer(article_codes)
            if (positions < 0).any():
                raise ValueError("Coverage input contains an unknown article")
            article_rows = index.article_row.to_numpy()[positions]
            t, e = has_text[article_rows], has_entity[article_rows]
            counts[name] += [
                len(t),
                (t & e).sum(),
                (t & ~e).sum(),
                (~t & e).sum(),
                (~t & ~e).sum(),
            ]

        # Scan each candidate file once, in bounded batches.
        for batch in pq.ParquetFile(
            processed / f"candidates_{split}.parquet"
        ).iter_batches(
            batch_size=cfg.parquet_batch_rows, columns=["news_code", "clicked"]
        ):
            frame = batch.to_pandas()
            add("slate", frame.news_code)
            add("click", frame.loc[frame.clicked.eq(1), "news_code"])
        seen = set()
        for batch in pq.ParquetFile(
            processed / f"impressions_{split}.parquet"
        ).iter_batches(
            batch_size=cfg.parquet_batch_rows, columns=["user_code", "history"]
        ):
            history_codes = []
            for row in (
                batch.to_pandas().drop_duplicates("user_code").itertuples(index=False)
            ):
                if row.user_code in seen:
                    continue
                seen.add(row.user_code)
                history = "" if pd.isna(row.history) else str(row.history)
                history_codes.extend(
                    int(token.removeprefix("N")) for token in history.split()
                )
            add("history", history_codes)
        for cohort, values in counts.items():
            total, both, text_only, entity_only, neither = map(int, values)
            rows.append(
                {
                    "split": split,
                    "cohort": cohort,
                    "entries": total,
                    "both": both,
                    "text_only": text_only,
                    "entity_only": entity_only,
                    "neither": neither,
                    "text_coverage": (both + text_only) / total if total else np.nan,
                    "entity_coverage": (both + entity_only) / total
                    if total
                    else np.nan,
                }
            )
    report = pd.DataFrame(rows)
    report.to_csv(out / "usage_coverage.csv", index=False)
    return report


def view_neighbors(root, queries=3, count=3):
    """Compare deterministic queries across title, abstract and fused coordinates."""
    out = Path(root) / "3_Representations/artifacts"
    index = pd.read_parquet(out / "article_index.parquet").set_index("article_row")
    fused = np.load(out / "text64.npy", mmap_mode="r")
    valid = np.flatnonzero(np.linalg.norm(fused, axis=1) > 0)
    if not len(valid):
        return pd.DataFrame()
    chosen = valid[np.linspace(0, len(valid) - 1, min(queries, len(valid)), dtype=int)]
    results = []
    for view in ("title64", "abstract64", "text64"):
        vectors = np.load(out / f"{view}.npy", mmap_mode="r")
        available = np.linalg.norm(vectors, axis=1) > 0
        for query in chosen:
            if not available[query]:
                continue
            scores = vectors @ vectors[query]
            candidates = np.flatnonzero(available & (np.arange(len(vectors)) != query))
            nearest = candidates[np.argsort(-scores[candidates], kind="stable")[:count]]
            for rank, article in enumerate(nearest, 1):
                results.append(
                    {
                        "view": view,
                        "query_code": index.loc[query, "news_code"],
                        "query_title": index.loc[query, "title"],
                        "rank": rank,
                        "neighbor_title": index.loc[article, "title"],
                        "query_category": index.loc[query, "category"],
                        "neighbor_category": index.loc[article, "category"],
                        "cosine": float(scores[article]),
                    }
                )
    return pd.DataFrame(results)


def check_temporal_profiles(state_factory, cfg):
    """Exercise same-time isolation, delayed feedback and interests beyond the cap."""
    from dataclasses import replace
    from types import SimpleNamespace

    cfg = replace(cfg, history_cap=2, interest_count=2)
    title = np.zeros((6, cfg.text_dim), np.float32)
    title[1:4, 0] = 1  # Three sports clicks dominate the full history.
    title[4:, 1] = 1  # Only two finance rows survive the sequence cap.
    entity = np.zeros((6, cfg.entity_dim), np.float32)
    entity[1:, 0] = 1
    bundle = SimpleNamespace(
        n=6,
        title=title,
        text=title,
        entity=entity,
        index=pd.DataFrame(
            {
                "category": ["sports"] * 3 + ["finance"] * 2,
                "subcategory": ["example"] * 5,
            }
        ),
        lookup={i: i for i in range(1, 6)},
        rows=lambda codes: np.asarray(list(codes), np.int32),
    )
    t = pd.Timestamp("2020-01-01")
    requests = pd.DataFrame(
        {
            "impression_uid": [1, 2],
            "user_code": [1, 1],
            "timestamp": [t, t],
            "history": ["N1 N2 N3 N4 N5"] * 2,
        }
    )
    events = pd.DataFrame(
        {
            "impression_uid": [1],
            "user_code": [1],
            "timestamp": [t],
            "news_code": [4],
            "clicked": [1],
            "split": ["train"],
        }
    )
    checks = []
    pre_feedback = None
    for label in (0, 1):
        state = state_factory(bundle, t, t + pd.Timedelta(hours=1), cfg)
        try:
            state.begin(requests, t)
            before = state.profile(1, t)
            assert before["selected_interest_categories"] == ["sports", "finance"]
            assert before["retained_length"] == 2 and before["history_length"] == 5
            np.testing.assert_allclose(before["interests"][0], title[1])
            # Both requests at t see the same profile before any labels are applied.
            np.testing.assert_array_equal(before["text"], state.profile(1, t)["text"])
            if pre_feedback is not None:
                np.testing.assert_array_equal(before["interests"], pre_feedback)
            pre_feedback = before["interests"].copy()
            state.finish(requests, events.assign(clicked=label), t, {"train"})
            next_time = t + pd.Timedelta(minutes=1)
            state.begin(
                requests.assign(timestamp=next_time, impression_uid=[3, 4]), next_time
            )
            after = state.profile(1, next_time)
            assert after["history_length"] == 5 + label
            assert after["state_version"] == label
            checks.append(
                {
                    "feedback_label": label,
                    "pre_feedback_clicks": before["history_length"],
                    "next_timestamp_clicks": after["history_length"],
                    "interests_beyond_cap": "passed",
                    "delayed_feedback": "passed",
                }
            )
        finally:
            state.close()
    return pd.DataFrame(checks)
