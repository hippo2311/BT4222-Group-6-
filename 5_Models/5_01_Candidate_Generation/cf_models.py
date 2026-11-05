"""Shared candidate-generation data access and user-item matrix.

The static binary user-item matrix is reused by the item-item CF baseline and
the neural candidate generators. Model-specific neighbour construction,
scoring and candidate selection live in the corresponding notebooks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


@dataclass(frozen=True)
class CFContext:
    n_users: int
    n_items: int
    news_code: np.ndarray
    available_from: np.ndarray
    popularity: np.ndarray


def load_context(root):
    root = Path(root)
    processed = root / "1_Data_Preparation" / "processed"
    artifacts = root / "3_Representations" / "artifacts"

    index = pl.read_parquet(
        artifacts / "article_index.parquet", columns=["article_row", "news_code"]
    ).sort("article_row")
    expected = np.arange(1, index.height + 1, dtype=np.int64)
    if not np.array_equal(index["article_row"].to_numpy(), expected):
        raise ValueError("article_row must be contiguous from 1")

    n_items = index.height + 1
    news_code = np.zeros(n_items, dtype=np.int32)
    news_code[1:] = index["news_code"].to_numpy().astype(np.int32)

    availability = pl.read_parquet(
        artifacts / "article_availability.parquet",
        columns=["news_code", "available_from"],
    )
    joined = index.join(availability, on="news_code", how="left").sort("article_row")
    if joined["available_from"].null_count():
        raise ValueError("Missing available_from for some articles")
    available_from = np.empty(n_items, dtype="datetime64[ns]")
    available_from[0] = np.datetime64("1970-01-01")
    available_from[1:] = joined["available_from"].to_numpy().astype(
        "datetime64[ns]"
    )

    catalogue = pl.read_parquet(
        processed / "news_catalogue.parquet", columns=["news_code", "exposures_train"]
    )
    with_popularity = index.join(catalogue, on="news_code", how="left").sort(
        "article_row"
    )
    if with_popularity["exposures_train"].null_count():
        raise ValueError("Missing exposures_train for some articles")
    popularity = np.zeros(n_items, dtype=np.float32)
    popularity[1:] = with_popularity["exposures_train"].to_numpy().astype(np.float32)
    popularity[0] = -1.0

    users = pl.read_parquet(processed / "users.parquet", columns=["user_code"])
    codes = users["user_code"].to_numpy()
    if codes.size == 0:
        raise ValueError("users table is empty")
    if codes.min() < 0:
        raise ValueError("user_code must be non-negative")
    # Row 0 is reserved for padding/unknown, so the embedding matrix needs
    # one more row than the largest assigned user_code.
    n_users = int(codes.max()) + 1

    return CFContext(
        n_users=int(n_users),
        n_items=int(n_items),
        news_code=news_code,
        available_from=available_from,
        popularity=popularity,
    )


def _load_profiles(root, split):
    root = Path(root)
    features = root / "4_Feature_Engineering" / "artifacts" / "prequential-all"
    parts = sorted((features / f"request_profiles_{split}").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No request profiles for {split}")
    frame = pl.read_parquet(
        parts,
        columns=["impression_uid", "user_code", "timestamp", "full_history_rows"],
    )
    if frame["impression_uid"].n_unique() != frame.height:
        raise ValueError("Duplicate impression_uid in request profiles")
    if frame["full_history_rows"].null_count():
        raise ValueError("Null full_history_rows in request profiles")
    return frame.sort("timestamp", maintain_order=True)


def _train_pairs(root, ctx):
    root = Path(root)
    profiles = _load_profiles(root, "train")
    latest = (
        profiles.sort(["user_code", "timestamp"])
        .group_by("user_code", maintain_order=True)
        .agg(pl.col("full_history_rows").last())
    )
    history = (
        latest.explode("full_history_rows")
        .rename({"full_history_rows": "article_row"})
        .drop_nulls("article_row")
        .select("user_code", "article_row")
        .with_columns(
            pl.col("user_code").cast(pl.Int32),
            pl.col("article_row").cast(pl.Int32),
        )
    )

    index = pl.read_parquet(
        root / "3_Representations" / "artifacts" / "article_index.parquet",
        columns=["article_row", "news_code"],
    )
    clicked = (
        pl.read_parquet(
            root / "1_Data_Preparation" / "processed" / "candidates_train.parquet",
            columns=["user_code", "news_code", "clicked"],
        )
        .filter(pl.col("clicked") == 1)
        .select("user_code", "news_code")
        .unique()
        .join(index, on="news_code", how="left")
        .select("user_code", "article_row")
        .with_columns(
            pl.col("user_code").cast(pl.Int32),
            pl.col("article_row").cast(pl.Int32),
        )
    )
    if clicked["article_row"].null_count():
        raise ValueError("Clicked article is missing from article_index")

    pairs = (
        pl.concat([history, clicked])
        .unique()
        .with_columns(
            pl.col("user_code").cast(pl.Int32),
            pl.col("article_row").cast(pl.Int32),
        )
    )
    if pairs["user_code"].min() < 0 or pairs["user_code"].max() >= ctx.n_users:
        raise ValueError("Interaction user_code is outside the matrix")
    if (
        pairs["article_row"].min() < 1
        or pairs["article_row"].max() >= ctx.n_items
    ):
        raise ValueError("Interaction article_row is outside the matrix")
    return pairs


def matrix_from_pairs(pairs, n_users, n_items):
    if pairs.height == 0:
        return sparse.csr_matrix((n_users, n_items), dtype=np.float32)
    rows = pairs["user_code"].to_numpy().astype(np.int64)
    cols = pairs["article_row"].to_numpy().astype(np.int64)
    matrix = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix


def build_interaction_matrix(root, ctx, *, rebuild=False):
    output = Path(root) / "5_Models" / "5_01_Candidate_Generation" / "artifacts" / "cf"
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "user_item_train.npz"
    manifest_path = output / "matrix_manifest.json"

    if matrix_path.exists() and manifest_path.exists() and not rebuild:
        matrix = sparse.load_npz(matrix_path)
        manifest = json.loads(manifest_path.read_text())
        if matrix.shape != (ctx.n_users, ctx.n_items):
            raise ValueError("Saved CF matrix shape no longer matches the catalogue")
        return matrix, manifest

    pairs = _train_pairs(root, ctx)
    matrix = matrix_from_pairs(pairs, ctx.n_users, ctx.n_items)
    sparse.save_npz(matrix_path, matrix, compressed=True)
    pairs.write_parquet(output / "user_item_train.parquet")
    manifest = {
        "version": "cf-static-v1",
        "created_at": _now(),
        "cutoff": str(
            pl.scan_parquet(
                Path(root) / "1_Data_Preparation" / "processed" / "impressions_train.parquet"
            )
            .select(pl.col("timestamp").max())
            .collect()
            .item()
        ),
        "rows": int(ctx.n_users),
        "columns": int(ctx.n_items),
        "nonzero": int(matrix.nnz),
        "values": "binary",
        "source": "latest request_profiles_train history plus candidates_train clicks",
    }
    _write_json(manifest_path, manifest)
    return matrix, manifest


def resolve_device(device="auto"):
    if device == "cpu":
        return "cpu"
    import torch

    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise ValueError("device must be auto, cpu or cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return "cuda"
