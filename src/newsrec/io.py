from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .common import DEFAULT_SETTINGS, normalised_mean, sha256, validate_feature_schema


def open_feature_run(root, run_dir, *, allow_smoke=False, verify_hashes=True):
    """Return schema/manifest and Arrow datasets without loading all features in RAM."""
    root = Path(root)
    path = Path(run_dir)
    status = json.loads((path / "run_status.json").read_text())
    meta = json.loads((path / "run_manifest.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Feature run is incomplete")
    if not meta["full_run"] and (not allow_smoke):
        raise ValueError("Smoke output is not a full training/evaluation dataset")
    if verify_hashes:
        for rel, expected in meta["outputs"].items():
            if sha256(root / rel) != expected:
                raise ValueError(f"Feature artifact mismatch: {rel}")
        for rel, expected in meta["sources"].items():
            if not (root / rel).exists() or sha256(root / rel) != expected:
                raise ValueError(f"Input or representation dependency changed: {rel}")
    schema = json.loads((path / "feature_schema.json").read_text())
    datasets = {}
    for split in meta["rows_written"]:
        p = path / f"ranking_features_{split}"
        if list(p.glob("*.parquet")):
            datasets[split] = ds.dataset(p, format="parquet")
    return {"schema": schema, "manifest": meta, "datasets": datasets}


def reconstruct_profile(reference, bundle):
    """Recover full pooled vectors and capped sequence inputs from a reference.

    Long-term supplied profiles use the separate supplied-profile matrix. No labels
    or future user state are needed for this reconstruction.
    """
    history = np.asarray(reference["history_rows"], np.int32)
    mask = np.asarray(reference["history_mask"], bool)
    history = history[mask]
    # v1 references only stored capped rows; preserve their original semantics.
    full_history = np.asarray(reference.get("full_history_rows", history), np.int32)
    text, _ = normalised_mean(bundle.text, full_history)
    entity, _ = normalised_mean(bundle.entity, full_history)
    recent, _ = normalised_mean(bundle.text, reference["recent_rows"])
    categories = bundle.meta.category.fillna("<MISSING>").astype(str).to_numpy()
    interests = []
    if "full_history_rows" not in reference and "interest_vectors" in reference:
        # Compatibility with the upstream centroid-only reference format.
        interests = reference["interest_vectors"]
    else:
        for category in reference["selected_interest_categories"]:
            rows = full_history[categories[full_history] == category]
            vector, count = normalised_mean(bundle.title, rows)
            if count and np.linalg.norm(vector) > 0:
                interests.append(vector)
    return {
        "text": text,
        "entity": entity,
        "recent_text": recent,
        "interests": np.asarray(interests, np.float32).reshape(-1, bundle.cfg.text_dim),
        "history_rows": history,
        "full_history_rows": full_history,
    }


def feature_run(
    root,
    cfg=DEFAULT_SETTINGS,
    *,
    target_split=None,
    max_timestamps=None,
    rebuild=False,
    schema_builder=None,
    **build_options,
):
    """Load a verified compatible run, or build it with the same dataset contract."""
    from .pipeline import build_ranking_dataset

    if not callable(schema_builder):
        raise TypeError("Pass schema_builder=feature_schema from the feature notebook")
    requested_schema = validate_feature_schema(schema_builder(cfg))
    if max_timestamps is not None and max_timestamps < 1:
        raise ValueError("max_timestamps must be positive or None")
    name = f"{cfg.update_mode}-{target_split or 'all'}"
    if build_options.get("optimized", True):
        name += "-optimized"
    if max_timestamps is not None:
        name += f"-smoke-{max_timestamps}"
    run_dir = Path(root) / "4_Feature_Engineering/artifacts" / name
    if run_dir.exists() and not rebuild:
        cached = open_feature_run(root, run_dir, allow_smoke=max_timestamps is not None)
        if cached["manifest"]["config_hash"] != cfg.fingerprint():
            raise ValueError("Saved feature settings differ. Set REBUILD=True.")
        if cached["schema"] != requested_schema:
            raise ValueError(
                "Saved feature selection/definitions differ. Set REBUILD=True to rebuild the selected run."
            )
        print("Loaded saved feature run:", run_dir)
        return run_dir
    return build_ranking_dataset(
        root,
        cfg,
        target_split=target_split,
        max_timestamps=max_timestamps,
        overwrite=rebuild,
        schema_builder=schema_builder,
        **build_options,
    )


def feature_sample(run_dir, split="train", rows=8):
    """Read only a small Arrow batch, never the whole feature dataset."""
    if split not in ("train", "validation", "test") or rows < 1:
        raise ValueError("Choose a valid split and positive sample size")
    parts = list((Path(run_dir) / f"ranking_features_{split}").glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    batch = next(
        ds.dataset(parts, format="parquet").scanner(batch_size=rows).to_batches(), None
    )
    if batch is None:
        return pd.DataFrame()
    result = batch.to_pandas()
    schema = json.loads((Path(run_dir) / "feature_schema.json").read_text())
    if not np.isfinite(result[schema["model_columns"]].to_numpy()).all():
        raise ValueError("Nonfinite model features in sample")
    return result
