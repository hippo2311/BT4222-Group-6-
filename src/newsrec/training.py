"""Bounded reads and fresh exposed-negative draws for epoch-based rankers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import Settings
from .ranking import sample_training_groups


def iter_training_epoch(run_dir, epoch, *, columns=None, allow_smoke=False):
    """Yield (labelled features, positive/negative groups) for one epoch.

    Each batch contains complete impressions. All positives and the unique union
    of up to K exposed negatives per positive are retained. Use the groups for
    grouped losses; use the frame for pointwise losses. Only schema model columns
    are loaded by default, alongside keys and clicked. Call once per epoch with a
    new zero-based epoch number. Same seed/epoch reproduces the same draw.

    Verify artifact hashes once with open_feature_run before starting training.
    This reader checks completion and the full-pool contract on every call, and
    reads training files only. It does not shuffle batches or fit preprocessing.
    """
    if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    run = Path(run_dir)
    status = json.loads((run / "run_status.json").read_text())
    meta = json.loads((run / "run_manifest.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Feature run is incomplete")
    if not meta["full_run"] and not allow_smoke:
        raise ValueError("Smoke output requires allow_smoke=True")
    if meta.get("training_candidate_pool") != "complete":
        raise ValueError(
            "Epoch sampling needs the complete training pool. Rebuild feature engineering."
        )
    cfg = Settings(**meta["config"])
    schema = json.loads((run / "feature_schema.json").read_text())
    columns = list(schema["model_columns"] if columns is None else columns)
    if set(columns) - set(schema["model_columns"]):
        raise ValueError("Select only model_columns from the feature schema")
    parts = sorted((run / "ranking_features_train").glob("*.parquet"))
    if not parts:
        raise ValueError("Run contains no training feature partitions")
    keys = ["impression_uid", "news_code"]

    def sample(frame):
        groups, selected = sample_training_groups(frame, cfg, epoch=int(epoch))
        index = pd.MultiIndex.from_frame(frame[keys])
        keep = index.isin(pd.MultiIndex.from_tuples(sorted(selected), names=keys))
        return frame.loc[keep].reset_index(drop=True), groups

    pending = None
    for part in parts:
        features = pq.read_table(part, columns=keys + columns).to_pandas()
        labels = pq.read_table(run / "labels_train" / part.name).to_pandas()
        frame = features.merge(
            labels,
            on=keys,
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
        if len(labels) != len(features) or not frame._merge.eq("both").all():
            raise ValueError("Feature/label keys do not match")
        frame = frame.drop(columns="_merge")
        if pending is not None:
            frame = pd.concat([pending, frame], ignore_index=True)
        if frame.empty:
            continue
        # The writer keeps each impression contiguous, including across parts.
        tail = frame.impression_uid.iloc[-1]
        pending = frame.loc[frame.impression_uid.eq(tail)].copy()
        ready = frame.loc[frame.impression_uid.ne(tail)]
        if not ready.empty:
            yield sample(ready)
    if pending is not None and not pending.empty:
        yield sample(pending)
