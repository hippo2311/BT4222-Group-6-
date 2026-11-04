"""Shared tuning and early-stopping helpers for the neural candidate generators.

The notebooks use the same protocol: hold out the last 10% of training
timestamps, score the exposed tuning slates each epoch, and stop when mean slate
NDCG@k stops improving. ``k`` is the largest number of clicks in any training
impression, so every impression's ideal DCG can include all of its positives.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import polars as pl

from newsrec.sanity import slate_metrics


class EpochNegativeSampler:
    """Precompute impression slates once, then redraw negatives per epoch.

    Draws are identical to ``newsrec.ranking.sample_training_groups`` because the
    per-impression seed key and RNG calls are preserved. Reusing the prepared
    slates avoids regrouping the full training pool on every epoch.
    """

    def __init__(self, rows, settings):
        if not isinstance(rows, pd.DataFrame):
            raise TypeError("rows must be a pandas DataFrame")
        missing = {"impression_uid", "news_code", "clicked"} - set(rows.columns)
        if missing:
            raise ValueError(f"rows missing columns: {sorted(missing)}")
        prepared = []
        for uid, slate in rows.groupby("impression_uid", sort=True):
            if slate["news_code"].duplicated().any() or not slate[
                "clicked"
            ].isin([0, 1]).all():
                raise ValueError("Sampling requires unique binary-labelled logged pairs")
            positives = sorted(
                map(int, slate.loc[slate["clicked"].eq(1), "news_code"])
            )
            negatives = np.sort(
                slate.loc[slate["clicked"].eq(0), "news_code"].to_numpy(np.int64)
            )
            if not positives:
                raise ValueError("Logged training impression has no positives")
            prepared.append((int(uid), positives, negatives))
        self._prepared = prepared
        self._seed = int(settings.seed)
        self._negative_count = int(settings.negative_count)

    def draw(self, epoch, *, uids=None):
        """Return (groups, selected pairs) for one nonnegative epoch."""
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, (int, np.integer))
            or epoch < 0
        ):
            raise ValueError("epoch must be a nonnegative integer")
        allowed = None if uids is None else set(map(int, uids))
        groups = []
        selected = set()
        for uid, positives, negatives in self._prepared:
            if allowed is not None and uid not in allowed:
                continue
            for positive in positives:
                seed_key = f"{self._seed}:{uid}:{positive}"
                if epoch:
                    seed_key += f":epoch:{int(epoch)}"
                seed = int.from_bytes(
                    hashlib.sha256(seed_key.encode()).digest()[:8], "little"
                )
                rng = np.random.default_rng(seed)
                sampled = sorted(
                    map(
                        int,
                        rng.choice(
                            negatives,
                            min(self._negative_count, len(negatives)),
                            replace=False,
                        ),
                    )
                )
                groups.append(
                    {
                        "impression_uid": uid,
                        "positive_news_code": positive,
                        "negative_news_codes": sampled,
                    }
                )
                selected.add((uid, positive))
                selected.update((uid, n) for n in sampled)
        return (
            pd.DataFrame(
                groups,
                columns=[
                    "impression_uid",
                    "positive_news_code",
                    "negative_news_codes",
                ],
            ),
            selected,
        )


def max_clicks_per_impression(candidates_path):
    """Return the largest click count per impression in a candidate table."""
    value = (
        pl.scan_parquet(candidates_path)
        .group_by("impression_uid")
        .agg(pl.col("clicked").sum().alias("clicks"))
        .select(pl.col("clicks").max())
        .collect()
        .item()
    )
    if value is None:
        raise ValueError("candidate table is empty")
    return int(value)


def chronological_cutoff(timestamps, quantile=0.9):
    """Return the integer nanosecond cutoff at a timestamp quantile."""
    values = np.asarray(timestamps, dtype="datetime64[ns]").astype(np.int64)
    if values.size == 0:
        raise ValueError("timestamps is empty")
    return int(np.quantile(values, quantile))


def chronological_mask(timestamps, cutoff_ns):
    """Return a boolean mask selecting timestamps at or before the cutoff."""
    values = np.asarray(timestamps, dtype="datetime64[ns]").astype(np.int64)
    return values <= int(cutoff_ns)


def tuning_slates(
    candidates_path,
    cutoff_ns,
    article_index,
    profile_index,
    extra_columns=(),
    max_impressions=None,
    seed=0,
):
    """Load post-cutoff exposed slates and attach article and profile keys.

    ``max_impressions`` optionally samples whole slates for bounded runs.
    """
    columns = [
        "impression_uid",
        "timestamp",
        "news_code",
        "clicked",
        *extra_columns,
    ]
    frame = pl.read_parquet(candidates_path, columns=columns)
    timestamps = frame["timestamp"].to_numpy().astype("datetime64[ns]").astype(
        np.int64
    )
    frame = frame.filter(pl.Series("ts_ns", timestamps) > int(cutoff_ns))
    if frame.height == 0:
        raise ValueError("No tuning rows after the chronological cutoff")
    if max_impressions is not None:
        if max_impressions < 1:
            raise ValueError("max_impressions must be positive")
        uids = frame["impression_uid"].unique()
        if uids.len() > max_impressions:
            uids = uids.sample(n=max_impressions, seed=seed, shuffle=True)
            frame = frame.filter(
                pl.col("impression_uid").is_in(uids.implode())
            )
    frame = frame.join(article_index, on="news_code", how="left")
    frame = frame.join(profile_index, on="impression_uid", how="left")
    return frame


def mean_slate_metrics(impression_uids, clicked, scores, k):
    """Return mean per-impression AUC and NDCG@k for scored slates."""
    if k < 1:
        raise ValueError("k must be positive")
    frame = pl.DataFrame(
        {
            "impression_uid": np.asarray(impression_uids, dtype=np.int64),
            "clicked": np.asarray(clicked, dtype=np.float32),
            "score": np.asarray(scores, dtype=np.float32),
        }
    )
    aucs = []
    ndcgs = []
    for _, group in frame.group_by("impression_uid", maintain_order=True):
        auc, ndcg = slate_metrics(
            group["clicked"].to_numpy(),
            group["score"].to_numpy(),
            k=k,
        )
        if auc is not None:
            aucs.append(auc)
        ndcgs.append(ndcg)
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else float("nan")
    return mean_auc, mean_ndcg


class EarlyStopper:
    """Track the best epoch by a higher-is-better tuning score."""

    def __init__(self, patience=3, min_delta=1e-6):
        if patience < 1:
            raise ValueError("patience must be positive")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score = -np.inf
        self.best_epoch = 0
        self.best_state = None
        self.wait = 0

    @property
    def should_stop(self):
        return self.best_state is not None and self.wait >= self.patience

    def step(self, score, model, epoch):
        """Record an epoch and return True when the patience budget is spent."""
        if score > self.best_score + self.min_delta:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            self.wait = 0
        else:
            self.wait += 1
        return self.should_stop

    def restore(self, model):
        """Load the best recorded state into a model."""
        if self.best_state is None:
            raise RuntimeError("No model state was selected")
        model.load_state_dict(self.best_state)
        return model
