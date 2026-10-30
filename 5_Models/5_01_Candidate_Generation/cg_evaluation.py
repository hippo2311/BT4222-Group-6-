"""Reusable Recall@M evaluation for candidate-generation models.

Every candidate table uses the same format:

    impression_uid: Int64
    news_code: Int32
    cutoff: Int16, 1..C

Each impression must contain exactly C candidates with unique cutoffs and unique
news codes. Recall ignores the order; ``cutoff`` only defines nested Top-M sets.

Example:
    from cg_evaluation import CandidateEvaluator

    evaluator = CandidateEvaluator(split="validation")
    results = evaluator.evaluate(
        recommendations,
        model_name="popularity",
        m_values=(50, 100, 200, 500),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_M_VALUES = (50, 100, 200, 500)
REQUIRED_COLUMNS = ("impression_uid", "news_code", "cutoff")
SPLITS = ("train", "validation", "test")


@dataclass
class _PreparedRecommendations:
    # Validated standard-format candidates plus the common candidate depth.
    frame: pl.DataFrame
    max_cutoff: int


def _normalise_m_values(m_values):
    if isinstance(m_values, int):
        values = (m_values,)
    else:
        values = tuple(dict.fromkeys(int(value) for value in m_values))
    if not values or any(value < 1 for value in values):
        raise ValueError("m_values must contain positive integers")
    return values


class CandidateEvaluator:
    """Load one split once, then evaluate any number of candidate models."""

    def __init__(self, root=None, split="validation"):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.root = (
            Path(root).resolve()
            if root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.split = split
        self.processed = self.root / "1_Data_Preparation" / "processed"
        self.artifacts = self.root / "3_Representations" / "artifacts"
        self._impressions = None
        self._positives = None
        self._relevant_totals = None
        self._availability = None
        self._eligible_pool_mean = None
        self._eligible_click_coverage = None

    @property
    def impressions(self):
        # Every request in the split is part of the evaluation universe.
        if self._impressions is None:
            frame = pl.read_parquet(
                self.processed / f"impressions_{self.split}.parquet",
                columns=["impression_uid", "user_code", "timestamp"],
            )
            if frame["impression_uid"].n_unique() != frame.height:
                raise ValueError("Impressions contain duplicate impression_uid values")
            if frame["impression_uid"].null_count() or frame["timestamp"].null_count():
                raise ValueError("Impressions contain null keys or timestamps")
            self._impressions = frame
        return self._impressions

    @property
    def positives(self):
        # Only observed clicks are ground truth; unshown articles stay unknown.
        if self._positives is None:
            frame = (
                pl.scan_parquet(
                    self.processed / f"candidates_{self.split}.parquet"
                )
                .filter(pl.col("clicked") == 1)
                .select("impression_uid", "timestamp", "news_code")
                .collect()
            )
            if frame["impression_uid"].null_count() or frame["news_code"].null_count():
                raise ValueError("Clicked ground truth contains null keys")
            duplicates = (
                frame.group_by(["impression_uid", "news_code"])
                .len()
                .filter(pl.col("len") > 1)
                .height
            )
            if duplicates:
                raise ValueError("Clicked ground truth contains duplicate pairs")
            self._positives = frame
        return self._positives

    @property
    def relevant_totals(self):
        # Per-impression denominator used by the impression-mean recall.
        if self._relevant_totals is None:
            self._relevant_totals = self.positives.group_by("impression_uid").agg(
                pl.len().alias("relevant")
            )
        return self._relevant_totals

    @property
    def availability(self):
        # Earliest observable appearance; used only to filter future candidates.
        if self._availability is None:
            frame = pl.read_parquet(
                self.artifacts / "article_availability.parquet",
                columns=["news_code", "available_from"],
            )
            if frame["news_code"].n_unique() != frame.height:
                raise ValueError("Availability contains duplicate news_code values")
            if frame["available_from"].null_count():
                raise ValueError("Availability contains null available_from values")
            self._availability = frame
        return self._availability

    @property
    def mean_eligible_pool(self):
        # Count the eligible catalogue per timestamp, then weight by impressions.
        if self._eligible_pool_mean is None:
            available = self.availability["available_from"].sort().to_numpy()
            timestamps = self.impressions["timestamp"].unique()
            counts = np.searchsorted(
                available, timestamps.to_numpy(), side="right"
            )
            per_timestamp = pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "eligible_pool": counts.astype(np.int32, copy=False),
                }
            )
            self._eligible_pool_mean = float(
                self.impressions.select("timestamp")
                .join(per_timestamp, on="timestamp", how="left")
                .select(pl.col("eligible_pool").mean())
                .item()
            )
        return self._eligible_pool_mean

    @property
    def eligible_click_coverage(self):
        # Report eligibility loss without removing clicks from the denominator.
        if self._eligible_click_coverage is None:
            joined = self.positives.join(
                self.availability, on="news_code", how="left"
            )
            if joined["available_from"].null_count():
                raise ValueError("Clicked ground truth contains unknown news_code values")
            covered = joined.filter(
                pl.col("available_from") <= pl.col("timestamp")
            ).height
            self._eligible_click_coverage = (
                covered / self.positives.height if self.positives.height else float("nan")
            )
        return self._eligible_click_coverage

    def _read_recommendations(self, recommendations):
        # Read only the three standard candidate columns.
        if isinstance(recommendations, (str, Path)):
            lazy = pl.scan_parquet(recommendations)
            available_columns = lazy.collect_schema().names()
        elif isinstance(recommendations, pl.DataFrame):
            lazy = None
            available_columns = recommendations.columns
        else:
            raise TypeError(
                "recommendations must be a Polars DataFrame, str, or Path"
            )

        missing = set(REQUIRED_COLUMNS) - set(available_columns)
        if missing:
            raise ValueError(
                "recommendations are missing required columns: "
                + ", ".join(sorted(missing))
            )

        if lazy is not None:
            return lazy.select(REQUIRED_COLUMNS).collect()
        return recommendations.select(REQUIRED_COLUMNS)

    def _prepare(self, recommendations):
        frame = self._read_recommendations(recommendations)

        if frame.height == 0:
            raise ValueError("recommendations are empty")
        if frame["impression_uid"].null_count() or frame["news_code"].null_count():
            raise ValueError("recommendations contain null keys")
        if frame["cutoff"].null_count():
            raise ValueError("recommendations contain null cutoffs")
        if (frame["cutoff"] < 1).any():
            raise ValueError("cutoff values must be positive")

        # Every impression must have the same contiguous cutoff depth C.
        max_cutoff = int(frame["cutoff"].max())
        shape = frame.group_by("impression_uid").agg(
            pl.len().alias("n_candidates"),
            pl.col("cutoff").min().alias("min_cutoff"),
            pl.col("cutoff").max().alias("max_cutoff"),
            pl.col("cutoff").n_unique().alias("n_unique_cutoffs"),
            pl.col("news_code").n_unique().alias("n_unique_news"),
        )
        invalid = shape.filter(
            (pl.col("n_candidates") != max_cutoff)
            | (pl.col("min_cutoff") != 1)
            | (pl.col("max_cutoff") != max_cutoff)
            | (pl.col("n_unique_cutoffs") != max_cutoff)
            | (pl.col("n_unique_news") != max_cutoff)
        )
        if invalid.height:
            raise ValueError(
                f"invalid candidate cutoffs for {invalid.height} impressions"
            )

        # Candidates must cover exactly the split's impression set.
        known_impressions = self.impressions.select("impression_uid")
        unknown_impressions = frame.join(
            known_impressions, on="impression_uid", how="anti"
        ).height
        if unknown_impressions:
            raise ValueError(
                f"{unknown_impressions} candidates use unknown impression_uid values"
            )
        missing_impressions = known_impressions.join(
            frame.select("impression_uid").unique(),
            on="impression_uid",
            how="anti",
        ).height
        if missing_impressions:
            raise ValueError(
                f"missing recommendations for {missing_impressions} impressions"
            )

        # Reject article codes outside the catalogue.
        unknown_news = frame.join(
            self.availability.select("news_code"), on="news_code", how="anti"
        ).height
        if unknown_news:
            raise ValueError(f"{unknown_news} candidates use unknown news_code values")

        # Guard against articles first observed after the request.
        with_availability = frame.join(
            self.impressions.select("impression_uid", "timestamp"),
            on="impression_uid",
            how="left",
        ).join(self.availability, on="news_code", how="left")
        ineligible = with_availability.filter(
            pl.col("available_from") > pl.col("timestamp")
        ).height
        if ineligible:
            raise ValueError(f"{ineligible} ineligible candidates are not yet available")

        return _PreparedRecommendations(frame=frame, max_cutoff=max_cutoff)

    def evaluate(
        self,
        recommendations,
        *,
        model_name,
        m_values=DEFAULT_M_VALUES,
    ):
        # All M values slice the same validated Top-C artifact.
        m_values = _normalise_m_values(m_values)
        requested_max = max(m_values)
        prepared = self._prepare(recommendations)
        if prepared.max_cutoff < requested_max:
            raise ValueError(
                f"recommendations only provide cutoff {prepared.max_cutoff}; "
                f"cannot evaluate M={requested_max}"
            )

        rows = [self._metrics_at(prepared, m, model_name) for m in m_values]
        return pl.DataFrame(rows)

    def _metrics_at(self, prepared, m, model_name):
        selected = prepared.frame.filter(pl.col("cutoff") <= m)

        # Hits are observed clicks found in the selected Top-M set.
        hits = (
            self.positives.select("impression_uid", "news_code")
            .join(
                selected.select("impression_uid", "news_code"),
                on=["impression_uid", "news_code"],
                how="inner",
            )
            .group_by("impression_uid")
            .agg(pl.len().alias("hits"))
        )
        candidate_counts = (
            self.impressions.select("impression_uid")
            .join(
                selected.group_by("impression_uid").agg(
                    pl.len().alias("n_candidates")
                ),
                on="impression_uid",
                how="left",
            )
            .select(
                "impression_uid",
                pl.col("n_candidates").fill_null(0).cast(pl.Int64),
            )
        )

        # Anchor on all impressions; a missing hit becomes zero recall.
        per_impression = (
            self.impressions.select("impression_uid")
            .join(self.relevant_totals, on="impression_uid", how="left")
            .with_columns(pl.col("relevant").fill_null(0).cast(pl.Int64))
            .join(hits, on="impression_uid", how="left")
            .with_columns(pl.col("hits").fill_null(0).cast(pl.Int64))
            .join(candidate_counts, on="impression_uid", how="left")
            .with_columns(pl.col("n_candidates").fill_null(0).cast(pl.Int64))
        )

        impressions = per_impression.height
        relevant_clicks = int(per_impression["relevant"].sum())
        retrieved_relevant = int(per_impression["hits"].sum())
        # Equal weight per impression; the pooled value is reported separately.
        with_relevant = per_impression.filter(pl.col("relevant") > 0)
        recall_impression_mean = (
            float((with_relevant["hits"] / with_relevant["relevant"]).mean())
            if with_relevant.height
            else float("nan")
        )
        hit_rate = (
            per_impression.select((pl.col("hits") > 0).sum()).item() / impressions
            if impressions
            else float("nan")
        )

        return {
            "model": model_name,
            "split": self.split,
            "m": int(m),
            # Click-weighted recall across the whole split.
            "recall_pooled": (
                retrieved_relevant / relevant_clicks
                if relevant_clicks
                else float("nan")
            ),
            "recall_impression_mean": recall_impression_mean,
            "hit_rate": hit_rate,
            "impressions": impressions,
            "relevant_clicks": relevant_clicks,
            "retrieved_relevant": retrieved_relevant,
            "mean_candidates": float(per_impression["n_candidates"].mean()),
            "mean_eligible_pool": self.mean_eligible_pool,
            "eligible_click_coverage": self.eligible_click_coverage,
        }


def evaluate_candidate_lists(
    recommendations,
    *,
    model_name,
    split="validation",
    m_values=DEFAULT_M_VALUES,
    root=None,
):
    evaluator = CandidateEvaluator(root=root, split=split)
    return evaluator.evaluate(
        recommendations,
        model_name=model_name,
        m_values=m_values,
    )


__all__ = [
    "CandidateEvaluator",
    "DEFAULT_M_VALUES",
    "REQUIRED_COLUMNS",
    "evaluate_candidate_lists",
]
