# =============================================================================
# cg_evaluation.py
# Shared evaluation for candidate-generation models (popular and niche channels)
# =============================================================================
#
# INPUT: one row per recommended article
#     impression_uid  Int64
#     news_code       Int32
#     cutoff          Int16   rank 1..C (every impression has exactly C rows)
#
# OUTPUT: one long table
#     model, split, m, metric, value, numerator, denominator
#
# USAGE:
#     from cg_evaluation import CandidateEvaluator, wide_table
#     ev = CandidateEvaluator(split="validation")
#     res = ev.evaluate_to_csv("validation_top500.parquet", model_name="niche")
#     wide_table(res)
#
# -----------------------------------------------------------------------------
# DEFINITIONS (all metrics use the Top-M list, i.e. cutoff <= M)
# -----------------------------------------------------------------------------
# Hit       : a clicked article that appears in the user's Top-M list.
#             Only real clicks count; articles never shown are unknown.
#
# Popular   : for a request in hour H, the Top-N articles (default 200) by
#             clicks in the previous completed hour H-1. Ties broken by shows / exposure,
#             then news_code.
# Long tail : every article that is not popular for that hour (niche).
# Cold      : an article never shown in any hour before the click's hour.
#
# recall_pooled           total hits / total clicks
# recall_impression_mean  average of (hits / clicks) per impression with a click
# hit_rate                impressions with >= 1 hit / all impressions
# long_tail_recall        hits on long-tail clicks / all long-tail clicks
# cold_article_recall     hits on cold clicks / all cold clicks
# novelty_bits            average of -log2((clicks_H-1 + 1) / (total_H-1 + K))
#                         over recommended articles, K = catalogue size
#                         (higher = less mainstream)
# catalogue_coverage      unique recommended articles / available articles
# personalisation         average Jaccard distance between two users' lists
#                         (higher = lists differ more between users)
# =============================================================================

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_M_VALUES = (50, 100, 200, 500)
POPULAR_TOP_N = 200
MAX_USER_PAIRS = 10_000
SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = ("impression_uid", "news_code", "cutoff")
ONE_HOUR = timedelta(hours=1)

OUTPUT_SCHEMA = {
    "model": pl.String,
    "split": pl.String,
    "m": pl.Int64,
    "metric": pl.String,
    "value": pl.Float64,
    "numerator": pl.Float64,
    "denominator": pl.Float64,
}


# Small helper functions


def _ratio(numerator, denominator):
    # safe division
    return float(numerator) / float(denominator) if denominator else float("nan")


def _to_hour(column="timestamp"):
    # round a timestamp down to the start of its hour
    return pl.col(column).dt.truncate("1h").cast(pl.Datetime("ns")).alias("hour")


def _clean_m_values(m_values):
    # Accept 100 or (50, 100, 200); remove duplicates; must be positive.
    values = (m_values,) if isinstance(m_values, int) else m_values
    values = tuple(dict.fromkeys(int(v) for v in values))
    if not values or min(values) < 1:
        raise ValueError("m_values must contain positive integers")
    return values


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------


class CandidateEvaluator:
    # Loads one split once, then evaluates any number of models.

    def __init__(self, root=None, split="validation", popular_top_n=POPULAR_TOP_N):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")

        # Default root = two folders above this file (the project root).
        self.root = (
            Path(root).resolve() if root else Path(__file__).resolve().parents[2]
        )
        self.split = split
        self.popular_top_n = int(popular_top_n)
        self.processed = self.root / "1_Data_Preparation" / "processed"
        self.artifacts = self.root / "3_Representations" / "artifacts"

        # Loaded lazily the first time they are needed.
        self._impressions = None
        self._availability = None
        self._hourly = None
        self._clicks = None

    # data related

    @property
    def impressions(self):
        # Loads the list of requests (impressions) for the configured split.
        if self._impressions is None:
            self._impressions = pl.read_parquet(
                self.processed / f"impressions_{self.split}.parquet",
                columns=["impression_uid", "user_code", "timestamp"],
            )
        return self._impressions

    @property
    def availability(self):
        # when each article became available (used for coverage and K).
        if self._availability is None:
            self._availability = pl.read_parquet(
                self.artifacts / "article_availability.parquet",
                columns=["news_code", "available_from"],
            )
        return self._availability

    @property
    def hourly(self):
        # Shows and clicks per (hour, article).
        # Earlier splits are included so the first validation/test hour can
        # look back at the last hour of the previous split.
        if self._hourly is None:
            splits_so_far = SPLITS[: SPLITS.index(self.split) + 1]
            logs = [
                pl.scan_parquet(self.processed / f"candidates_{s}.parquet").select(
                    "timestamp", "news_code", "clicked"
                )
                for s in splits_so_far
            ]
            self._hourly = (
                pl.concat(logs)
                .with_columns(_to_hour())
                .group_by("hour", "news_code")
                .agg(
                    pl.len().alias("shows"),
                    pl.col("clicked").cast(pl.Int64).sum().alias("clicks"),
                )
                .collect()
            )
        return self._hourly

    @property
    def clicks(self):
        # collect all clicks and note their hours
        if self._clicks is None:
            clicks = (
                pl.scan_parquet(self.processed / f"candidates_{self.split}.parquet")
                .filter(pl.col("clicked") == 1)
                .select("impression_uid", "news_code")
                .collect()
                .join(
                    self.impressions.select("impression_uid", "timestamp"),
                    on="impression_uid",
                )
                .with_columns(_to_hour())
            )

            # Popular in hour H = the popular channel's top-N list for hour H:
            # every article available by the end of hour H, ranked by clicks in H-1, then shows in H-1, then news_code.
            request_hours = clicks.select("hour").unique()
            prev_counts = self.hourly.select(
                (pl.col("hour") + ONE_HOUR).cast(pl.Datetime("ns")).alias("hour"),
                "news_code",
                "clicks",
                "shows",
            )
            popular = (
                request_hours.join(self.availability, how="cross")
                .filter(pl.col("available_from") <= pl.col("hour") + ONE_HOUR)
                .join(prev_counts, on=["hour", "news_code"], how="left")
                .with_columns(pl.col("clicks", "shows").fill_null(0))
                # sort by no. of clicks, then exposure / shows (tie-breaker), then news_code.
                .sort(
                    ["hour", "clicks", "shows", "news_code"],
                    descending=[False, True, True, False],
                )
                .group_by("hour", maintain_order=True)
                .head(self.popular_top_n)
                .select("hour", "news_code", pl.lit(True).alias("is_popular"))
            )

            # First hour each article was ever shown.
            first_shown = self.hourly.group_by("news_code").agg(
                pl.col("hour").min().alias("first_shown")
            )

            self._clicks = (
                clicks.join(popular, on=["hour", "news_code"], how="left")
                .join(first_shown, on="news_code", how="left")
                .select(
                    "impression_uid",
                    "news_code",
                    # Long tail = not in that hour's popular set.
                    pl.col("is_popular").is_null().alias("is_long_tail"),
                    # Cold = not shown in any earlier hour.
                    (
                        pl.col("first_shown").is_null()
                        | (pl.col("first_shown") >= pl.col("hour"))
                    ).alias("is_cold"),
                )
            )
        return self._clicks

    # input checks

    def _load_recommendations(self, recommendations):
        # Accept a parquet path or a Polars DataFrame.
        if isinstance(recommendations, (str, Path)):
            frame = pl.read_parquet(recommendations)
        elif isinstance(recommendations, pl.DataFrame):
            frame = recommendations
        else:
            raise TypeError("recommendations must be a Polars DataFrame, str, or Path")

        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"recommendations are missing columns: {sorted(missing)}")
        if frame.height == 0:
            raise ValueError("recommendations are empty")
        return frame.select(REQUIRED_COLUMNS)

    def _select_impressions(self, impression_uids):
        # All impressions in the split, or only the requested subset.
        if impression_uids is None:
            return self.impressions

        if isinstance(impression_uids, pl.DataFrame):
            requested = impression_uids.select("impression_uid")
        else:
            requested = pl.DataFrame(
                {"impression_uid": np.asarray(impression_uids, dtype=np.int64)}
            )

        if requested.height == 0:
            raise ValueError("impression_uids is empty")
        if requested["impression_uid"].n_unique() != requested.height:
            raise ValueError("impression_uids contains duplicates")

        subset = self.impressions.join(requested, on="impression_uid")
        if subset.height != requested.height:
            raise ValueError("some impression_uids are not in the split")
        return subset

    def _check_shape(self, frame, impressions):
        # Every impression needs exactly C rows: cutoffs 1..C, no duplicate articles.
        depth = int(frame["cutoff"].max())
        bad = (
            frame.group_by("impression_uid")
            .agg(
                pl.len().alias("rows"),
                pl.col("cutoff").min().alias("first"),
                pl.col("cutoff").n_unique().alias("ranks"),
                pl.col("news_code").n_unique().alias("articles"),
            )
            .filter(
                (pl.col("rows") != depth)
                | (pl.col("first") != 1)
                | (pl.col("ranks") != depth)
                | (pl.col("articles") != depth)
            )
        )
        if bad.height:
            raise ValueError(f"invalid candidate cutoffs for {bad.height} impressions")

        # Recommendations must cover exactly the evaluated impressions.
        known = impressions.select("impression_uid")
        given = frame.select("impression_uid").unique()
        unknown = given.join(known, on="impression_uid", how="anti").height
        missing = known.join(given, on="impression_uid", how="anti").height
        if unknown:
            raise ValueError(f"{unknown} impression_uids are not in the split")
        if missing:
            raise ValueError(f"missing recommendations for {missing} impressions")
        return depth

    # evaluation metrics

    # recall-related metrics
    def _recall_metrics(self, clicks, impressions, m):
        # clicks has a 'cutoff' column: the click's rank in the list (null = absent).
        clicks = clicks.with_columns(
            (pl.col("cutoff").fill_null(m + 1) <= m).alias("hit")
        )

        per_impression = (
            impressions.select("impression_uid")
            .join(
                clicks.group_by("impression_uid").agg(
                    pl.len().alias("clicks"), pl.col("hit").sum().alias("hits")
                ),
                on="impression_uid",
                how="left",
            )
            .fill_null(0)
        )

        # recall_pooled: total hits / total clicks
        total_clicks = per_impression["clicks"].sum()
        total_hits = per_impression["hits"].sum()

        # recall_impression_mean: average of hits / clicks per impression with a click
        clicked = per_impression.filter(pl.col("clicks") > 0)
        impression_mean = (clicked["hits"] / clicked["clicks"]).mean()

        # hit_rate: impressions with at least 1 hit / all impressions
        hit_impressions = (per_impression["hits"] > 0).sum()

        # long_tail_recall: recall on clicks for non-popular articles only
        tail = clicks.filter("is_long_tail")

        # cold_article_recall: recall on clicks for never-shown-before articles only
        cold = clicks.filter("is_cold")

        # (metric, value, numerator, denominator)
        return [
            (
                "recall_pooled",
                _ratio(total_hits, total_clicks),
                total_hits,
                total_clicks,
            ),
            ("recall_impression_mean", impression_mean, None, clicked.height),
            (
                "hit_rate",
                _ratio(hit_impressions, per_impression.height),
                hit_impressions,
                per_impression.height,
            ),
            (
                "long_tail_recall",
                _ratio(tail["hit"].sum(), tail.height),
                tail["hit"].sum(),
                tail.height,
            ),
            (
                "cold_article_recall",
                _ratio(cold["hit"].sum(), cold.height),
                cold["hit"].sum(),
                cold.height,
            ),
        ]

    # novelty_bits: how unpopular the recommended articles were last hour (a score for how rare an article is, based on its share of last hour’s clicks:)
    def _novelty_per_row(self, frame, impressions):
        # Novelty (bits) of every recommended row, using clicks in hour H-1.
        # +1 / +K smoothing so articles with 0 clicks still get a finite value.
        catalogue_size = self.availability.height
        prev_hour = self.hourly.select(
            (pl.col("hour") + ONE_HOUR).cast(pl.Datetime("ns")).alias("hour"),
            "news_code",
            "clicks",
        )
        hour_totals = prev_hour.group_by("hour").agg(
            pl.col("clicks").sum().alias("total")
        )

        return (
            frame.join(
                impressions.select("impression_uid", _to_hour()), on="impression_uid"
            )
            .join(prev_hour, on=["hour", "news_code"], how="left")
            .join(hour_totals, on="hour", how="left")
            .select(
                "cutoff",
                -(
                    (pl.col("clicks").fill_null(0) + 1)
                    / (pl.col("total").fill_null(0) + catalogue_size)
                )
                .log(2)
                .alias("bits"),
            )
        )

    # coverage = different articles recommended / articles available (how much of the catalogue the model actually uses)
    def _coverage(self, frame, impressions, m):
        # Unique recommended articles / articles available by the split's end.
        recommended = frame.filter(pl.col("cutoff") <= m)["news_code"].n_unique()
        available = self.availability.filter(
            pl.col("available_from") <= impressions["timestamp"].max()
        ).height
        return _ratio(recommended, available), recommended, available

    # personalisation: how different the Top-M lists are between users
    def _personalisation(self, frame, impressions, m, seed):
        # Average Jaccard distance between Top-M lists of random user pairs.
        # Each user is represented by their first impression in the split.
        first = (
            impressions.sort("user_code", "timestamp", "impression_uid")
            .group_by("user_code", maintain_order=True)
            .first()
            .select("user_code", "impression_uid")
        )
        users = np.sort(first["user_code"].to_numpy())
        n = users.size
        if n < 2:
            return float("nan"), 0.0, 0

        # Sample pairs; offsetting by 1..n-1 means a user is never paired with themselves.
        rng = np.random.default_rng(seed)
        n_pairs = min(MAX_USER_PAIRS, n * (n - 1) // 2)
        left = rng.integers(0, n, size=n_pairs)
        right = (left + rng.integers(1, n, size=n_pairs)) % n

        # Fetch the Top-M list of every sampled user.
        sampled = np.unique(np.concatenate([users[left], users[right]]))
        lists = (
            frame.filter(pl.col("cutoff") <= m)
            .join(
                first.filter(pl.col("user_code").is_in(sampled.tolist())),
                on="impression_uid",
            )
            .group_by("user_code")
            .agg(pl.col("news_code").unique().sort())
        )
        user_list = {u: np.asarray(codes) for u, codes in lists.iter_rows()}

        # Jaccard distance = 1 - shared / union.
        total = 0.0
        for a_user, b_user in zip(users[left], users[right]):
            a, b = user_list[a_user], user_list[b_user]
            shared = np.intersect1d(a, b, assume_unique=True).size
            total += 1.0 - shared / (a.size + b.size - shared)
        return total / n_pairs, total, n_pairs

    def evaluate(
        self,
        recommendations,
        *,
        model_name,
        m_values=DEFAULT_M_VALUES,
        impression_uids=None,
        seed=0,
    ):
        m_values = _clean_m_values(m_values)
        impressions = self._select_impressions(impression_uids)
        frame = self._load_recommendations(recommendations)

        depth = self._check_shape(frame, impressions)
        if depth < max(m_values):
            raise ValueError(
                f"lists only go to rank {depth}; cannot evaluate M={max(m_values)}"
            )

        # Computed once, then sliced for each M.
        ranked_clicks = self.clicks.join(
            impressions.select("impression_uid"), on="impression_uid"
        ).join(frame, on=["impression_uid", "news_code"], how="left")
        novelty = self._novelty_per_row(frame, impressions)

        rows = []
        for m in m_values:
            top_novelty = novelty.filter(pl.col("cutoff") <= m)["bits"]
            metrics = self._recall_metrics(ranked_clicks, impressions, m)
            metrics.append(
                (
                    "novelty_bits",
                    top_novelty.mean(),
                    top_novelty.sum(),
                    top_novelty.len(),
                )
            )
            metrics.append(
                ("catalogue_coverage", *self._coverage(frame, impressions, m))
            )
            metrics.append(
                ("personalisation", *self._personalisation(frame, impressions, m, seed))
            )

            for name, value, numerator, denominator in metrics:
                rows.append(
                    {
                        "model": model_name,
                        "split": self.split,
                        "m": m,
                        "metric": name,
                        "value": None if value is None else float(value),
                        "numerator": None if numerator is None else float(numerator),
                        "denominator": None
                        if denominator is None
                        else float(denominator),
                    }
                )

        return pl.DataFrame(rows, schema=OUTPUT_SCHEMA).sort("m", "metric")

    def evaluate_to_csv(
        self,
        recommendations,
        *,
        model_name,
        m_values=DEFAULT_M_VALUES,
        output_dir=None,
        write_comparison=True,
        impression_uids=None,
        suffix="",
        seed=0,
    ):
        # Evaluate, save this model's CSV, and update the shared comparison CSV.
        table = self.evaluate(
            recommendations,
            model_name=model_name,
            m_values=m_values,
            impression_uids=impression_uids,
            seed=seed,
        )

        output_dir = Path(output_dir) if output_dir else self.root / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        table.write_csv(
            output_dir / f"candidate_generation_{model_name}_{self.split}{suffix}.csv"
        )

        if write_comparison:
            update_comparison(
                table,
                output_dir
                / f"candidate_generation_comparison_{self.split}{suffix}.csv",
            )
        return table


# Output helpers


def update_comparison(evaluation, comparison_path):
    # Replace this model's rows in the shared comparison table (others are kept).
    comparison_path = Path(comparison_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    if comparison_path.exists():
        existing = pl.read_csv(comparison_path)
        if set(OUTPUT_SCHEMA) <= set(existing.columns):
            models = evaluation["model"].unique().to_list()
            existing = existing.filter(~pl.col("model").is_in(models))
            evaluation = pl.concat(
                [existing.select(list(OUTPUT_SCHEMA)), evaluation],
                how="vertical_relaxed",
            )

    evaluation.sort("model", "m", "metric").write_csv(comparison_path)
    return comparison_path


def wide_table(evaluation):
    # One row per (model, m), one column per metric. Easier to read.
    return evaluation.pivot(
        on="metric", index=["model", "split", "m"], values="value"
    ).sort("model", "m")


__all__ = [
    "DEFAULT_M_VALUES",
    "OUTPUT_SCHEMA",
    "POPULAR_TOP_N",
    "CandidateEvaluator",
    "update_comparison",
    "wide_table",
]
