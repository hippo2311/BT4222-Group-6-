"""Shared evaluation for every reranking notebook.

The module has four parts:

1. ``RerankingEvaluator`` scores complete, labelled logged slates.  MIND only has
   labels for articles shown in a historical impression, so a model must score
   every row in ``labels_<split>`` and is ranked within each logged impression.
2. ``score_and_evaluate`` streams a split's feature partitions through a notebook's
   ``predict`` function, saves the score file and evaluates it.
3. ``compare_results`` and ``rank_logged_slate`` build comparison tables and show
   one user's ranked slate.
4. ``evaluate_retrieved_candidates`` reports known-click metrics for generated
   (retrieved) candidate lists, whose unshown articles have no labels.

Model, data-preparation and training code stays inside the notebooks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED_COLUMNS = ("impression_uid", "news_code", "score")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class MetricContribution:
    """One metric's contribution from a single impression.

    A numeric denominator produces a pooled/mean metric by accumulating both
    values across impressions.  ``None`` denotes a count, which is summed.
    """

    numerator: float
    denominator: float | None = 1.0


def mean(value: float) -> MetricContribution:
    """Contribute one equally weighted per-impression value."""
    return MetricContribution(float(value), 1.0)


def ratio(numerator: float, denominator: float) -> MetricContribution:
    """Contribute a pooled numerator/denominator metric."""
    return MetricContribution(float(numerator), float(denominator))


def total(value: float) -> MetricContribution:
    """Contribute a count or other value that should be summed."""
    return MetricContribution(float(value), None)


@dataclass(frozen=True)
class SlateContext:
    """Validated data for one complete logged impression slate."""

    labels: np.ndarray
    scores: np.ndarray
    news_codes: np.ndarray
    ranked_labels: np.ndarray
    k_values: tuple[int, ...]

    @property
    def relevant(self) -> float:
        return float(self.labels.sum())


MetricFunction = Callable[[SlateContext], Mapping[str, MetricContribution]]
METRIC_REGISTRY: dict[str, MetricFunction] = {}


def register_metric(name: str):
    """Register a per-slate metric function with a small decorator.

    A function receives :class:`SlateContext` and returns a mapping from output
    column name to ``mean(...)``, ``ratio(...)`` or ``total(...)`` contribution.
    """

    def decorator(function: MetricFunction) -> MetricFunction:
        if name in METRIC_REGISTRY:
            raise ValueError(f"metric already registered: {name}")
        METRIC_REGISTRY[name] = function
        return function

    return decorator


def _dcg(labels: np.ndarray, k: int) -> float:
    if not len(labels):
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, min(k, len(labels)) + 2))
    return float(labels[:k] @ discounts)


def _ranked_labels(labels: np.ndarray, scores: np.ndarray, news_codes: np.ndarray):
    """Rank labels using the documented deterministic top-k tie rule."""
    order = np.lexsort((news_codes, -scores))
    return labels[order].astype(np.float64, copy=False)


@register_metric("mrr")
def mrr_metric(context: SlateContext) -> Mapping[str, MetricContribution]:
    """Mean reciprocal rank of the first known clicked article."""
    first_positive = np.flatnonzero(context.ranked_labels > 0)
    value = float(1.0 / (first_positive[0] + 1)) if len(first_positive) else 0.0
    return {"mrr": mean(value)}


@register_metric("ndcg")
def ndcg_metric(context: SlateContext) -> Mapping[str, MetricContribution]:
    """Mean nDCG at every requested K."""
    ideal = np.sort(context.labels)[::-1]
    return {
        f"ndcg_at_{k}": mean(
            _dcg(context.ranked_labels, k) / _dcg(ideal, k) if context.relevant else 0.0
        )
        for k in context.k_values
    }


@register_metric("recall")
def recall_metric(context: SlateContext) -> Mapping[str, MetricContribution]:
    """Impression-mean Recall at every requested K."""
    return {
        f"recall_at_{k}": mean(
            float(context.ranked_labels[:k].sum()) / context.relevant if context.relevant else 0.0
        )
        for k in context.k_values
    }


@register_metric("click_retention")
def click_retention_metric(context: SlateContext) -> Mapping[str, MetricContribution]:
    """Pooled fraction of known clicks retained at every requested K."""
    if not context.relevant:
        return {}
    return {
        f"click_retention_at_{k}": ratio(
            float(context.ranked_labels[:k].sum()), context.relevant
        )
        for k in context.k_values
    }


@register_metric("engaged_impression_rate")
def engaged_impression_rate_metric(context: SlateContext) -> Mapping[str, MetricContribution]:
    """Share of impressions with at least one known Top-K click."""
    return {
        f"engaged_impression_rate_at_{k}": mean(
            float(context.ranked_labels[:k].sum() > 0)
        )
        for k in context.k_values
    }


@register_metric("observed_ctr")
def observed_ctr_metric(context: SlateContext) -> Mapping[str, MetricContribution]:
    """Pooled observed clicked-article rate in Top-K-or-shorter slates."""
    contributions: dict[str, MetricContribution] = {}
    for k in context.k_values:
        slots = min(k, len(context.ranked_labels))
        if slots:
            contributions[f"observed_ctr_at_{k}"] = ratio(
                float(context.ranked_labels[:k].sum()), slots
            )
    return contributions


class RerankingEvaluator:
    """Validate complete scored logged slates and calculate impression metrics."""

    def __init__(self, root=None, split="validation", feature_run=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
        self.split = split
        self.feature_run = (
            Path(feature_run).resolve()
            if feature_run is not None
            else self.root / "4_Feature_Engineering" / "artifacts" / "prequential-all"
        )

    @property
    def labels_path(self):
        return self.feature_run / f"labels_{self.split}"

    def _read_scores(self, scores):
        if isinstance(scores, pl.DataFrame):
            frame = scores
        elif isinstance(scores, (str, Path)):
            frame = pl.scan_parquet(scores).select(REQUIRED_COLUMNS).collect()
        else:
            raise TypeError("scores must be a Polars DataFrame, str, or Path")
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError("scores are missing required columns: " + ", ".join(sorted(missing)))
        return frame.select(REQUIRED_COLUMNS)

    def _labelled_scores(self, scores):
        scored = self._read_scores(scores)
        if not scored.height:
            raise ValueError("scores are empty")
        if scored.select(pl.any_horizontal(pl.all().is_null()).any()).item():
            raise ValueError("scores contain null keys or scores")
        scored = scored.with_columns(pl.col("score").cast(pl.Float64, strict=True))
        if not scored.select(pl.col("score").is_finite().all()).item():
            raise ValueError("scores must be finite")
        duplicate_scores = scored.group_by(["impression_uid", "news_code"]).len().filter(pl.col("len") > 1)
        if duplicate_scores.height:
            raise ValueError("scores contain duplicate impression/news pairs")

        labels = pl.scan_parquet(self.labels_path).select(
            "impression_uid", "news_code", "clicked"
        ).collect()
        duplicate_labels = labels.group_by(["impression_uid", "news_code"]).len().filter(pl.col("len") > 1)
        if duplicate_labels.height:
            raise ValueError("saved labels contain duplicate impression/news pairs")
        keys = ["impression_uid", "news_code"]
        if scored.join(labels.select(keys), on=keys, how="anti").height:
            raise ValueError("scores include pairs outside the labelled logged slates")
        if labels.join(scored.select(keys), on=keys, how="anti").height:
            raise ValueError("scores do not cover every labelled logged-slate pair")
        return scored.join(labels, on=keys, how="inner")

    def evaluate(
        self,
        scores,
        *,
        model_name,
        impression_cohorts=None,
        k_values=(5, 10),
        metric_names=None,
    ):
        """Evaluate scores and return one row for overall plus optional cohorts.

        ``impression_cohorts`` may contain ``impression_uid`` and ``cohort``.  It
        must assign every evaluated impression exactly one cohort.

        ``metric_names`` selects decorated functions in ``METRIC_REGISTRY``.  By
        default every registered metric runs, so new metrics become available
        without changing this evaluator's aggregation loop.
        """
        k_values = tuple(dict.fromkeys(int(k) for k in k_values))
        if not k_values or any(k < 1 for k in k_values):
            raise ValueError("k_values must contain positive integers")
        metric_names = tuple(METRIC_REGISTRY) if metric_names is None else tuple(metric_names)
        unknown_metrics = set(metric_names) - set(METRIC_REGISTRY)
        if unknown_metrics:
            raise ValueError("unknown metrics: " + ", ".join(sorted(unknown_metrics)))
        frame = self._labelled_scores(scores)
        if impression_cohorts is None:
            frame = frame.with_columns(pl.lit("overall").alias("cohort"))
        else:
            required = {"impression_uid", "cohort"}
            missing = required - set(impression_cohorts.columns)
            if missing:
                raise ValueError("impression_cohorts missing: " + ", ".join(sorted(missing)))
            cohorts = impression_cohorts.select("impression_uid", "cohort")
            if cohorts["impression_uid"].n_unique() != cohorts.height or cohorts["cohort"].null_count():
                raise ValueError("impression_cohorts must have one non-null cohort per impression")
            frame = frame.join(cohorts, on="impression_uid", how="left")
            if frame["cohort"].null_count():
                raise ValueError("impression_cohorts do not cover every scored impression")

        rows = []
        for cohort, group in frame.partition_by("cohort", as_dict=True, maintain_order=True).items():
            metric_totals: dict[str, list[float | None]] = {}
            impressions = 0
            for slate in group.partition_by("impression_uid", maintain_order=True):
                labels = slate["clicked"].to_numpy()
                scores_array = slate["score"].to_numpy()
                news_codes = slate["news_code"].to_numpy()
                context = SlateContext(
                    labels=labels,
                    scores=scores_array,
                    news_codes=news_codes,
                    ranked_labels=_ranked_labels(labels, scores_array, news_codes),
                    k_values=k_values,
                )
                for metric_name in metric_names:
                    for column, contribution in METRIC_REGISTRY[metric_name](context).items():
                        values = metric_totals.setdefault(
                            column,
                            [0.0, None if contribution.denominator is None else 0.0],
                        )
                        if (values[1] is None) != (contribution.denominator is None):
                            raise ValueError(
                                f"metric column {column!r} mixes count and ratio contributions"
                            )
                        values[0] += contribution.numerator
                        if contribution.denominator is not None:
                            values[1] = float(values[1]) + contribution.denominator
                impressions += 1
            row = {
                "model": model_name,
                "split": self.split,
                "cohort": cohort if isinstance(cohort, str) else cohort[0],
                "impressions": impressions,
                "tie_break": "news_code_ascending_for_top_k",
            }
            for column, (numerator, denominator) in metric_totals.items():
                row[column] = numerator if denominator is None else numerator / denominator if denominator else None
            rows.append(row)
        return pl.DataFrame(rows)


def add_relative_lift(results, *, baseline_model):
    """Add percentage observed-CTR lift relative to one model in a result table.

    The lift is a logged-slate, observed-label proxy only. It is not an estimate
    of online CTR lift and must not be applied to global retrieved candidates.
    """
    required = {"model", "split", "cohort"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError("results missing: " + ", ".join(sorted(missing)))
    baseline = results.filter(pl.col("model") == baseline_model)
    if not baseline.height:
        raise ValueError(f"baseline model not found: {baseline_model}")
    keys = ["split", "cohort"]
    if baseline.group_by(keys).len().filter(pl.col("len") > 1).height:
        raise ValueError("baseline must have one row per split/cohort")
    metric_columns = [name for name in results.columns if name.startswith("observed_ctr_at_")]
    renamed = baseline.select(
        *keys, *[pl.col(name).alias(f"_baseline_{name}") for name in metric_columns]
    )
    output = results.join(renamed, on=keys, how="left")
    expressions = []
    for name in metric_columns:
        baseline_name = f"_baseline_{name}"
        expressions.append(
            pl.when(pl.col(baseline_name).is_not_null() & (pl.col(baseline_name) != 0))
            .then(100 * (pl.col(name) - pl.col(baseline_name)) / pl.col(baseline_name))
            .otherwise(None)
            .alias(f"{name}_lift_pct_vs_baseline")
        )
    return output.with_columns(expressions).drop([f"_baseline_{name}" for name in metric_columns])


def evaluate_scored_slates(scores, *, model_name, split="validation", root=None, feature_run=None):
    """Convenience wrapper for a one-off complete logged-slate evaluation."""
    return RerankingEvaluator(root=root, split=split, feature_run=feature_run).evaluate(
        scores, model_name=model_name
    )


# ---------------------------------------------------------------------------
# Scoring complete logged slates
# ---------------------------------------------------------------------------

def write_scores(predict, *, split, feature_run, output_path, columns=()):
    """Stream every logged candidate in ``split`` through ``predict`` into one score file.

    ``predict`` receives one pandas DataFrame per feature partition, holding
    ``impression_uid``, ``news_code`` and the requested ``columns``, and must return
    one finite score per row.  The partition path is available as
    ``frame.attrs["part"]`` for notebooks that join a per-partition sidecar.
    """
    feature_run, output_path = Path(feature_run), Path(output_path)
    parts = sorted((feature_run / f"ranking_features_{split}").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No {split} feature partitions in {feature_run}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    read_columns = list(dict.fromkeys(["impression_uid", "news_code", *columns]))

    writer, rows_written = None, 0
    try:
        for part in parts:
            frame = pq.read_table(part, columns=read_columns).to_pandas()
            frame.attrs["part"] = part
            scores = np.asarray(predict(frame), dtype=np.float64)
            if scores.shape != (len(frame),) or not np.isfinite(scores).all():
                raise ValueError(f"predict must return one finite score per row ({part.name})")
            table = pa.table({
                "impression_uid": frame["impression_uid"],
                "news_code": frame["news_code"],
                "score": scores,
            })
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows_written += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if not rows_written:
        raise ValueError(f"No {split} candidates were scored")
    return output_path


def score_and_evaluate(predict, *, split, feature_run, output_dir, model_name,
                       columns=(), results_dir=None):
    """Score a complete split, evaluate it, and save the scores and metrics.

    Scores are written to ``output_dir/<split>_scores.parquet`` only after the
    evaluator has confirmed they cover every labelled logged-slate pair.  When
    ``results_dir`` is given, metrics are also saved as
    ``reranking_<model_name>_<split>.csv``.
    """
    output_dir = Path(output_dir)
    score_path = output_dir / f"{split}_scores.parquet"
    partial_path = output_dir / f"{split}_scores.partial.parquet"
    write_scores(predict, split=split, feature_run=feature_run,
                 output_path=partial_path, columns=columns)

    evaluator = RerankingEvaluator(split=split, feature_run=feature_run)
    result = evaluator.evaluate(partial_path, model_name=model_name)
    partial_path.replace(score_path)
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        result.write_csv(results_dir / f"reranking_{model_name}_{split}.csv")
    return result, score_path


# ---------------------------------------------------------------------------
# Comparing models and inspecting rankings
# ---------------------------------------------------------------------------

def compare_results(results, *, baseline_model=None, sort_by="ndcg_at_10"):
    """Combine result tables (DataFrames or CSV paths) into one sorted comparison.

    When ``baseline_model`` is present in the table, observed-CTR lift relative to
    it is added with :func:`add_relative_lift`.
    """
    frames = [pl.read_csv(item) if isinstance(item, (str, Path)) else item for item in results]
    if not frames:
        raise ValueError("results must contain at least one table")
    comparison = pl.concat(frames, how="diagonal_relaxed")
    if baseline_model is not None and baseline_model in comparison["model"].to_list():
        comparison = add_relative_lift(comparison, baseline_model=baseline_model)
    return comparison.sort(["cohort", sort_by, "model"], descending=[False, True, False])


def rank_logged_slate(user_code, score_file, *, feature_run, split="validation", top_n=5):
    """Rank one user's latest logged impression with a saved score file.

    This reorders the articles that were actually shown in that impression; it
    does not retrieve new articles from the catalogue and never reads labels.
    Ties use ascending ``news_code``, matching the evaluator.
    """
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    score_file = Path(score_file)
    if not score_file.is_file():
        raise FileNotFoundError(f"Score file not found: {score_file}")

    # Find the user's latest impression (latest timestamp, then largest impression ID).
    rows = (
        pl.scan_parquet(Path(feature_run) / f"ranking_features_{split}" / "*.parquet")
        .filter(pl.col("user_code") == user_code)
        .select("user_code", "impression_uid", "news_code", "timestamp")
        .collect()
    )
    if not rows.height:
        raise ValueError(f"No {split} impressions found for user_code={user_code}")
    latest = rows.sort(["timestamp", "impression_uid"], descending=True)["impression_uid"][0]
    slate = rows.filter(pl.col("impression_uid") == latest)

    scores = pl.scan_parquet(score_file).filter(pl.col("impression_uid") == latest).collect()
    ranked = slate.join(scores, on=["impression_uid", "news_code"], how="left")
    if ranked["score"].null_count():
        raise ValueError("The score file does not cover this complete logged slate")
    return (
        ranked.sort(["score", "news_code"], descending=[True, False])
        .select("user_code", "impression_uid", "news_code", "score")
        .head(top_n)
    )


# ---------------------------------------------------------------------------
# Generated (retrieved) candidate lists
# ---------------------------------------------------------------------------

def evaluate_retrieved_candidates(candidates, *, feature_run, split="validation",
                                  score_column="score", model_name="model", k_values=(5, 10)):
    """Known-click metrics for a generated candidate list ranked by ``score_column``.

    ``candidates`` holds ``impression_uid``, ``news_code`` and the score column.
    Retrieved articles that were never shown have unknown relevance: they count as
    "not a known click", not as confirmed negatives.  Every known click in an
    impression stays in the denominator, so clicks the generator missed are misses.
    ``candidate_recall`` is the pooled share of known clicks that were retrieved,
    an upper bound for the ranking metrics.
    """
    frame = candidates.lazy() if isinstance(candidates, pl.DataFrame) else candidates
    checks = frame.select(
        pl.col(score_column).is_null().any().alias("nulls"),
        pl.col(score_column).is_finite().all().alias("finite"),
    ).collect()
    if checks["nulls"].item() or not checks["finite"].item():
        raise ValueError("scores must be present and finite")
    labels = (
        pl.scan_parquet(Path(feature_run) / f"labels_{split}" / "*.parquet")
        .select("impression_uid", "news_code", "clicked")
    )
    # A caller may score a subset of the split.  Unscored impressions must not
    # enter the denominator: they would look like impressions where every click
    # was missed.
    scored_impressions = frame.select("impression_uid").unique()
    relevant = (
        labels.filter(pl.col("clicked") == 1)
        .join(scored_impressions, on="impression_uid", how="semi")
        .group_by("impression_uid")
        .agg(pl.len().alias("relevant"))
    )
    ranked = (
        frame.select("impression_uid", "news_code", pl.col(score_column).alias("_score"))
        .sort(["impression_uid", "_score", "news_code"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("impression_uid").alias("position"))
        .join(labels, on=["impression_uid", "news_code"], how="left")
        .with_columns(pl.col("clicked").fill_null(0))
    )
    is_click = pl.col("clicked") == 1
    per_impression = ranked.group_by("impression_uid").agg(
        pl.when(is_click).then(1.0 / pl.col("position")).otherwise(0.0).max().alias("mrr"),
        *[
            pl.when(is_click & (pl.col("position") <= k))
            .then(1.0 / (pl.col("position") + 1).log(2)).otherwise(0.0).sum().alias(f"dcg_{k}")
            for k in k_values
        ],
        *[(is_click & (pl.col("position") <= k)).sum().alias(f"hits_{k}") for k in k_values],
        is_click.sum().alias("retrieved_clicks"),
    )
    # Only impressions with at least one known click are evaluated.
    per_impression = (
        relevant.join(per_impression, on="impression_uid", how="left").fill_null(0).collect()
    )

    row = {
        "model": model_name,
        "split": split,
        "impressions": per_impression.height,
        "candidate_recall": per_impression["retrieved_clicks"].sum() / per_impression["relevant"].sum(),
        "mrr": per_impression["mrr"].mean(),
    }
    n_relevant = per_impression["relevant"].to_numpy()
    for k in k_values:
        # Ideal DCG places min(relevant, k) known clicks at the top positions.
        ideal_prefix = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))])
        ideal = ideal_prefix[np.minimum(n_relevant, k)]
        row[f"ndcg_at_{k}"] = float(np.mean(per_impression[f"dcg_{k}"].to_numpy() / ideal))
        row[f"recall_at_{k}"] = float(np.mean(per_impression[f"hits_{k}"].to_numpy() / n_relevant))
    return pl.DataFrame([row])


__all__ = [
    "METRIC_REGISTRY",
    "REQUIRED_COLUMNS",
    "MetricContribution",
    "RerankingEvaluator",
    "SlateContext",
    "add_relative_lift",
    "compare_results",
    "evaluate_retrieved_candidates",
    "evaluate_scored_slates",
    "mean",
    "rank_logged_slate",
    "ratio",
    "register_metric",
    "score_and_evaluate",
    "total",
    "write_scores",
]
