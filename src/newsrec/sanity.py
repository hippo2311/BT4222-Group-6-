"""Bounded training smoke check against popularity on complete validation slates.

Run: PYTHONPATH=src python -m newsrec.sanity
This diagnostic is not a tuned baseline, calibration study or leakage proof.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import sha256, write_json
from .io import open_feature_run


def slate_metrics(labels, scores, k=10):
    """Binary per-slate AUC and expected NDCG under random ordering of score ties."""
    labels = np.asarray(labels, np.float64)
    scores = np.asarray(scores, np.float64)
    order = np.argsort(scores, kind="stable")
    labels, scores = labels[order], scores[order]
    _, start, sizes = np.unique(scores, return_index=True, return_counts=True)
    positives = np.add.reduceat(labels, start)
    negatives = sizes - positives
    p, n = labels.sum(), len(labels) - labels.sum()
    auc = (
        float(
            np.sum(positives * (np.cumsum(negatives) - negatives + 0.5 * negatives))
            / (p * n)
        )
        if p and n
        else None
    )
    expected_relevance = np.repeat((positives / sizes)[::-1], sizes[::-1])
    discounts = 1 / np.log2(np.arange(2, min(k, len(labels)) + 2))
    ideal = discounts[: min(int(p), k)].sum()
    ndcg = float(expected_relevance[:k] @ discounts / ideal) if ideal else 0.0
    return auc, ndcg


def labelled_part(run, split, feature_part, columns):
    features = pq.read_table(
        feature_part, columns=["impression_uid", "news_code", *columns]
    ).to_pandas()
    labels = pq.read_table(run / f"labels_{split}" / feature_part.name).to_pandas()
    result = features.merge(
        labels, on=["impression_uid", "news_code"], validate="one_to_one"
    )
    if len(result) != len(features):
        raise ValueError("Feature/label keys do not match")
    return result


def sample_training_rows(run, columns, limit, seed=4222):
    parts = sorted((run / "ranking_features_train").glob("*.parquet"))
    sizes = [pq.ParquetFile(part).metadata.num_rows for part in parts]
    remaining_population = sum(sizes)
    remaining_take = min(limit, remaining_population)
    rng = np.random.default_rng(seed)
    frames = []
    for part, size in zip(parts, sizes):
        take = int(
            rng.hypergeometric(size, remaining_population - size, remaining_take)
        )
        if take:
            frame = labelled_part(run, "train", part, columns)
            frames.append(frame.iloc[rng.choice(size, take, replace=False)])
        remaining_take -= take
        remaining_population -= size
    return pd.concat(frames, ignore_index=True)


def validation_slates(run, columns, model=None, model_columns=None):
    """Carry an impression across file boundaries so no evaluation slate is cut."""
    pending = None
    last_uid = None
    for part in sorted((run / "ranking_features_validation").glob("*.parquet")):
        frame = labelled_part(run, "validation", part, columns)
        if model is not None:
            frame["_logistic_score"] = model.predict_proba(frame[model_columns])[:, 1]
        if not frame.impression_uid.is_monotonic_increasing:
            # Writer order is timestamp/user grouping, not numeric impression ID.
            # Each request is contiguous; verify that assumption instead.
            groups = frame.impression_uid.ne(frame.impression_uid.shift()).cumsum()
            if groups.groupby(frame.impression_uid).nunique().max() > 1:
                raise ValueError("Non-contiguous validation impression")
        if pending is not None:
            if frame.impression_uid.iloc[0] != last_uid:
                yield pending
                pending = None
            else:
                frame = pd.concat([pending, frame], ignore_index=True)
        last_uid = frame.impression_uid.iloc[-1]
        pending = frame[frame.impression_uid.eq(last_uid)]
        complete = frame[~frame.impression_uid.eq(last_uid)]
        for _, slate in complete.groupby("impression_uid", sort=False):
            yield slate
    if pending is not None:
        yield pending


def run_check(root, run, *, max_train_rows=200000, validation_impressions=None):
    opened = open_feature_run(root, run)
    schema = opened["schema"]
    # Older artifacts predate categorical metadata. Preserve correct treatment.
    categorical = schema.get(
        "categorical_columns",
        [
            c
            for c in ("category_code", "subcategory_code", "day_part", "weekday")
            if c in schema["model_columns"]
        ],
    )
    excluded = {"weekday", "day_part", "population_startup", "age_left_censored"}
    columns = [name for name in schema["model_columns"] if name not in excluded]
    categorical = [name for name in categorical if name in columns]
    numeric = [name for name in columns if name not in categorical]
    print(
        f"Sampling up to {max_train_rows:,} training rows; fitting logistic smoke model.",
        flush=True,
    )
    training = sample_training_rows(run, columns, max_train_rows)
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = make_pipeline(
        transform, LogisticRegression(max_iter=300, random_state=4222)
    )
    model.fit(training[columns], training.clicked)
    print("Scoring complete validation slates (test labels are not used).", flush=True)
    read_columns = list(dict.fromkeys([*columns, "smoothed_prior_article_ctr"]))
    totals = {
        name: {"auc": 0.0, "auc_slates": 0, "ndcg10": 0.0}
        for name in ("popularity", "logistic")
    }
    count = rows = 0
    for slate in validation_slates(run, read_columns, model, columns):
        scores = {
            "popularity": slate.smoothed_prior_article_ctr.to_numpy(),
            "logistic": slate._logistic_score.to_numpy(),
        }
        for name, score in scores.items():
            auc, ndcg = slate_metrics(slate.clicked, score)
            if auc is not None:
                totals[name]["auc"] += auc
                totals[name]["auc_slates"] += 1
            totals[name]["ndcg10"] += ndcg
        count += 1
        rows += len(slate)
        if count % 10000 == 0:
            print(f"Evaluated {count:,} complete validation impressions.", flush=True)
        if validation_impressions is not None and count >= validation_impressions:
            break
    metrics = {
        name: {
            "impression_auc": values["auc"] / max(1, values["auc_slates"]),
            "ndcg_at_10": values["ndcg10"] / max(1, count),
            "auc_eligible_impressions": values["auc_slates"],
        }
        for name, values in totals.items()
    }
    report = {
        "artifact_schema_version": schema["version"],
        "artifact_manifest_sha256": sha256(run / "run_manifest.json"),
        "training_rows": len(training),
        "validation_impressions": count,
        "validation_rows": rows,
        "complete_validation": validation_impressions is None,
        "model_columns": columns,
        "metrics": metrics,
        "limitations": "Training subset diagnostic; no tuning or calibration. Uses the saved artifact version above. Ranking signal is not proof of no leakage; it does not evaluate retrieval recall or the end-to-end pipeline.",
    }
    write_json(root / "results/model_smoke_check.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-train-rows", type=int, default=200000)
    parser.add_argument("--validation-impressions", type=int)
    args = parser.parse_args()
    if args.max_train_rows < 1 or (
        args.validation_impressions is not None and args.validation_impressions < 1
    ):
        parser.error("Row/impression limits must be positive")
    root = args.root.resolve()
    run_check(
        root,
        root / "4_Feature_Engineering/artifacts/prequential-all",
        max_train_rows=args.max_train_rows,
        validation_impressions=args.validation_impressions,
    )


if __name__ == "__main__":
    main()
