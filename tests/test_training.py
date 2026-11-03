import json
from dataclasses import asdict, replace

import pandas as pd
import pytest
from notebook_support import api

from newsrec.common import Settings
from newsrec.io import feature_run
from newsrec.ranking import sample_training_groups
from newsrec.training import iter_training_epoch


def test_epoch_draws_reproduce_and_vary_within_logged_slate():
    frame = pd.DataFrame(
        {
            "impression_uid": [10] * 32,
            "news_code": range(32),
            "clicked": [1, 1] + [0] * 30,
        }
    )
    a, selected = sample_training_groups(frame, epoch=3)
    b, shuffled = sample_training_groups(frame.sample(frac=1, random_state=7), epoch=3)
    pd.testing.assert_frame_equal(a, b)
    assert selected == shuffled
    assert selected != sample_training_groups(frame, epoch=4)[1]
    assert {(10, 0), (10, 1)} <= selected
    assert all(
        len(set(ids)) == 4 and set(ids) <= set(range(2, 32))
        for ids in a.negative_news_codes
    )
    for epoch in (-1, 1.5, True):
        with pytest.raises(ValueError, match="epoch"):
            sample_training_groups(frame, epoch=epoch)
    groups, pairs = sample_training_groups(frame.iloc[:2], epoch=1)
    assert pairs == {(10, 0), (10, 1)}
    assert all(not ids for ids in groups.negative_news_codes)


def test_build_retains_full_pool_and_fixed_mode_still_works(dataset, config):
    cfg = replace(config, negative_count=1)
    run = api.build_ranking_dataset(dataset, cfg)
    source = pd.read_parquet(
        dataset / "1_Data_Preparation/processed/candidates_train.parquet"
    )
    assert len(pd.read_parquet(run / "ranking_features_train")) == len(source)
    validation = pd.read_parquet(run / "ranking_features_validation")
    summary = json.loads((run / "sampling_summary.json").read_text())
    assert summary["discarded_unique_negative_fraction"] == 0
    batches = list(iter_training_epoch(run, 2, columns=["text_cosine"]))
    assert sum(int(frame.clicked.sum()) for frame, _ in batches) == int(
        source.clicked.sum()
    )
    assert sum(len(groups) for _, groups in batches) == int(source.clicked.sum())
    assert sum(len(frame) for frame, _ in batches) < len(source)
    with pytest.raises(ValueError, match="only model_columns"):
        list(iter_training_epoch(run, 0, columns=["clicked"]))
    fixed = replace(cfg, training_negative_sampling="fixed")
    # Cached full-pool artifacts cannot silently serve an incompatible config.
    with pytest.raises(ValueError, match="settings differ"):
        feature_run(dataset, fixed, schema_builder=api.feature_schema)
    run = api.build_ranking_dataset(dataset, fixed, overwrite=True)
    assert len(pd.read_parquet(run / "ranking_features_train")) < len(source)
    pd.testing.assert_frame_equal(
        validation, pd.read_parquet(run / "ranking_features_validation")
    )
    with pytest.raises(ValueError, match="complete training pool"):
        list(iter_training_epoch(run, 0))


def test_epoch_reader_carries_slates_across_parts(tmp_path):
    cfg = Settings()
    (tmp_path / "run_status.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "full_run": True,
                "training_candidate_pool": "complete",
                "config": asdict(cfg),
            }
        )
    )
    (tmp_path / "feature_schema.json").write_text(
        json.dumps({"model_columns": ["score"]})
    )
    (tmp_path / "ranking_features_train").mkdir()
    (tmp_path / "labels_train").mkdir()
    # IDs deliberately decrease across impressions; the middle slate spans parts.
    full = pd.DataFrame(
        [
            (uid, article, int(article < 2), float(article))
            for uid in (10, 9, 11)
            for article in range(22)
        ],
        columns=["impression_uid", "news_code", "clicked", "score"],
    )
    for i, frame in enumerate((full.iloc[:25], full.iloc[25:35], full.iloc[35:])):
        name = f"part-{i:05d}.parquet"
        frame.drop(columns="clicked").to_parquet(
            tmp_path / "ranking_features_train" / name, index=False
        )
        frame.drop(columns="score").to_parquet(
            tmp_path / "labels_train" / name, index=False
        )
    for epoch in (0, 1, 3):
        batches = list(iter_training_epoch(tmp_path, epoch))
        actual = pd.concat([frame for frame, _ in batches], ignore_index=True)
        actual_groups = pd.concat([groups for _, groups in batches], ignore_index=True)
        expected_groups, selected = sample_training_groups(full, cfg, epoch=epoch)
        assert set(zip(actual.impression_uid, actual.news_code)) == selected
        assert actual.clicked.sum() == 6
        assert not actual.duplicated(["impression_uid", "news_code"]).any()
        pd.testing.assert_frame_equal(
            actual_groups.sort_values(
                ["impression_uid", "positive_news_code"]
            ).reset_index(drop=True),
            expected_groups,
        )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "full_run": True,
                "config": asdict(cfg),
            }
        )
    )
    with pytest.raises(ValueError, match="Rebuild"):
        list(iter_training_epoch(tmp_path, 0))
