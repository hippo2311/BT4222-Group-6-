"""Run all three representation notebooks on an explicit synthetic dataset."""

# These tests execute trusted notebook source from this repository.
# ruff: noqa: S102

import contextlib
import io
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from newsrec.audits import check_temporal_profiles
from newsrec.common import Settings, saved_stage_ready
from newsrec.io import reconstruct_profile
from newsrec.pipeline import (
    _profile_references,
    build_ranking_dataset,
    build_supplied_profiles,
)
from newsrec.representations import Bundle
from newsrec.temporal import TemporalState


def make_fixture(root):
    processed = root / "1_Data_Preparation/processed"
    processed.mkdir(parents=True)
    annotation = json.dumps([{"WikidataId": "Q1"}, {"WikidataId": "Q1"}])
    pd.DataFrame(
        {
            "news_id": [f"N{i}" for i in range(1, 7)],
            "news_code": range(1, 7),
            "category": ["sports"] * 3 + ["finance"] * 3,
            "subcategory": ["example"] * 6,
            "title": [
                "sports team wins",
                "sports team scores",
                "sports match wins",
                "finance market rises",
                "unseenword onlyvalidation",
                "",
            ],
            "abstract": [
                "team wins sports",
                "",
                "sports match",
                "market rises",
                "unseenword",
                "",
            ],
            "title_entities": [
                annotation,
                "[]",
                "bad json",
                annotation,
                '[{"WikidataId":"Q_missing"}]',
                "[]",
            ],
            "abstract_entities": [annotation] + ["[]"] * 5,
            "in_news_train": [True] * 4 + [False] * 2,
            "in_news_validation": [True] * 5 + [False],
            "in_news_test": [True] * 6,
        }
    ).to_parquet(processed / "news_catalogue.parquet", index=False)
    pd.DataFrame({"user_code": [1, 2]}).to_parquet(
        processed / "users.parquet", index=False
    )
    for i, split in enumerate(("train", "validation", "test")):
        timestamp = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i)
        user = 2 if split == "test" else 1
        pd.DataFrame(
            {
                "impression_uid": [i + 1],
                "user_code": [user],
                "timestamp": [timestamp],
                "history": ["" if user == 2 else "N1 N2 N3 N4 N5"],
            }
        ).to_parquet(processed / f"impressions_{split}.parquet", index=False)
        pd.DataFrame(
            {
                "impression_uid": [i + 1] * 6,
                "user_code": [user] * 6,
                "timestamp": [timestamp] * 6,
                "news_code": range(1, 7),
                "clicked": [1, 0, 0, 0, 0, 0],
            }
        ).to_parquet(processed / f"candidates_{split}.parquet", index=False)
    (root / "train").mkdir()
    (root / "train/entity_embedding.vec").write_text(
        "Q1 " + " ".join(["1"] + ["0"] * 99) + "\n"
    )


class RepresentationTests(unittest.TestCase):
    def test_notebooks_and_saved_contracts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_fixture(root)
            with patch.dict(os.environ, NEWSREC_CODE=str(REPO), NEWSREC_ROOT=str(root)):
                for path in sorted((REPO / "3_Representations").glob("*.ipynb")):
                    namespace = {"__name__": "__main__"}
                    notebook = json.loads(path.read_text())
                    with contextlib.redirect_stdout(io.StringIO()):
                        for i, cell in enumerate(notebook["cells"]):
                            if cell["cell_type"] == "code":
                                exec(
                                    compile(
                                        "".join(cell["source"]),
                                        f"{path.name}:cell{i}",
                                        "exec",
                                    ),
                                    namespace,
                                )
            out = root / "3_Representations/artifacts"
            quality = pd.read_csv(out / "text_quality.csv")
            self.assertGreater(
                quality.query(
                    "split == 'validation' and field == 'title'"
                ).oov_fraction.iloc[0],
                0,
            )
            with (out / "text_transform.pkl").open("rb") as file:
                frozen = pickle.load(file)
            self.assertNotIn("unseenword", frozen["tfidf"].vocabulary_)
            entity = pd.read_parquet(out / "entity_quality.parquet")
            self.assertEqual(entity.n_annotated_entities.iloc[0], 1)
            self.assertEqual(entity.title_entity_status.iloc[2], "invalid")
            usage = pd.read_csv(out / "usage_coverage.csv")
            self.assertTrue(usage.query("cohort == 'slate'").entries.eq(6).all())
            self.assertTrue(usage.query("cohort == 'click'").entries.eq(1).all())
            self.assertEqual(
                usage.query("split == 'test' and cohort == 'history'").entries.iloc[0],
                0,
            )
            self.assertTrue(
                (
                    usage[["both", "text_only", "entity_only", "neither"]].sum(axis=1)
                    == usage.entries
                ).all()
            )
            for stage in ("news", "entity", "availability", "profiles"):
                self.assertTrue(saved_stage_ready(root, Settings(), stage))
            # Execute the build cells again to cover the saved-artifact branches.
            with contextlib.redirect_stdout(io.StringIO()):
                for i in (7, 11):
                    exec("".join(notebook["cells"][i]["source"]), namespace)
            meta_path = out / "news_manifest.json"
            meta = json.loads(meta_path.read_text())
            meta["implementation_hashes"] = {}
            meta_path.write_text(json.dumps(meta))
            self.assertFalse(saved_stage_ready(root, Settings(), "news"))
            bundle = Bundle(root)
            # Check that saved full-history interests round-trip beyond the sequence cap.
            from dataclasses import replace

            cfg = replace(Settings(), history_cap=2)
            state = TemporalState(bundle, "2020-01-01", "2020-01-02", cfg)
            try:
                req = pd.read_parquet(
                    root / "1_Data_Preparation/processed/impressions_train.parquet"
                )
                t = req.timestamp.iloc[0]
                state.begin(req, t)
                profile = state.profile(1, t)
                ref = _profile_references(req, state, t).iloc[0].to_dict()
                restored = reconstruct_profile(ref, bundle)
                np.testing.assert_allclose(restored["interests"], profile["interests"])
            finally:
                state.close()
            # Verify the Arrow reference schema accepts and reloads full-history centroids.
            feature_notebook = json.loads(
                (
                    REPO / "4_Feature_Engineering/4_01_build_ranking_features.ipynb"
                ).read_text()
            )
            feature_namespace = {"__name__": "__main__"}
            with (
                patch.dict(os.environ, NEWSREC_ROOT=str(root)),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                for cell in feature_notebook["cells"]:
                    source = "".join(cell["source"])
                    if cell["cell_type"] == "code":
                        if source.startswith("# A notebook run defaults"):
                            break
                        exec(source, feature_namespace)
            run = build_ranking_dataset(
                root,
                cfg,
                max_timestamps=3,
                state_factory=feature_namespace["TemporalState"],
                feature_builder=feature_namespace["build_features"],
                negative_sampler=feature_namespace["sample_training_groups"],
                schema_builder=feature_namespace["feature_schema"],
            )
            reference_path = next((run / "request_profiles_train").glob("*.parquet"))
            import pyarrow.parquet as pq

            reference = pq.read_table(reference_path).to_pylist()[0]
            restored = reconstruct_profile(reference, bundle)
            self.assertEqual(restored["interests"].shape[1], cfg.text_dim)
            self.assertEqual(reference["selected_interest_categories"][0], "sports")
            # Conflicting histories inside one batch must raise before user deduplication.
            path = root / "1_Data_Preparation/processed/impressions_train.parquet"
            frame = pd.read_parquet(path)
            pd.concat(
                [frame, frame.assign(history="N1")], ignore_index=True
            ).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "Supplied history changed"):
                build_supplied_profiles(root)

    def test_causal_state(self):
        report = check_temporal_profiles(TemporalState, Settings())
        self.assertEqual(report.next_timestamp_clicks.tolist(), [5, 6])


if __name__ == "__main__":
    unittest.main()
