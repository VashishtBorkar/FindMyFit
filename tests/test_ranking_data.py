"""Regression checks for official partitions and sampled retrieval candidates."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.metric_learning.ranking_data import make_ranking_splits


class RankingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.embeddings = {}
        for split, filename in (
            ("train", "train_no_dup.json"),
            ("validation", "valid_no_dup.json"),
            ("test", "test_no_dup.json"),
        ):
            rows = []
            for index in range(8):
                set_id = f"{split}-{index}"
                rows.append({"set_id": set_id, "items": [{"index": 1}, {"index": 2}]})
                for item_index, category in ((1, "top"), (2, "pants")):
                    item = f"{set_id}_{item_index}"
                    self.embeddings[item] = {
                        "category": category, "image_hash": item,
                        "embedding": np.ones(4, dtype=np.float32),
                    }
            (self.root / filename).write_text(json.dumps(rows))

    def prepare(self):
        return make_ranking_splits(
            self.root, self.embeddings, seed=42, candidate_count=3,
            anchor_category="top", target_categories=["pants"],
        )

    def test_official_splits_and_candidates_are_reproducible(self):
        examples, audit = self.prepare()
        self.assertEqual((examples, audit), self.prepare())
        self.assertTrue(all(value == 0 for value in audit["original_image_pair_overlap"].values()))
        for split, rows in examples.items():
            self.assertEqual(len(rows), 8)
            for anchor, category, candidates, positive_index in rows:
                self.assertTrue(anchor.startswith(split))
                self.assertEqual(category, "pants")
                self.assertEqual(len(set(candidates)), 3)
                self.assertEqual(candidates[positive_index], anchor[:-1] + "2")
                self.assertTrue(all(item.startswith(split) for item in candidates))
                self.assertTrue(all(self.embeddings[item]["category"] == category for item in candidates))

    def test_duplicate_images_are_reserved_for_heldout_partition(self):
        self.embeddings["train-0_1"]["image_hash"] = "test-0_1"
        examples, audit = self.prepare()
        self.assertEqual(audit["original_image_overlap"]["train/test"], 1)
        self.assertEqual(audit["items_removed_for_heldout_overlap"]["train"], 1)
        self.assertTrue(all(value == 0 for value in audit["retained_image_overlap"].values()))
        self.assertFalse(any(row[0] == "train-0_1" for row in examples["train"]))
        self.assertTrue(any(row[0] == "test-0_1" for row in examples["test"]))

    def test_other_observed_partners_are_excluded_from_negatives(self):
        path = self.root / "train_no_dup.json"
        rows = json.loads(path.read_text())
        rows[0]["items"].append({"index": 3})
        self.embeddings["train-0_3"] = {
            "category": "pants", "image_hash": "alternate-pants",
            "embedding": np.ones(4, dtype=np.float32),
        }
        path.write_text(json.dumps(rows))
        examples, _ = self.prepare()
        for anchor, _, candidates, positive_index in examples["train"]:
            if anchor == "train-0_1":
                negatives = set(candidates) - {candidates[positive_index]}
                self.assertFalse(negatives & {"train-0_2", "train-0_3"})


if __name__ == "__main__":
    unittest.main()
