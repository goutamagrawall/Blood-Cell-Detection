"""Regression checks for portable paths, annotations, and class counting."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from main import check_dataset, count_cells, read_dataset, resolved_yaml


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data.yaml"
        self.data.write_text("train: train/images\nval: valid/images\ntest: test/images\n"
                             "nc: 3\nnames: [RBC, WBC, platelets]\n", encoding="utf-8")
        for split in ("train", "valid", "test"):
            image_dir = self.root / split / "images"
            label_dir = self.root / split / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir()
            cv2.imwrite(str(image_dir / "cell.png"), np.zeros((16, 16, 3), dtype=np.uint8))
            (label_dir / "cell.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def test_paths_resolve_from_config_location(self):
        config = read_dataset(self.data)
        self.assertEqual(Path(config["train"]), self.root / "train/images")
        generated = resolved_yaml(self.data, self.root / "output")
        self.assertEqual(read_dataset(generated)["test"], str(self.root / "test/images"))
        self.assertNotIn(str(self.root), self.data.read_text(encoding="utf-8"))

    def test_complete_dataset(self):
        summary = check_dataset(self.data)
        self.assertEqual(summary["test"], {"images": 1, "annotations": {"RBC": 1}})
        json.dumps(summary)

    def test_invalid_box_is_rejected(self):
        for annotation in ("0 nan 0.5 0.2 0.2", "3 0.5 0.5 0.2 0.2",
                           "0 0.5 0.5 -0.2 0.2", "0 0.5 0.5 0.2"):
            with self.subTest(annotation=annotation):
                (self.root / "train/labels/cell.txt").write_text(annotation, encoding="utf-8")
                with self.assertRaises(ValueError):
                    check_dataset(self.data)

    def test_missing_label_is_rejected(self):
        (self.root / "train/labels/cell.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            check_dataset(self.data)

    def test_unreadable_image_is_rejected(self):
        (self.root / "train/images/cell.png").write_bytes(b"invalid image")
        with self.assertRaisesRegex(ValueError, "Cannot decode"):
            check_dataset(self.data)


class CountingTests(unittest.TestCase):
    def test_counts_use_model_class_mapping(self):
        result = SimpleNamespace(names={0: "platelets", 1: "WBC", 2: "RBC"},
                                 boxes=SimpleNamespace(cls=np.array([2, 2, 0])))
        self.assertEqual(count_cells(result), {"platelets": 1, "WBC": 0, "RBC": 2})

    def test_empty_detection_preserves_all_classes(self):
        result = SimpleNamespace(names={0: "RBC", 1: "WBC", 2: "platelets"},
                                 boxes=SimpleNamespace(cls=np.array([])))
        self.assertEqual(count_cells(result), {"RBC": 0, "WBC": 0, "platelets": 0})


if __name__ == "__main__":
    unittest.main()
