"""Tests for the segmentation branch of the pre-training validator."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.validation_service import _validate_segmentation_structure  # noqa: E402


class _FakePaginator:
    def __init__(self, keys):
        self._keys = keys

    def paginate(self, Bucket, Prefix, Delimiter=None):
        contents = [{"Key": k, "Size": 1024 * 1024} for k in self._keys if k.startswith(Prefix)]
        yield {"Contents": contents, "KeyCount": len(contents)}


class _FakeS3:
    def __init__(self, keys):
        self._keys = keys

    def get_paginator(self, name):
        return _FakePaginator(self._keys)


def _case(split, case, files=("img.nii.gz", "label.nii.gz")):
    return [f"data/{split}/{case}/{f}" for f in files]


class SegmentationValidationTest(unittest.TestCase):
    def test_reports_cases_per_split(self):
        keys = _case("train", "c1") + _case("train", "c2") + _case("val", "c3") + _case("test", "c4")
        checks, stats, ok = _validate_segmentation_structure(
            _FakeS3(keys), "b", "data", ["test", "train", "val"])
        self.assertTrue(ok)
        self.assertEqual(stats["layout"], "segmentation")
        self.assertEqual(list(stats["splits"]), ["train", "val", "test"])
        self.assertEqual(stats["splits"]["train"]["cases"], 2)
        self.assertEqual(stats["splits"]["val"]["cases"], 1)
        self.assertEqual(stats["splits"]["test"]["cases"], 1)

    def test_valid_folder_accepted_as_val(self):
        keys = _case("train", "c1") + _case("valid", "c2")
        _, stats, ok = _validate_segmentation_structure(_FakeS3(keys), "b", "data", ["train", "valid"])
        self.assertTrue(ok)
        self.assertIn("valid", stats["splits"])

    def test_missing_val_fails(self):
        checks, _, ok = _validate_segmentation_structure(
            _FakeS3(_case("train", "c1")), "b", "data", ["train"])
        self.assertFalse(ok)
        self.assertIn("val", checks[0]["message"])

    def test_case_missing_label_fails(self):
        keys = _case("train", "c1") + _case("train", "c2", files=("img.nii.gz",)) + _case("val", "c3")
        checks, stats, ok = _validate_segmentation_structure(
            _FakeS3(keys), "b", "data", ["train", "val"])
        self.assertFalse(ok)
        self.assertEqual(stats["splits"]["train"]["num_incomplete"], 1)
        self.assertTrue(any("c2" in c["message"] for c in checks if not c["passed"]))


if __name__ == "__main__":
    unittest.main()
