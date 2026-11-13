"""Unit tests for the generic MONAI Bundle support (src/monai_bundle)."""
import json
import os
import sys
import tarfile
import tempfile
import unittest

# Make src/monai_bundle importable (mirrors how SageMaker flattens /opt/program).
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src", "monai_bundle")
sys.path.insert(0, os.path.abspath(_SRC))

import bundle_utils  # noqa: E402
import datalist_gen  # noqa: E402
import package  # noqa: E402


def _make_bundle(root, nested=False, with_train=True, with_infer=True, with_ckpt=True):
    base = os.path.join(root, "NV-Bundle") if nested else root
    cfg = os.path.join(base, "configs")
    os.makedirs(cfg, exist_ok=True)
    with open(os.path.join(cfg, "metadata.json"), "w") as f:
        json.dump({"task": "segmentation", "version": "1.0"}, f)
    if with_train:
        with open(os.path.join(cfg, "train_continual_softmax.json"), "w") as f:
            json.dump({"epochs": 100}, f)
    if with_infer:
        with open(os.path.join(cfg, "inference.json"), "w") as f:
            json.dump({}, f)
    if with_ckpt:
        models = os.path.join(base, "models")
        os.makedirs(models, exist_ok=True)
        with open(os.path.join(models, "model.pt"), "wb") as f:
            f.write(b"\x00")
    return base


class TestInspectBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_valid_flat_bundle(self):
        _make_bundle(self.tmp)
        info = bundle_utils.inspect_bundle(self.tmp)
        self.assertTrue(info.valid)
        self.assertEqual(info.task, "segmentation")
        self.assertIn("train_continual_softmax.json", info.train_configs)
        self.assertEqual(info.default_inference_config(), "inference.json")
        self.assertEqual(info.default_train_config(), "train_continual_softmax.json")
        self.assertEqual(len(info.checkpoints), 1)

    def test_nested_bundle(self):
        _make_bundle(self.tmp, nested=True)
        info = bundle_utils.inspect_bundle(self.tmp)
        self.assertTrue(info.valid)
        self.assertTrue(info.configs_dir.endswith(os.path.join("NV-Bundle", "configs")))

    def test_invalid_bundle(self):
        # empty dir -> no configs
        info = bundle_utils.inspect_bundle(self.tmp)
        self.assertFalse(info.valid)

    def test_train_config_classification(self):
        _make_bundle(self.tmp)
        info = bundle_utils.inspect_bundle(self.tmp)
        # softmax config classified as training via "continual"/"train"
        self.assertIn("train_continual_softmax.json", info.train_configs)


class TestDatalistGen(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "dataset.json"), "w") as f:
            json.dump({
                "labels": {"0": "background", "1": "pancreas", "2": "cancer"},
                "training": [
                    {"image": f"./imagesTr/p_{i}.nii.gz", "label": f"./labelsTr/p_{i}.nii.gz"}
                    for i in range(10)
                ],
            }, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_split_and_normalize(self):
        dl = datalist_gen.build_datalist(self.tmp, val_fraction=0.2, seed=42)
        self.assertEqual(len(dl["training"]) + len(dl["validation"]), 10)
        self.assertEqual(len(dl["validation"]), 2)
        # paths normalized: no leading ./
        self.assertFalse(dl["training"][0]["image"].startswith("./"))
        self.assertEqual(dl["labels"]["1"], "pancreas")

    def test_deterministic_seed(self):
        a = datalist_gen.build_datalist(self.tmp, seed=1)
        b = datalist_gen.build_datalist(self.tmp, seed=1)
        self.assertEqual(a["validation"], b["validation"])

    def test_missing_dataset_json(self):
        empty = tempfile.mkdtemp()
        with self.assertRaises(FileNotFoundError):
            datalist_gen.build_datalist(empty)


class TestPackage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parse_s3_uri_valid(self):
        b, k = package.parse_s3_uri("s3://my-bucket/path/model.tar.gz")
        self.assertEqual(b, "my-bucket")
        self.assertEqual(k, "path/model.tar.gz")

    def test_parse_s3_uri_invalid(self):
        for bad in ("https://x/y", "s3://bucket-only", "s3://"):
            with self.assertRaises(ValueError):
                package.parse_s3_uri(bad)

    def test_package_valid_bundle(self):
        _make_bundle(self.tmp)
        tar_path = os.path.join(self.tmp, "model.tar.gz")
        package.package_bundle(self.tmp, tar_path)
        self.assertTrue(os.path.isfile(tar_path))
        with tarfile.open(tar_path) as tar:
            names = tar.getnames()
        self.assertTrue(any(n.startswith("configs") for n in names))
        self.assertTrue(any(n.startswith("models") for n in names))

    def test_package_invalid_bundle_rejected(self):
        tar_path = os.path.join(self.tmp, "model.tar.gz")
        with self.assertRaises(ValueError):
            package.package_bundle(self.tmp, tar_path)  # empty -> not a bundle


if __name__ == "__main__":
    unittest.main()
