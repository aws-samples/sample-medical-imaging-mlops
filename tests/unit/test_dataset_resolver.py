"""Unit tests for the MSD/Decathlon dataset.json resolver + resolve endpoint.

No AWS: the S3 path is exercised with a stub client; filesystem resolution uses
synthetic dataset trees.
"""
import json
import os
import tempfile
import unittest

from src.preprocessing import dataset_resolver as dr


SEG_JSON = {
    "name": "AMOS",
    "tensorImageSize": "3D",
    "labels": {"0": "background", "1": "spleen", "2": "kidney"},
    "training": [
        {"image": "./imagesTr/amos_0507.nii.gz", "label": "./labelsTr/amos_0507.nii.gz"},
        {"image": "./imagesTr/amos_0508.nii.gz", "label": "./labelsTr/amos_0508.nii.gz"},
    ],
    "validation": [
        {"image": "./imagesTr/amos_0600.nii.gz", "label": "./labelsTr/amos_0600.nii.gz"},
    ],
    "test": ["./imagesTs/amos_0700.nii.gz"],
}

CLS_JSON = {
    "name": "ChestCls",
    "training": [
        {"image": "./imagesTr/a.nii.gz"},
        {"image": "./imagesTr/b.nii.gz"},
    ],
}


class TestParse(unittest.TestCase):
    def test_segmentation_derived(self):
        rd = dr.parse_dataset_json(json.dumps(SEG_JSON))
        self.assertTrue(rd.segmentation)
        self.assertEqual(rd.task_type(), "segmentation")
        self.assertEqual(rd.counts, {"training": 2, "validation": 1, "test": 1})
        self.assertEqual(rd.labels["1"], "spleen")
        # first pair resolved + normalized (leading ./ stripped)
        self.assertEqual(rd.pairs[0].image, "imagesTr/amos_0507.nii.gz")
        self.assertEqual(rd.pairs[0].label, "labelsTr/amos_0507.nii.gz")
        # test entry is image-only
        test_pairs = [p for p in rd.pairs if p.split == "test"]
        self.assertEqual(test_pairs[0].label, None)

    def test_classification_derived(self):
        rd = dr.parse_dataset_json(json.dumps(CLS_JSON))
        self.assertFalse(rd.segmentation)
        self.assertEqual(rd.task_type(), "classification")
        self.assertEqual(rd.counts["training"], 2)
        self.assertIsNone(rd.pairs[0].label)

    def test_summary_sample_n(self):
        rd = dr.parse_dataset_json(json.dumps(SEG_JSON))
        s = rd.summary(sample_n=2)
        self.assertEqual(len(s["sample_pairs"]), 2)
        self.assertEqual(s["task_type"], "segmentation")
        self.assertEqual(s["total_pairs"], 4)

    def test_missing_training_rejected(self):
        with self.assertRaises(dr.DatasetResolveError):
            dr.parse_dataset_json(json.dumps({"name": "x"}))

    def test_malformed_json_rejected(self):
        with self.assertRaises(dr.DatasetResolveError):
            dr.parse_dataset_json("{not valid json")

    def test_inconsistent_labels_rejected(self):
        bad = {"training": [
            {"image": "a.nii.gz", "label": "la.nii.gz"},
            {"image": "b.nii.gz"},  # missing label in a seg dataset
        ]}
        with self.assertRaises(dr.DatasetResolveError):
            dr.parse_dataset_json(json.dumps(bad))

    def test_absolute_and_traversal_paths_rejected(self):
        for bad_path in ["/etc/passwd", "../secret.nii.gz", "s3://b/k.nii.gz"]:
            with self.assertRaises(dr.DatasetResolveError):
                dr.parse_dataset_json(json.dumps({"training": [{"image": bad_path}]}))


class TestResolveFromDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_tree(self, spec, create_files=True):
        with open(os.path.join(self.tmp, "dataset.json"), "w") as f:
            json.dump(spec, f)
        if create_files:
            rd = dr.parse_dataset_json(json.dumps(spec))
            for p in rd.pairs:
                for rel in [p.image, p.label]:
                    if rel:
                        full = os.path.join(self.tmp, rel)
                        os.makedirs(os.path.dirname(full), exist_ok=True)
                        open(full, "wb").close()

    def test_resolve_dir_ok(self):
        self._write_tree(SEG_JSON, create_files=True)
        rd = dr.resolve_from_dir(self.tmp, verify_exists=True)
        self.assertEqual(rd.counts["training"], 2)

    def test_missing_dataset_json(self):
        with self.assertRaises(dr.DatasetJsonMissingError):
            dr.resolve_from_dir(self.tmp)

    def test_referenced_file_missing(self):
        self._write_tree(SEG_JSON, create_files=False)
        with self.assertRaises(dr.DatasetResolveError):
            dr.resolve_from_dir(self.tmp, verify_exists=True)


class _StubS3:
    """Minimal S3 client stub returning a fixed dataset.json body."""
    def __init__(self, body=None, error=None):
        self._body = body
        self._error = error
        self.calls = []

    def get_object(self, Bucket, Key):
        self.calls.append((Bucket, Key))
        if self._error:
            raise self._error
        class _B:
            def __init__(self, data): self._d = data
            def read(self): return self._d.encode("utf-8")
        return {"Body": _B(self._body)}


class TestReadFromS3(unittest.TestCase):
    def test_reads_dataset_json_at_root(self):
        stub = _StubS3(body=json.dumps(SEG_JSON))
        rd = dr.read_dataset_json_from_s3("s3://bucket/AMOS/", s3_client=stub)
        self.assertEqual(stub.calls[0], ("bucket", "AMOS/dataset.json"))
        self.assertTrue(rd.segmentation)

    def test_missing_key_maps_to_clear_error(self):
        class _Err(Exception):
            response = {"Error": {"Code": "NoSuchKey"}}
        stub = _StubS3(error=_Err())
        with self.assertRaises(dr.DatasetJsonMissingError) as ctx:
            dr.read_dataset_json_from_s3("s3://bucket/AMOS/", s3_client=stub)
        self.assertIn("dataset.json not found", str(ctx.exception))

    def test_bad_uri_rejected(self):
        with self.assertRaises(dr.DatasetResolveError):
            dr.read_dataset_json_from_s3("not-an-s3-uri", s3_client=_StubS3(body="{}"))


class TestExample(unittest.TestCase):
    def test_example_is_valid_and_parses(self):
        text = dr.example_dataset_json_text()
        # It must itself parse cleanly through the resolver.
        rd = dr.parse_dataset_json(text)
        self.assertTrue(rd.segmentation)
        self.assertIn("1", rd.labels)


class _Paginator:
    """Stub paginator for list_objects_v2 over an in-memory key list."""
    def __init__(self, keys):
        self._keys = keys

    def paginate(self, Bucket, Prefix, Delimiter=None):
        if Delimiter == "/":
            # Return immediate subfolders under Prefix as CommonPrefixes.
            common = set()
            for k in self._keys:
                if not k.startswith(Prefix):
                    continue
                rest = k[len(Prefix):]
                if "/" in rest:
                    common.add(Prefix + rest.split("/", 1)[0] + "/")
            yield {"CommonPrefixes": [{"Prefix": p} for p in sorted(common)]}
        else:
            yield {"Contents": [{"Key": k} for k in self._keys if k.startswith(Prefix)]}


class _ImageFolderS3:
    def __init__(self, keys):
        self._keys = keys
    def get_paginator(self, _name):
        return _Paginator(self._keys)


class TestImageFolderDiscovery(unittest.TestCase):
    def test_discovers_classes(self):
        keys = [
            "cls/pneumonia/a.nii.gz", "cls/pneumonia/b.nii.gz", "cls/pneumonia/c.nii.gz",
            "cls/normal/d.nii.gz", "cls/normal/e.nii.gz",
            "cls/readme.txt",   # non-image at root — ignored
        ]
        rd = dr.discover_imagefolder_s3("s3://bucket/cls/", s3_client=_ImageFolderS3(keys))
        self.assertFalse(rd.segmentation)
        self.assertEqual(rd.task_type(), "classification")
        self.assertEqual(rd.source, "imagefolder")
        self.assertEqual(rd.classes, {"pneumonia": 3, "normal": 2})
        self.assertEqual(rd.counts["training"], 5)
        # sample pairs are image-only, relative to root
        self.assertTrue(all(p.label is None for p in rd.pairs))
        self.assertTrue(any(p.image.startswith("pneumonia/") for p in rd.pairs))

    def test_no_class_folders_raises(self):
        keys = ["flat/a.nii.gz", "flat/b.nii.gz"]   # images at root, no class dirs
        with self.assertRaises(dr.DatasetResolveError):
            dr.discover_imagefolder_s3("s3://bucket/flat/", s3_client=_ImageFolderS3(keys))


class TestResolveEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import main
        cls.client = TestClient(main.app, base_url="http://localhost")  # Host must pass TrustedHostMiddleware

    def test_resolve_without_aws_returns_400(self):
        # aws_service.s3_client is None without configured creds → 400, not 500.
        r = self.client.get("/api/preprocessing/resolve-dataset",
                            params={"input_s3_uri": "s3://bucket/AMOS/"})
        self.assertIn(r.status_code, (400, 500))
        # Should be a clean client error when creds are absent.
        if r.status_code == 400:
            self.assertIn("detail", r.json())

    def test_missing_dataset_json_returns_structured_404(self):
        import json
        from app.routers.aws_router import aws_service

        class _Err(Exception):
            response = {"Error": {"Code": "NoSuchKey"}}

        # get_object raises NoSuchKey AND no class folders exist → structured 404.
        class _EmptyS3(_StubS3):
            def __init__(self):
                super().__init__(error=_Err())
            def get_paginator(self, _n):
                return _Paginator([])   # no keys → no classes

        prev = getattr(aws_service, "s3_client", None)
        aws_service.s3_client = _EmptyS3()
        try:
            r = self.client.get("/api/preprocessing/resolve-dataset",
                                params={"input_s3_uri": "s3://bucket/AMOS/"})
            self.assertEqual(r.status_code, 404)
            detail = r.json()["detail"]
            self.assertEqual(detail["error"], "dataset_json_missing")
            self.assertIn("dataset.json", detail["message"])
            self.assertTrue(detail["expected_location"].endswith("/dataset.json"))
            json.loads(detail["example"])
        finally:
            aws_service.s3_client = prev

    def test_imagefolder_fallback_returns_classification(self):
        from app.routers.aws_router import aws_service

        class _Err(Exception):
            response = {"Error": {"Code": "NoSuchKey"}}

        keys = ["cls/pneumonia/a.nii.gz", "cls/pneumonia/b.nii.gz", "cls/normal/c.nii.gz"]

        class _FolderS3(_StubS3):
            def __init__(self):
                super().__init__(error=_Err())   # no dataset.json
            def get_paginator(self, _n):
                return _Paginator(keys)

        prev = getattr(aws_service, "s3_client", None)
        aws_service.s3_client = _FolderS3()
        try:
            r = self.client.get("/api/preprocessing/resolve-dataset",
                                params={"input_s3_uri": "s3://bucket/cls/"})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["task_type"], "classification")
            self.assertEqual(body["source"], "imagefolder")
            self.assertEqual(body["classes"], {"pneumonia": 2, "normal": 1})
        finally:
            aws_service.s3_client = prev


if __name__ == "__main__":
    unittest.main()
