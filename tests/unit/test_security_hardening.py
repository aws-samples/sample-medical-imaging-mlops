"""Regression tests for the ProbeScan (2026-11-12) security fixes.

Covers: request-origin middleware (DNS rebinding / CSRF), the static_script
template helper, MONAI CLI argument validation in the inference handlers,
safe tar extraction, checkpoint path confinement, and requirement-line
sanitisation.
"""
import importlib.util
import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path, extra_path: Path = None):
    """Import a container-side module by file path (they are not packages)."""
    if extra_path and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRequestOriginMiddleware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main
        cls.main = main
        cls.client = TestClient(main.app, base_url="http://localhost")

    def test_loopback_host_allowed(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_rebound_host_rejected(self):
        r = self.client.get("/health", headers={"host": "attacker.example"})
        self.assertEqual(r.status_code, 400)

    def test_cross_site_post_rejected(self):
        r = self.client.post("/api/does-not-exist", headers={"sec-fetch-site": "cross-site"})
        self.assertEqual(r.status_code, 403)

    def test_foreign_origin_post_rejected(self):
        r = self.client.post("/api/does-not-exist", headers={"origin": "https://attacker.example"})
        self.assertEqual(r.status_code, 403)

    def test_same_origin_post_passes_middleware(self):
        r = self.client.post("/api/does-not-exist",
                             headers={"origin": "http://localhost", "sec-fetch-site": "same-origin"})
        self.assertIn(r.status_code, (404, 405))   # reached routing, not blocked

    def test_non_browser_post_passes_middleware(self):
        r = self.client.post("/api/does-not-exist")
        self.assertIn(r.status_code, (404, 405))

    def test_static_script_helper(self):
        tag = str(self.main.static_script("js/main.js"))
        self.assertTrue(tag.startswith('<script src="/static/js/main.js?v='))
        for bad in ("../main.py", "js/x.js\"><script>", "/etc/passwd.js", "js/a.py"):
            with self.assertRaises(ValueError):
                self.main.static_script(bad)


class TestMonaiBundleInferenceValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load("mb_inference", ROOT / "src/monai_bundle/inference.py", ROOT / "src/monai_bundle")

    def test_numeric_prompts(self):
        self.mod._check_numeric([1, 2, [3.5, 4]], "points")
        for bad in ("$__import__('os')", ["$x"], [True], {"a": 1}):
            with self.assertRaises(ValueError):
                self.mod._check_numeric(bad, "label_prompt")

    def test_overrides_reject_expressions_and_reserved_keys(self):
        self.assertEqual(self.mod._validated_overrides({"num_workers": 2, "foo#bar": "x"}),
                         {"num_workers": 2, "foo#bar": "x"})
        for bad in ({"x": "$__import__('os').system('id')"}, {"x": "@network_def"},
                    {"x": "%configs/other.json#net"}, {"x": ["ok", "$bad"]},
                    {"bundle_root": "/tmp"}, {"output_dir": "/"}, {"--evil": 1}, {"a b": 1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.mod._validated_overrides(bad)


class TestNvmedtechInferenceValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load("nv_inference", ROOT / "src/nvmedtech/inference.py")

    def test_checkpoint_confined_to_models_dir(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "models"))
            ok = os.path.join(d, "models", "model_finetune.pt")
            Path(ok).write_bytes(b"x")
            Path(d, "secret.pt").write_bytes(b"x")
            self.assertEqual(self.mod._resolve_checkpoint(d, "models/model_finetune.pt"), os.path.realpath(ok))
            for bad in ("../secret.pt", "/etc/passwd", "models/../../secret.pt", "models/missing.pt"):
                with self.assertRaises(ValueError, msg=bad):
                    self.mod._resolve_checkpoint(d, bad)

    def test_prompts_numeric_only(self):
        with self.assertRaises(ValueError):
            self.mod._check_numeric(["$__import__('os')"], "label_prompt")


class TestSafeExtractAndRequirements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = ROOT / "src/monai_bundle"
        sys.path.insert(0, str(src))
        cls.mod = _load("mb_train", src / "train.py", src)

    @staticmethod
    def _tar_with(name: str, linkname: str = None) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name)
            if linkname:
                info.type = tarfile.SYMTYPE
                info.linkname = linkname
                tar.addfile(info)
            else:
                data = b"payload"
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    def _extract(self, blob: bytes, dest: str):
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            self.mod._safe_extract(tar, dest)

    def test_traversal_and_links_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for blob in (self._tar_with("../escape.txt"), self._tar_with("/abs.txt"),
                         self._tar_with("link", linkname="../../etc/passwd")):
                with self.assertRaises(tarfile.TarError):
                    self._extract(blob, d)
            self._extract(self._tar_with("configs/ok.json"), d)
            self.assertTrue(os.path.isfile(os.path.join(d, "configs", "ok.json")))

    def test_requirement_line_filter(self):
        ok = ["monai==1.6.1", "timm", "pytorch-ignite>=0.4.11", "numpy<2",
              "fire>=0.5.0,<1", "pkg[extra]==1.0", "x==1; python_version >= '3.10'"]
        bad = ["--index-url https://evil.example/simple", "-e git+https://evil/x.git",
               "evil @ https://evil.example/x.whl", "-r other.txt", "https://evil.example/x.whl"]
        for line in ok:
            self.assertTrue(self.mod._REQ_LINE_RE.match(line), line)
        for line in bad:
            self.assertFalse(self.mod._REQ_LINE_RE.match(line), line)


class TestSlingshotInputConfinement(unittest.TestCase):
    """Slingshot 2026-11-12: local-path reads, S3 bucket gate, filename patterns,
    evaluation cache paths, bucket ownership."""

    def setUp(self):
        self._env = dict(os.environ)
        self.tmp = tempfile.TemporaryDirectory()
        self.allowed = os.path.join(self.tmp.name, "input")
        os.makedirs(self.allowed)
        self.inside = os.path.join(self.allowed, "case.dcm")
        Path(self.inside).write_bytes(b"x")
        self.outside = os.path.join(self.tmp.name, "secret.txt")
        Path(self.outside).write_bytes(b"x")
        os.environ["LOCAL_INPUT_ROOTS"] = self.allowed

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        self.tmp.cleanup()

    def _handlers(self):
        mods = [_load("cls_inf", ROOT / "src/inference/classification/inference.py"),
                _load("cls_inf_train", ROOT / "src/classification/inference.py"),
                _load("mb_inf2", ROOT / "src/monai_bundle/inference.py", ROOT / "src/monai_bundle")]
        try:
            mods.append(_load("seg_inf", ROOT / "src/inference/segmentation/inference.py"))
        except ImportError:  # optional heavy deps
            pass
        return mods

    def test_local_paths_confined(self):
        for mod in self._handlers():
            self.assertEqual(mod._validated_local_path(self.inside), os.path.realpath(self.inside))
            for bad in (self.outside, os.path.join(self.allowed, "..", "secret.txt"), "/etc/passwd"):
                with self.assertRaises((ValueError, FileNotFoundError), msg=f"{mod.__name__}:{bad}"):
                    mod._validated_local_path(bad)

    def test_classification_input_fn_rejects_outside_path(self):
        mod = _load("cls_inf3", ROOT / "src/inference/classification/inference.py")
        with self.assertRaises(ValueError):
            mod.input_fn('{"file_path": "/etc/passwd"}', "application/json")
        self.assertEqual(mod.input_fn('{"file_path": "%s"}' % self.inside, "application/json")["local_path"],
                         os.path.realpath(self.inside))

    def test_monai_bundle_base64_jpeg_not_treated_as_path(self):
        mod = _load("mb_inf3", ROOT / "src/monai_bundle/inference.py", ROOT / "src/monai_bundle")
        self.assertFalse(mod._is_confined_local_path("/9j/4AAQSkZJRgABAQ"))
        self.assertFalse(mod._is_confined_local_path("/etc/passwd"))
        self.assertTrue(mod._is_confined_local_path(self.inside))

    def test_s3_bucket_allow_list(self):
        mod = _load("cls_inf4", ROOT / "src/inference/classification/inference.py")
        self.assertEqual(mod._s3_bucket_key("s3://any-bucket/k.dcm"), ("any-bucket", "k.dcm"))
        os.environ["ALLOWED_INPUT_BUCKETS"] = "good-bucket"
        self.assertEqual(mod._s3_bucket_key("s3://good-bucket/a/b.dcm"), ("good-bucket", "a/b.dcm"))
        for bad in ("s3://other-bucket/x", "s3://good-bucket/", "https://good-bucket/x"):
            with self.assertRaises(ValueError):
                mod._s3_bucket_key(bad)

    def test_output_pattern_restricted(self):
        mod = _load("cls_inf5", ROOT / "src/inference/classification/inference.py")
        d = "{patient_id}.json"
        self.assertEqual(mod._safe_output_pattern("{patient_id}_{timestamp}.json", d), "{patient_id}_{timestamp}.json")
        for bad in ("{patient_id.__class__}", "{0}", "{x}", "{patient_id!r}", "../{patient_id}", "/{patient_id}"):
            self.assertEqual(mod._safe_output_pattern(bad, d), d, bad)

    def test_evaluation_cache_paths_confined(self):
        from app.services.evaluation_service import EvaluationService
        svc = EvaluationService.__new__(EvaluationService)
        svc.RESULTS_DIR = Path(self.tmp.name) / "results"
        svc.RESULTS_DIR.mkdir()
        self.assertEqual(svc._job_results_dir("eval-ep-20261112-010203").name, "eval-ep-20261112-010203")
        for bad in ("..", "../..", "/etc", "a/../../b", "", "x" * 70):
            with self.assertRaises(ValueError, msg=bad):
                svc._job_results_dir(bad)

    def test_bucket_owner_check(self):
        from botocore.exceptions import ClientError
        from app.services import s3_ownership

        class FakeS3:
            def __init__(self, code=None):
                self.code, self.calls = code, []

            def head_bucket(self, **kw):
                self.calls.append(kw)
                if self.code:
                    raise ClientError({"Error": {"Code": self.code}}, "HeadBucket")

        os.environ["AWS_ACCOUNT_ID"] = "123456789012"
        ok = FakeS3()
        s3_ownership.assert_bucket_owned(ok, "mine")
        self.assertEqual(ok.calls[0], {"Bucket": "mine", "ExpectedBucketOwner": "123456789012"})
        with self.assertRaises(s3_ownership.BucketOwnershipError):
            s3_ownership.assert_bucket_owned(FakeS3("403"), "squatted")
        del os.environ["AWS_ACCOUNT_ID"]
        os.environ["REQUIRE_BUCKET_OWNER_CHECK"] = "true"
        with self.assertRaises(s3_ownership.BucketOwnershipError):
            s3_ownership.assert_bucket_owned(FakeS3(), "mine")


if __name__ == "__main__":
    unittest.main()
