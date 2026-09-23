"""Tests for batch preprocessing: preprocess.py end-to-end (local, no AWS) and
PreprocessingService.submit_batch with mocked SageMaker/S3 clients.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import nibabel as nib

from src.preprocessing import preprocess as pp
from app.services.preprocessing_service import PreprocessingService


def _write_nii(path, shape=(32, 32, 16), value=200.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vol = np.zeros(shape, dtype=np.float32)
    vol[5:25, 5:25, 3:12] = value
    nib.save(nib.Nifti1Image(vol, np.diag([1.0, 1.0, 1.5, 1.0])), path)


SPEC = {
    "task_type": "segmentation",
    "keys": ["image", "label"],
    "steps": [
        {"type": "load", "params": {}},
        {"type": "ensure_channel_first", "params": {}},
        {"type": "resize", "params": {"spatial_size": [16, 16, 8]}},
    ],
}


class TestPreprocessBatchSegmentation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.inp = os.path.join(self.tmp, "in")
        self.out = os.path.join(self.tmp, "out")
        # A tiny Decathlon-style dataset with 2 training pairs + 1 test image.
        for cid in ("case_1", "case_2"):
            _write_nii(os.path.join(self.inp, "imagesTr", f"{cid}.nii.gz"))
            _write_nii(os.path.join(self.inp, "labelsTr", f"{cid}.nii.gz"), value=1.0)
        _write_nii(os.path.join(self.inp, "imagesTs", "case_9.nii.gz"))
        dataset = {
            "name": "Tiny", "labels": {"0": "bg", "1": "fg"},
            "training": [
                {"image": "./imagesTr/case_1.nii.gz", "label": "./labelsTr/case_1.nii.gz"},
                {"image": "./imagesTr/case_2.nii.gz", "label": "./labelsTr/case_2.nii.gz"},
            ],
            "test": ["./imagesTs/case_9.nii.gz"],
        }
        with open(os.path.join(self.inp, "dataset.json"), "w") as f:
            json.dump(dataset, f)
        self.spec_path = os.path.join(self.tmp, "spec.json")
        with open(self.spec_path, "w") as f:
            json.dump(SPEC, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_end_to_end(self):
        result = pp.main([
            "--spec-path", self.spec_path,
            "--dataset-input", self.inp,
            "--output", self.out,
        ])
        self.assertEqual(result["processed"], 3)   # 2 train + 1 test
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["task_type"], "segmentation")

        # Output dataset.json written and points at processed files.
        with open(os.path.join(self.out, "dataset.json")) as f:
            od = json.load(f)
        self.assertEqual(len(od["training"]), 2)
        self.assertEqual(len(od["test"]), 1)
        self.assertIn("label", od["training"][0])

        # Processed image files exist and were resized.
        img0 = os.path.join(self.out, "imagesTr", "case_1.nii.gz")
        self.assertTrue(os.path.isfile(img0))
        arr = nib.load(img0).get_fdata()
        self.assertEqual(tuple(arr.shape[-3:]), (16, 16, 8))
        # result.json + disclaimer present
        with open(os.path.join(self.out, "result.json")) as f:
            res = json.load(f)
        self.assertIn("RESEARCH USE ONLY", res["disclaimer"])


class TestPreprocessBatchImageFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.inp = os.path.join(self.tmp, "in")
        self.out = os.path.join(self.tmp, "out")
        for cls in ("normal", "abnormal"):
            _write_nii(os.path.join(self.inp, cls, "a.nii.gz"))
            _write_nii(os.path.join(self.inp, cls, "b.nii.gz"))
        self.spec_path = os.path.join(self.tmp, "spec.json")
        with open(self.spec_path, "w") as f:
            json.dump({"task_type": "classification", "keys": ["image"],
                       "steps": [{"type": "load", "params": {}},
                                 {"type": "ensure_channel_first", "params": {}},
                                 {"type": "resize", "params": {"spatial_size": [16, 16, 8]}}]}, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_imagefolder_no_dataset_json(self):
        result = pp.main(["--spec-path", self.spec_path,
                          "--dataset-input", self.inp, "--output", self.out])
        self.assertEqual(result["task_type"], "classification")
        self.assertEqual(result["source"], "imagefolder")
        self.assertEqual(result["processed"], 4)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "dataset.json")))


class TestPreprocessMissingFiles(unittest.TestCase):
    """dataset.json references files that don't all exist (like the real
    Task06_Lung case) — the job should skip missing ones, not abort."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.inp = os.path.join(self.tmp, "in")
        self.out = os.path.join(self.tmp, "out")
        # Only case_1 actually exists on disk.
        _write_nii(os.path.join(self.inp, "imagesTr", "case_1.nii.gz"))
        _write_nii(os.path.join(self.inp, "labelsTr", "case_1.nii.gz"), value=1.0)
        dataset = {
            "name": "Partial", "labels": {"0": "bg", "1": "fg"},
            "training": [
                {"image": "./imagesTr/case_1.nii.gz", "label": "./labelsTr/case_1.nii.gz"},
                {"image": "./imagesTr/case_MISSING.nii.gz", "label": "./labelsTr/case_MISSING.nii.gz"},
            ],
        }
        with open(os.path.join(self.inp, "dataset.json"), "w") as f:
            json.dump(dataset, f)
        self.spec_path = os.path.join(self.tmp, "spec.json")
        with open(self.spec_path, "w") as f:
            json.dump(SPEC, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skips_missing_and_processes_rest(self):
        result = pp.main(["--spec-path", self.spec_path,
                          "--dataset-input", self.inp, "--output", self.out])
        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["failed"], 0)
        # Only the existing case ends up in the output dataset.json.
        with open(os.path.join(self.out, "dataset.json")) as f:
            od = json.load(f)
        self.assertEqual(len(od["training"]), 1)

    def test_all_missing_raises(self):
        # Point at a dataset where NO referenced file exists.
        bad_in = os.path.join(self.tmp, "empty_in")
        os.makedirs(bad_in)
        with open(os.path.join(bad_in, "dataset.json"), "w") as f:
            json.dump({"name": "None", "training": [
                {"image": "./imagesTr/nope.nii.gz", "label": "./labelsTr/nope.nii.gz"}]}, f)
        with self.assertRaises(SystemExit):
            pp.main(["--spec-path", self.spec_path,
                     "--dataset-input", bad_in, "--output", os.path.join(self.tmp, "empty_out")])


class TestSubmitBatchMocked(unittest.TestCase):
    def setUp(self):
        # Point the service's job store at a temp file so we don't touch real data.
        self.tmp = tempfile.mkdtemp()
        patcher = mock.patch("app.services.preprocessing_service.PREPROCESS_JOBS_FILE",
                             __import__("pathlib").Path(self.tmp) / "jobs.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.svc = PreprocessingService()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_submit_builds_job_and_persists(self):
        sm = mock.Mock()
        s3 = mock.Mock()
        rec = self.svc.submit_batch(
            input_s3_uri="s3://bucket/AMOS/",
            output_s3_uri="s3://bucket/processed/",
            spec=SPEC,
            image_uri="123.dkr.ecr.us-east-1.amazonaws.com/sm-preprocessing:latest",
            role_arn="arn:aws:iam::123:role/Sm",
            sagemaker_client=sm,
            s3_client=s3,
            instance_type="ml.m5.2xlarge",
            region="us-east-1",
        )
        # create_processing_job called with a sane config
        self.assertTrue(sm.create_processing_job.called)
        cfg = sm.create_processing_job.call_args.kwargs
        self.assertTrue(cfg["ProcessingJobName"].startswith("preprocess-"))
        self.assertEqual(cfg["ProcessingResources"]["ClusterConfig"]["InstanceType"], "ml.m5.2xlarge")
        input_names = {i["InputName"] for i in cfg["ProcessingInputs"]}
        self.assertEqual(input_names, {"dataset", "code"})
        self.assertEqual(cfg["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
                         "s3://bucket/processed/")
        # code + spec.json uploaded
        self.assertTrue(s3.upload_file.called)
        self.assertTrue(s3.put_object.called)
        put_key = s3.put_object.call_args.kwargs["Key"]
        self.assertTrue(put_key.endswith("/spec.json"))
        # record persisted with the spec
        self.assertEqual(rec["status"], "InProgress")
        self.assertEqual(rec["spec"], SPEC)
        self.assertIn(rec["job_name"], self.svc.jobs)

    def test_submit_rejects_missing_image(self):
        with self.assertRaises(ValueError):
            self.svc.submit_batch(
                input_s3_uri="s3://bkt/in/", output_s3_uri="s3://bkt/out/", spec=SPEC,
                image_uri="", role_arn="arn:aws:iam::1:role/x",
                sagemaker_client=mock.Mock(), s3_client=mock.Mock())

    def test_delete_keeps_s3(self):
        sm, s3 = mock.Mock(), mock.Mock()
        rec = self.svc.submit_batch(
            input_s3_uri="s3://bkt/in/", output_s3_uri="s3://bkt/out/", spec=SPEC,
            image_uri="img", role_arn="arn:role", sagemaker_client=sm, s3_client=s3)
        self.svc.delete_job(rec["job_name"])
        self.assertNotIn(rec["job_name"], self.svc.jobs)
        # deletion must not call any S3 delete
        self.assertFalse(s3.delete_object.called)


if __name__ == "__main__":
    unittest.main()
