"""Unit tests for the MONAI Bundle dashboard integration (service + router)."""
import json
import os
import tempfile
import unittest
from unittest import mock


class TestParseS3Uri(unittest.TestCase):
    def test_valid(self):
        from app.services.monai_bundle_service import parse_s3_uri
        b, k = parse_s3_uri("s3://bucket/a/b/model.tar.gz")
        self.assertEqual(b, "bucket")
        self.assertEqual(k, "a/b/model.tar.gz")

    def test_invalid(self):
        from app.services.monai_bundle_service import parse_s3_uri
        for bad in ("https://x/y", "s3://bucket-only", "s3://", "not-a-uri"):
            with self.assertRaises(ValueError):
                parse_s3_uri(bad)


class TestFinetuneLaunch(unittest.TestCase):
    def setUp(self):
        # Isolate the jobs file to a temp location.
        self._tmp = tempfile.mkdtemp()
        patcher = mock.patch(
            "app.services.monai_bundle_service.JOBS_FILE",
            __import__("pathlib").Path(self._tmp) / "jobs.json",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_start_finetune_builds_channels_and_tracks(self):
        from app.services.monai_bundle_service import MonaiBundleService
        svc = MonaiBundleService()

        sm_client = mock.Mock()
        info = svc.start_finetune(
            job_name="mb-test-1",
            model_artifacts="s3://b/bundle/model.tar.gz",
            data_s3_uri="s3://b/Task09_Spleen/",
            role_arn="arn:aws:iam::123:role/x",
            image_uri="123.dkr.ecr.us-east-1.amazonaws.com/sm-monai-bundle:latest",
            sagemaker_client=sm_client,
            instance_type="ml.g5.4xlarge",
            train_config="train.json",
            config_overrides={"epochs": 3, "cache_rate": 0.5},
            output_s3_uri="s3://b/out/",
        )

        # create_training_job called with the validated contract
        sm_client.create_training_job.assert_called_once()
        req = sm_client.create_training_job.call_args.kwargs
        algo = req["AlgorithmSpecification"]
        self.assertEqual(algo["ContainerEntrypoint"], ["python", "/opt/program/train.py"])
        self.assertEqual(algo["TrainingImage"], "123.dkr.ecr.us-east-1.amazonaws.com/sm-monai-bundle:latest")
        # overrides serialized into container args
        args = algo["ContainerArguments"]
        self.assertIn("--config_overrides", args)
        ov = json.loads(args[args.index("--config_overrides") + 1])
        self.assertEqual(ov, {"epochs": 3, "cache_rate": 0.5})
        self.assertIn("--train_config", args)
        # both channels present
        channels = {c["ChannelName"] for c in req["InputDataConfig"]}
        self.assertEqual(channels, {"model", "data"})
        # tracked
        self.assertEqual(info["job_name"], "mb-test-1")
        self.assertEqual(info["status"], "InProgress")

    def test_finetune_rejects_bad_s3(self):
        from app.services.monai_bundle_service import MonaiBundleService
        svc = MonaiBundleService()
        with self.assertRaises(ValueError):
            svc.start_finetune(
                job_name=None, model_artifacts="not-s3", data_s3_uri="s3://b/d/",
                role_arn="arn:x", image_uri="123.dkr.ecr.us-east-1.amazonaws.com/sm-monai-bundle:latest",
                sagemaker_client=mock.Mock(),
            )


class TestRouterWiring(unittest.TestCase):
    def test_routes_registered(self):
        import main
        paths = [r.path for r in main.app.routes]
        for p in ("/api/monai-bundle/stage", "/api/monai-bundle/finetune",
                  "/api/monai-bundle/jobs", "/api/monai-bundle/jobs/{job_name}",
                  "/api/monai-bundle/supported", "/api/monai-bundle/deploy",
                  "/api/monai-bundle/artifacts"):
            self.assertIn(p, paths)


class TestSupportedRegistry(unittest.TestCase):
    def test_registry_shape(self):
        from app.services.monai_bundle_service import list_supported_bundles
        bundles = list_supported_bundles()
        self.assertTrue(len(bundles) >= 1)
        names = {b["name"] for b in bundles}
        self.assertIn("spleen_ct_segmentation", names)
        for b in bundles:
            # fields the training grid renders
            for key in ("name", "repo_id", "task", "description", "parameters", "input_size"):
                self.assertIn(key, b)
            self.assertEqual(b["source_type"], "monai_bundle")


if __name__ == "__main__":
    unittest.main()
