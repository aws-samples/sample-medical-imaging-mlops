"""Unit tests for the preprocessing pipeline (models + builder + local preview).

Runs from the repo root so `app.models.*` and `src.preprocessing.*` import.
No AWS, no GPU — a synthetic NIfTI exercises the full local preview path.
"""
import os
import tempfile
import unittest

import numpy as np
import nibabel as nib
from pydantic import ValidationError

from app.models.preprocessing import PipelineSpec, Step, PreviewRequest
from src.preprocessing.pipeline_builder import build_transforms, get_catalog, STEP_CATALOG
from src.preprocessing import local_preview


def _default_steps(save=False):
    steps = [
        Step(type="load"),
        Step(type="ensure_channel_first"),
        Step(type="orientation", params={"axcodes": "RAS"}),
        Step(type="resample", params={"pixdim": [2.0, 2.0, 2.0]}),
        Step(type="intensity_window", params={"a_min": -100, "a_max": 200, "b_min": 0.0, "b_max": 1.0, "clip": True}),
        Step(type="resize", params={"spatial_size": [32, 32, 16]}),
    ]
    if save:
        steps.append(Step(type="save"))
    return steps


class TestModels(unittest.TestCase):
    def test_valid_segmentation_spec(self):
        spec = PipelineSpec(task_type="segmentation", keys=["image", "label"], steps=_default_steps())
        self.assertEqual(spec.steps[0].type, "load")

    def test_load_must_be_first(self):
        with self.assertRaises(ValidationError):
            PipelineSpec(keys=["image"], steps=[Step(type="orientation"), Step(type="load")])

    def test_save_must_be_last(self):
        with self.assertRaises(ValidationError):
            PipelineSpec(keys=["image"], steps=[Step(type="save"), Step(type="load")])

    def test_load_appears_once(self):
        with self.assertRaises(ValidationError):
            PipelineSpec(keys=["image"], steps=[Step(type="load"), Step(type="load")])

    def test_empty_pipeline_rejected(self):
        with self.assertRaises(ValidationError):
            PipelineSpec(keys=["image"], steps=[])

    def test_classification_rejects_label_key(self):
        with self.assertRaises(ValidationError):
            PipelineSpec(task_type="classification", keys=["image", "label"], steps=[Step(type="load")])

    def test_keys_must_include_image(self):
        with self.assertRaises(ValidationError):
            PipelineSpec(keys=["label"], steps=[Step(type="load")])

    def test_dataset_prep_split_must_sum_to_one(self):
        with self.assertRaises(ValidationError):
            PipelineSpec(keys=["image"], steps=[Step(type="load")],
                         dataset_prep={"enabled": True, "split": {"train": 0.5, "val": 0.2, "test": 0.2}})


class TestBuilder(unittest.TestCase):
    def test_catalog_covers_all_step_types(self):
        cat = get_catalog()
        # Every StepType in the models must be in the catalog.
        from app.models.preprocessing import StepType
        import typing
        for t in typing.get_args(StepType):
            self.assertIn(t, cat)
            self.assertIn("label", cat[t])
            self.assertIn("params", cat[t])

    def test_every_catalog_step_builds(self):
        """Each step in the catalog must build into a Compose without error."""
        from app.models.preprocessing import StepType
        import typing
        for t in typing.get_args(StepType):
            if t == "load":
                steps = [Step(type="load", params=dict(STEP_CATALOG["load"]["params"]))]
            else:
                steps = [Step(type="load"), Step(type="ensure_channel_first"),
                         Step(type=t, params=dict(STEP_CATALOG[t]["params"]))]
                if t == "save":
                    steps = [Step(type="load"), Step(type="ensure_channel_first"), Step(type="save")]
            spec = PipelineSpec(task_type="classification", keys=["image"], steps=steps)
            compose = build_transforms(spec, for_preview=(t != "save"))
            self.assertTrue(len(compose.transforms) >= 1, f"step {t} produced no transforms")

    def test_build_drops_save_for_preview(self):
        from monai.transforms import SaveImaged
        spec = PipelineSpec(keys=["image"], steps=_default_steps(save=True))
        compose = build_transforms(spec, for_preview=True)
        self.assertFalse(any(isinstance(t, SaveImaged) for t in compose.transforms))

    def test_build_keeps_save_for_batch(self):
        from monai.transforms import SaveImaged
        spec = PipelineSpec(keys=["image"], steps=_default_steps(save=True))
        compose = build_transforms(spec, for_preview=False, save_dir=tempfile.gettempdir())
        self.assertTrue(any(isinstance(t, SaveImaged) for t in compose.transforms))

    def test_label_gets_nearest_mode(self):
        from monai.transforms import Spacingd
        spec = PipelineSpec(task_type="segmentation", keys=["image", "label"],
                            steps=[Step(type="load"), Step(type="ensure_channel_first"),
                                   Step(type="resample", params={"pixdim": [2, 2, 2]})])
        compose = build_transforms(spec, for_preview=True)
        sp = [t for t in compose.transforms if isinstance(t, Spacingd)][0]
        # image -> bilinear, label -> nearest (positional to keys order)
        modes = [str(m) for m in sp.mode]
        self.assertIn("nearest", modes)


class TestLocalPreview(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Synthetic 3D volume with a bright cube in the middle.
        vol = np.zeros((40, 40, 20), dtype=np.float32)
        vol[10:30, 10:30, 5:15] = 300.0
        affine = np.diag([1.0, 1.0, 1.5, 1.0])
        self.nii_path = os.path.join(self.tmp, "case.nii.gz")
        nib.save(nib.Nifti1Image(vol, affine), self.nii_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_preview_end_to_end(self):
        spec = PipelineSpec(task_type="classification", keys=["image"], steps=_default_steps())
        res = local_preview.preview(spec, self.nii_path)

        self.assertIn("output", res)
        self.assertIn("preview_image_base64", res)
        # resize target was [32,32,16]; output spatial should match (channel-first)
        out_shape = res["output"]["shape"]
        self.assertEqual(out_shape[-3:], [32, 32, 16])
        # intensity_window clipped to [0,1]
        self.assertLessEqual(res["output"]["intensity_max"], 1.0 + 1e-6)
        self.assertGreaterEqual(res["output"]["intensity_min"], 0.0 - 1e-6)
        # a slice PNG was produced
        self.assertIsInstance(res["preview_image_base64"], str)
        self.assertGreater(len(res["preview_image_base64"]), 100)
        self.assertIn("RESEARCH USE ONLY", res["disclaimer"])

    def test_preview_rejects_traversal_path(self):
        spec = PipelineSpec(task_type="classification", keys=["image"], steps=[Step(type="load")])
        with self.assertRaises(local_preview.PreviewError):
            local_preview.preview(spec, "/etc/passwd")

    def test_preview_error_is_sanitized(self):
        # A nonexistent path under an allowed root -> sanitized "not found".
        spec = PipelineSpec(task_type="classification", keys=["image"], steps=[Step(type="load")])
        bad = os.path.join(self.tmp, "does_not_exist.nii.gz")
        with self.assertRaises(local_preview.PreviewError) as ctx:
            local_preview.preview(spec, bad)
        # No absolute path leaked in the message.
        self.assertNotIn(self.tmp, str(ctx.exception))

    def test_preview_request_model(self):
        req = PreviewRequest(input_ref=self.nii_path,
                             spec=PipelineSpec(keys=["image"], steps=[Step(type="load")]))
        self.assertEqual(req.input_ref, self.nii_path)


class TestValidateEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import main
        cls.client = TestClient(main.app)

    def test_catalog_endpoint(self):
        r = self.client.get("/api/preprocessing/catalog")
        self.assertEqual(r.status_code, 200)
        cat = r.json()["catalog"]
        # expanded catalog should have many more than the original 9
        self.assertGreaterEqual(len(cat), 20)
        for t in ["scale_intensity_percentiles", "spatial_pad", "gaussian_smooth", "zoom"]:
            self.assertIn(t, cat)

    def test_validate_accepts_good_spec(self):
        spec = {
            "task_type": "classification", "keys": ["image"],
            "steps": [{"type": "load", "params": {}},
                      {"type": "scale_intensity_percentiles", "params": {"lower": 2, "upper": 98}}],
        }
        r = self.client.post("/api/preprocessing/validate", json=spec)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()["valid"])
        self.assertEqual(len(r.json()["spec"]["steps"]), 2)

    def test_validate_rejects_unknown_step(self):
        spec = {"task_type": "classification", "keys": ["image"],
                "steps": [{"type": "not_a_real_transform", "params": {}}]}
        r = self.client.post("/api/preprocessing/validate", json=spec)
        self.assertEqual(r.status_code, 422)

    def test_validate_rejects_bad_ordering(self):
        spec = {"task_type": "classification", "keys": ["image"],
                "steps": [{"type": "resize", "params": {}}, {"type": "load", "params": {}}]}
        r = self.client.post("/api/preprocessing/validate", json=spec)
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
