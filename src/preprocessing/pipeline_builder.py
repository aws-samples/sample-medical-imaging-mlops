"""Shared MONAI preprocessing pipeline builder.

The single place that maps a PipelineSpec's steps to a monai.transforms.Compose.
Used by the local preview runner (app server) and, later, the SageMaker batch
processing script — so "what you preview is what the batch job runs".

Dependency-light: only imports MONAI lazily inside build_transforms so that
get_catalog() (used to populate the UI) works without importing torch/monai.

RESEARCH USE ONLY.
"""
from __future__ import annotations

from typing import Any, Dict, List


# ── Catalog ────────────────────────────────────────────────────────────────────
# type -> { label, params (defaults), doc }.  Kept in sync with StepType in
# app/models/preprocessing.py and the JS fallback catalog.
STEP_CATALOG: Dict[str, Dict[str, Any]] = {
    "load": {
        "label": "Load image (DICOM/NIfTI)",
        "params": {"reader": "auto"},
        "doc": "LoadImaged — loads a NIfTI file or stacks a DICOM series directory.",
    },
    "ensure_channel_first": {
        "label": "Ensure channel first",
        "params": {},
        "doc": "EnsureChannelFirstd — moves/creates the channel dim to front.",
    },
    "orientation": {
        "label": "Orientation (RAS)",
        "params": {"axcodes": "RAS"},
        "doc": "Orientationd — reorient to a target anatomical convention.",
    },
    "resample": {
        "label": "Resample spacing",
        "params": {"pixdim": [1.5, 1.5, 2.0], "mode": ["bilinear", "nearest"]},
        "doc": "Spacingd — resample to a target voxel spacing.",
    },
    "intensity_window": {
        "label": "Intensity window (CT HU)",
        "params": {"a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1.0, "clip": True},
        "doc": "ScaleIntensityRanged — window/clip HU (CT) and rescale.",
    },
    "scale_intensity": {
        "label": "Scale intensity (min–max)",
        "params": {"minv": 0.0, "maxv": 1.0},
        "doc": "ScaleIntensityd — linearly rescale to [minv, maxv].",
    },
    "scale_intensity_percentiles": {
        "label": "Scale intensity (percentiles)",
        "params": {"lower": 1.0, "upper": 99.0, "b_min": 0.0, "b_max": 1.0, "clip": True},
        "doc": "ScaleIntensityRangePercentilesd — robust rescale using intensity percentiles (good for MR).",
    },
    "clip_percentiles": {
        "label": "Clip intensity (percentiles)",
        "params": {"lower": 1.0, "upper": 99.0},
        "doc": "ClipIntensityPercentilesd — clip outliers at the given percentiles.",
    },
    "threshold_intensity": {
        "label": "Threshold intensity",
        "params": {"threshold": 0.0, "above": True, "cval": 0.0},
        "doc": "ThresholdIntensityd — zero out values above/below a threshold.",
    },
    "histogram_normalize": {
        "label": "Histogram normalize",
        "params": {"num_bins": 256, "min": 0.0, "max": 1.0},
        "doc": "HistogramNormalized — equalize the intensity histogram.",
    },
    "normalize": {
        "label": "Normalize intensity",
        "params": {},
        "doc": "NormalizeIntensityd — zero-mean unit-variance (nonzero by default).",
    },
    "gaussian_smooth": {
        "label": "Gaussian smooth",
        "params": {"sigma": 1.0},
        "doc": "GaussianSmoothd — Gaussian blur (image only).",
    },
    "median_smooth": {
        "label": "Median smooth",
        "params": {"radius": 1},
        "doc": "MedianSmoothd — median filter denoising (image only).",
    },
    "crop_foreground": {
        "label": "Crop to foreground",
        "params": {"source_key": "image"},
        "doc": "CropForegroundd — crop background borders using a source key.",
    },
    "center_crop": {
        "label": "Center spatial crop",
        "params": {"roi_size": [128, 128, 128]},
        "doc": "CenterSpatialCropd — crop a centered region of interest.",
    },
    "spatial_pad": {
        "label": "Spatial pad",
        "params": {"spatial_size": [128, 128, 128], "mode": "constant"},
        "doc": "SpatialPadd — pad up to a target spatial size.",
    },
    "divisible_pad": {
        "label": "Divisible pad",
        "params": {"k": 16, "mode": "constant"},
        "doc": "DivisiblePadd — pad so each dim is divisible by k (for encoders).",
    },
    "resize": {
        "label": "Resize",
        "params": {"spatial_size": [128, 128, 128], "mode": ["area", "nearest"]},
        "doc": "Resized — resize to a target spatial size.",
    },
    "zoom": {
        "label": "Zoom",
        "params": {"zoom": 1.0, "mode": ["area", "nearest"]},
        "doc": "Zoomd — scale by a zoom factor (keeps content, changes size).",
    },
    "flip": {
        "label": "Flip",
        "params": {"spatial_axis": 0},
        "doc": "Flipd — deterministic flip along a spatial axis.",
    },
    "rotate90": {
        "label": "Rotate 90°",
        "params": {"k": 1, "spatial_axes": [0, 1]},
        "doc": "Rotate90d — rotate 90°·k in the given plane.",
    },
    "cast": {
        "label": "Cast dtype",
        "params": {"dtype": "float32"},
        "doc": "CastToTyped — cast the array to a numpy dtype.",
    },
    "ensure_type": {
        "label": "Ensure type (tensor)",
        "params": {"data_type": "tensor"},
        "doc": "EnsureTyped — coerce to tensor/numpy of a consistent type.",
    },
    "save": {
        "label": "Save (.nii.gz)",
        "params": {"output_postfix": "proc", "output_ext": ".nii.gz"},
        "doc": "SaveImaged — write processed volume to disk (batch only).",
    },
}


def get_catalog() -> Dict[str, Dict[str, Any]]:
    """Return the catalog for the UI (labels + default params + docs)."""
    return {k: {"label": v["label"], "params": v["params"], "doc": v["doc"]}
            for k, v in STEP_CATALOG.items()}


def _split_modes(keys: List[str], mode, image_default, label_default):
    """Resolve a per-key interpolation mode list.

    Accepts a scalar (applied to all keys) or a 2-list [image_mode, label_mode].
    Ensures labels get nearest-neighbour by default.
    """
    if mode is None:
        return [image_default if k == "image" else label_default for k in keys]
    if isinstance(mode, (list, tuple)):
        # If a 2-list [image, label] is given, map positionally by key role.
        if len(mode) == len(keys):
            return list(mode)
        img_m = mode[0]
        lbl_m = mode[1] if len(mode) > 1 else label_default
        return [img_m if k == "image" else lbl_m for k in keys]
    # scalar
    return [mode for _ in keys]


def build_transforms(spec, *, for_preview: bool = False, save_dir: str | None = None):
    """Build a monai.transforms.Compose from a PipelineSpec.

    Args:
        spec: a PipelineSpec (or an object with .task_type, .keys, .steps).
        for_preview: if True, drop the 'save' step (preview never writes).
        save_dir: output dir for SaveImaged when a 'save' step is present.

    Returns:
        monai.transforms.Compose
    """
    from monai import transforms as mt  # lazy import

    keys = list(spec.keys)
    label_keys = [k for k in keys if k != "image"]
    xforms = []

    for step in spec.steps:
        t = step.type
        p = dict(step.params or {})

        if t == "load":
            reader = p.get("reader", "auto")
            kwargs = {"keys": keys, "image_only": True}
            if reader and reader != "auto":
                kwargs["reader"] = reader
            xforms.append(mt.LoadImaged(**kwargs))

        elif t == "ensure_channel_first":
            xforms.append(mt.EnsureChannelFirstd(keys=keys))

        elif t == "orientation":
            xforms.append(mt.Orientationd(keys=keys, axcodes=p.get("axcodes", "RAS")))

        elif t == "resample":
            modes = _split_modes(keys, p.get("mode"), "bilinear", "nearest")
            xforms.append(mt.Spacingd(keys=keys, pixdim=p.get("pixdim", [1.5, 1.5, 2.0]), mode=modes))

        elif t == "intensity_window":
            # Applied to the image key(s) only; labels are untouched.
            img_keys = [k for k in keys if k == "image"] or ["image"]
            xforms.append(mt.ScaleIntensityRanged(
                keys=img_keys,
                a_min=p.get("a_min", -175), a_max=p.get("a_max", 250),
                b_min=p.get("b_min", 0.0), b_max=p.get("b_max", 1.0),
                clip=bool(p.get("clip", True)),
            ))

        elif t == "normalize":
            img_keys = [k for k in keys if k == "image"] or ["image"]
            xforms.append(mt.NormalizeIntensityd(keys=img_keys, nonzero=bool(p.get("nonzero", True))))

        elif t == "scale_intensity":
            img_keys = [k for k in keys if k == "image"] or ["image"]
            xforms.append(mt.ScaleIntensityd(keys=img_keys, minv=p.get("minv", 0.0), maxv=p.get("maxv", 1.0)))

        elif t == "scale_intensity_percentiles":
            img_keys = [k for k in keys if k == "image"] or ["image"]
            xforms.append(mt.ScaleIntensityRangePercentilesd(
                keys=img_keys, lower=p.get("lower", 1.0), upper=p.get("upper", 99.0),
                b_min=p.get("b_min", 0.0), b_max=p.get("b_max", 1.0), clip=bool(p.get("clip", True)),
            ))

        elif t == "clip_percentiles":
            img_keys = [k for k in keys if k == "image"] or ["image"]
            xforms.append(mt.ClipIntensityPercentilesd(
                keys=img_keys, lower=p.get("lower", 1.0), upper=p.get("upper", 99.0)))

        elif t == "threshold_intensity":
            img_keys = [k for k in keys if k == "image"] or ["image"]
            xforms.append(mt.ThresholdIntensityd(
                keys=img_keys, threshold=p.get("threshold", 0.0),
                above=bool(p.get("above", True)), cval=p.get("cval", 0.0)))

        elif t == "histogram_normalize":
            img_keys = [k for k in keys if k == "image"] or ["image"]
            xforms.append(mt.HistogramNormalized(
                keys=img_keys, num_bins=int(p.get("num_bins", 256)),
                min=p.get("min", 0.0), max=p.get("max", 1.0)))

        elif t == "gaussian_smooth":
            img_keys = [k for k in keys if k == "image"] or ["image"]
            xforms.append(mt.GaussianSmoothd(keys=img_keys, sigma=p.get("sigma", 1.0)))

        elif t == "median_smooth":
            img_keys = [k for k in keys if k == "image"] or ["image"]
            xforms.append(mt.MedianSmoothd(keys=img_keys, radius=int(p.get("radius", 1))))

        elif t == "crop_foreground":
            xforms.append(mt.CropForegroundd(keys=keys, source_key=p.get("source_key", "image")))

        elif t == "center_crop":
            xforms.append(mt.CenterSpatialCropd(keys=keys, roi_size=p.get("roi_size", [128, 128, 128])))

        elif t == "spatial_pad":
            xforms.append(mt.SpatialPadd(keys=keys, spatial_size=p.get("spatial_size", [128, 128, 128]),
                                         mode=p.get("mode", "constant")))

        elif t == "divisible_pad":
            xforms.append(mt.DivisiblePadd(keys=keys, k=int(p.get("k", 16)), mode=p.get("mode", "constant")))

        elif t == "resize":
            modes = _split_modes(keys, p.get("mode"), "area", "nearest")
            xforms.append(mt.Resized(keys=keys, spatial_size=p.get("spatial_size", [128, 128, 128]), mode=modes))

        elif t == "zoom":
            modes = _split_modes(keys, p.get("mode"), "area", "nearest")
            xforms.append(mt.Zoomd(keys=keys, zoom=p.get("zoom", 1.0), mode=modes))

        elif t == "flip":
            xforms.append(mt.Flipd(keys=keys, spatial_axis=p.get("spatial_axis", 0)))

        elif t == "rotate90":
            xforms.append(mt.Rotate90d(keys=keys, k=int(p.get("k", 1)),
                                       spatial_axes=tuple(p.get("spatial_axes", [0, 1]))))

        elif t == "cast":
            xforms.append(mt.CastToTyped(keys=keys, dtype=p.get("dtype", "float32")))

        elif t == "ensure_type":
            xforms.append(mt.EnsureTyped(keys=keys, data_type=p.get("data_type", "tensor")))

        elif t == "save":
            if for_preview:
                continue  # preview never writes to disk
            xforms.append(mt.SaveImaged(
                keys=[k for k in keys if k == "image"] or ["image"],
                output_dir=save_dir or ".",
                output_postfix=p.get("output_postfix", "proc"),
                output_ext=p.get("output_ext", ".nii.gz"),
                resample=False, separate_folder=False, print_log=False,
            ))

        else:  # pragma: no cover - guarded by Pydantic StepType
            raise ValueError(f"Unknown step type: {t!r}")

    return mt.Compose(xforms)
