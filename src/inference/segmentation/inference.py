"""
Inference script for 3D medical image segmentation on SageMaker endpoints.
Input:  NIfTI volume via S3 URI, local path, or base64.
Output: Binary segmentation mask as base64-encoded NIfTI bytes.
"""
import json
import logging
import os
import base64
import tempfile
from typing import Any, Dict, Union

import torch
import numpy as np
import nibabel as nib
import boto3
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Resized, ScaleIntensityRanged,
    Activations, AsDiscrete,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SPATIAL_SIZE = (128, 128, 64)


def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load segmentation model from model_dir."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Files in model_dir: %s", os.listdir(model_dir))

    # Dynamically load ModelDef
    import importlib.util
    model_def_path = os.path.join(model_dir, "model_def.py")
    if os.path.exists(model_def_path):
        spec   = importlib.util.spec_from_file_location("model_def", model_def_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_name = os.environ.get("MODEL_NAME", "segresnet")
        model = module.ModelDef(num_classes=1, model_name=model_name).get_model()
    else:
        from monai.networks.nets import SegResNet
        model = SegResNet(spatial_dims=3, in_channels=1, out_channels=1,
                          init_filters=16, dropout_prob=0.2)

    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        logger.info("Loaded weights from %s", model_path)

    model.to(device).eval()

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE),
        ScaleIntensityRanged(keys="image", a_min=-100, a_max=500,
                             b_min=0, b_max=1, clip=True),
    ])

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    return {"model": model, "device": device,
            "transforms": transforms, "post_trans": post_trans}


def _s3_to_tmp(s3_uri: str, suffix: str = ".nii.gz") -> str:
    parts  = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.flush()
        tmp_path = f.name
    s3.download_file(bucket, key, tmp_path)
    return tmp_path


def input_fn(request_body: Union[str, bytes], content_type: str) -> str:
    """Returns a local file path to a NIfTI volume."""
    if content_type == "application/octet-stream":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as f:
            f.write(request_body)
            f.flush()
            return f.name

    if content_type == "application/json":
        data = json.loads(request_body)
        uri  = data.get("s3_uri") or data.get("file_path", "")

        if uri.startswith("s3://"):
            ext = ".nii.gz" if uri.endswith(".gz") else ".nii"
            return _s3_to_tmp(uri, ext)

        if os.path.exists(uri):
            return uri

        # Try base64
        try:
            raw = base64.b64decode(uri)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as f:
                f.write(raw)
                f.flush()
                return f.name
        except Exception:
            raise ValueError(f"Cannot resolve input: {uri}")

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(file_path: str, model_components: Dict[str, Any]) -> np.ndarray:
    """Run segmentation and return binary mask as numpy array."""
    model      = model_components["model"]
    device     = model_components["device"]
    transforms = model_components["transforms"]
    post_trans = model_components["post_trans"]

    batch = transforms({"image": file_path})
    tensor = batch["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        mask   = post_trans(output)

    return mask.squeeze().cpu().numpy().astype(np.uint8)


def output_fn(mask: np.ndarray, accept: str) -> str:
    """Return mask as base64-encoded NIfTI bytes or JSON with stats."""
    nii  = nib.Nifti1Image(mask, affine=np.eye(4))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as f:
        f.flush()
        tmp_path = f.name
    nib.save(nii, tmp_path)
    with open(tmp_path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("utf-8")

    result = {
        "mask_nifti_b64": encoded,
        "shape":          list(mask.shape),
        "voxels_positive": int(mask.sum()),
        "voxels_total":    int(mask.size),
        "volume_fraction": round(float(mask.sum()) / mask.size, 4),
    }
    return json.dumps(result)
