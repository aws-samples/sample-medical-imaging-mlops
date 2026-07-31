"""
Inference script for medical image segmentation on SageMaker.

Supports SegResNet and SwinUNETR models trained with MONAI.
Accepts NIfTI files via S3 URI, local path, or base64-encoded payload.
Returns segmentation mask as a base64-encoded NIfTI file.
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
from monai.networks.nets import SegResNet, SwinUNETR
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    Resized,
    ScaleIntensityRanged,
    EnsureTyped,
    Activations,
    AsDiscrete,
)
from monai.inferers import SlidingWindowInferer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load the trained segmentation model and set up inference pipeline."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Read model config if available
    config_path = os.path.join(model_dir, "config.json")
    model_name = "SegResNet"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            model_name = config.get("model_name", "SegResNet")

    logger.info(f"Loading model: {model_name}")

    # Initialize model
    if model_name.lower() == "swinunetr":
        model = SwinUNETR(
            img_size=(128, 128, 64),
            in_channels=1,
            out_channels=1,
            feature_size=24,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
        )
    else:
        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            init_filters=16,
            dropout_prob=0.2,
        )

    # Load weights - try multiple common filenames
    weight_files = ["best_model.pth", "final_model.pth", "model.pth", "model.pt"]
    loaded = False
    for weight_file in weight_files:
        weight_path = os.path.join(model_dir, weight_file)
        if os.path.exists(weight_path):
            # weights_only=False required here because training checkpoints contain
            # dicts with "model", "epoch", "optimizer" keys — not just raw tensors.
            # Trust boundary: model artifacts come from our own training pipeline via S3.
            checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded weights from: {weight_file}")
            loaded = True
            break

    if not loaded:
        logger.warning("No model weights found, using random initialization")

    model = model.to(device)
    model.eval()

    # Preprocessing transforms (must match training transforms)
    preprocessing = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=[1.5, 1.5, 2.0], mode="bilinear"),
        ScaleIntensityRanged(
            keys="image", a_min=-100, a_max=500, b_min=0, b_max=1, clip=True
        ),
        Resized(keys="image", spatial_size=(128, 128, 64)),
        EnsureTyped(keys="image"),
    ])

    # Post-processing
    post_transforms = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5),
    ])

    # Sliding window inferer for large volumes
    inferer = SlidingWindowInferer(
        roi_size=(128, 128, 64),
        sw_batch_size=4,
        overlap=0.5,
    )

    return {
        "model": model,
        "device": device,
        "preprocessing": preprocessing,
        "post_transforms": post_transforms,
        "inferer": inferer,
        "model_name": model_name,
    }


def input_fn(request_body: Union[str, bytes], content_type: str) -> Dict[str, str]:
    """Parse input data and return a dict with local file path and metadata.

    Supports:
      - application/json with {"file_path": "<s3_uri_or_local_path>"}
      - application/json with {"s3_uri": "<s3_uri>"} (compatibility)
      - application/x-nifti or application/octet-stream with raw NIfTI binary

    Returns:
        Dict with "local_path" and optionally "s3_uri" for output path derivation.
    """
    if content_type == "application/json":
        input_data = json.loads(request_body)

        # Support both "file_path" and "s3_uri" keys
        file_path = input_data.get("file_path") or input_data.get("s3_uri")
        if not file_path:
            raise ValueError("JSON payload must contain 'file_path' or 's3_uri' key")

        if file_path.startswith("s3://"):
            import boto3
            parts = file_path.replace("s3://", "").split("/", 1)
            bucket_name, key = parts[0], parts[1]
            s3_client = boto3.client("s3")
            suffix = ".nii.gz" if key.endswith(".nii.gz") else ".nii"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = temp_file.name
            s3_client.download_file(bucket_name, key, temp_path)
            logger.info(f"Downloaded S3 file to: {temp_path}")
            return {"local_path": temp_path, "s3_uri": file_path}

        elif file_path.endswith((".nii.gz", ".nii")):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            return {"local_path": file_path}

        else:
            # Assume base64-encoded NIfTI
            try:
                image_data = base64.b64decode(file_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
                    temp_file.write(image_data)
                    temp_path = temp_file.name
                logger.info(f"Decoded base64 to: {temp_path}")
                return {"local_path": temp_path}
            except Exception as e:
                raise ValueError(
                    f"Invalid file_path: expected S3 URI, local path, or base64. Error: {e}"
                )

    elif content_type in ("application/x-nifti", "application/octet-stream"):
        if isinstance(request_body, str):
            request_body = request_body.encode()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            temp_file.write(request_body)
            temp_path = temp_file.name
        logger.info(f"Saved binary NIfTI to: {temp_path}")
        return {"local_path": temp_path}

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: Dict[str, str], model_components: Dict[str, Any]) -> Dict[str, Any]:
    """Run segmentation inference on a NIfTI image.
    Saves the segmentation mask as .nii.gz to S3 (alongside the input).
    Returns statistics in the JSON response."""
    model = model_components["model"]
    device = model_components["device"]
    preprocessing = model_components["preprocessing"]
    post_transforms = model_components["post_transforms"]
    inferer = model_components["inferer"]

    image_path = input_data["local_path"]
    source_s3_uri = input_data.get("s3_uri")
    logger.info(f"Processing: {image_path}")

    try:
        # Load original image for metadata
        original_img = nib.load(image_path)
        original_affine = original_img.affine
        original_shape = original_img.shape

        # Preprocess
        data_dict = {"image": image_path}
        preprocessed = preprocessing(data_dict)
        input_tensor = preprocessed["image"].unsqueeze(0).to(device)
        logger.info(f"Input tensor shape: {input_tensor.shape}")

        # Inference
        with torch.no_grad():
            prediction = inferer(input_tensor, model)

        # Post-process
        prediction = post_transforms(prediction[0])  # remove batch dim
        seg_array = prediction.cpu().numpy().squeeze()
        logger.info(f"Segmentation shape: {seg_array.shape}")

        # Compute statistics
        voxel_count = int(np.sum(seg_array > 0))
        total_voxels = int(seg_array.size)
        seg_percentage = (voxel_count / total_voxels) * 100

        # Save segmentation as NIfTI
        seg_nifti = nib.Nifti1Image(
            seg_array.astype(np.float32), affine=original_affine
        )

        # Upload NIfTI mask to S3 (next to input file)
        mask_s3_uri = None
        if source_s3_uri:
            try:
                import boto3
                s3_client = boto3.client("s3")
                parts = source_s3_uri.replace("s3://", "").split("/", 1)
                bucket_name, key = parts[0], parts[1]
                # Place mask alongside input: img.nii.gz → img_seg.nii.gz
                if key.endswith(".nii.gz"):
                    mask_key = key.replace(".nii.gz", "_seg.nii.gz")
                elif key.endswith(".nii"):
                    mask_key = key.replace(".nii", "_seg.nii.gz")
                else:
                    mask_key = key + "_seg.nii.gz"

                with tempfile.NamedTemporaryFile(delete=False, suffix="_seg.nii.gz") as temp_out:
                    nib.save(seg_nifti, temp_out.name)
                    s3_client.upload_file(temp_out.name, bucket_name, mask_key)
                    os.unlink(temp_out.name)

                mask_s3_uri = f"s3://{bucket_name}/{mask_key}"
                logger.info(f"Saved segmentation mask to: {mask_s3_uri}")
            except Exception as e:
                logger.warning(f"Failed to upload mask to S3: {e}")

        # Cleanup temp input
        if "/tmp/" in image_path:
            try:
                os.unlink(image_path)
            except OSError:
                pass

        result = {
            "model_name": model_components["model_name"],
            "original_shape": list(original_shape),
            "segmentation_shape": list(seg_array.shape),
            "segmented_voxels": voxel_count,
            "total_voxels": total_voxels,
            "segmentation_percentage": round(seg_percentage, 2),
        }
        if mask_s3_uri:
            result["mask_s3_uri"] = mask_s3_uri

        return result

    except Exception as e:
        logger.error(f"Inference error: {e}")
        if "/tmp/" in image_path:
            try:
                os.unlink(image_path)
            except OSError:
                pass
        return {"error": str(e), "model_name": model_components.get("model_name", "unknown")}


def output_fn(prediction: Dict[str, Any], accept: str) -> str:
    """Format the prediction output as JSON."""
    return json.dumps(prediction)
