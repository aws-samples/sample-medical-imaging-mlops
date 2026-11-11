"""
Inference script for medical image segmentation on SageMaker.

⚠️ RESEARCH USE ONLY — This inference handler produces AI predictions for
research and development purposes. Outputs must NOT be used as the sole basis
for clinical decisions. Clinical deployment requires:
  - Regulatory clearance (FDA 510(k)/De Novo, CE marking, or equivalent)
  - Clinical validation studies
  - Qualified healthcare professional review of all predictions
  - Fairness and bias evaluation across demographic subgroups

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
from monai.networks.nets import SegResNet, SwinUNETR, UNet
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
    Activationsd,
    AsDiscreted,
    Invertd,
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

    # Read model config written by train.py (result.json). Fall back to a legacy
    # config.json, then to binary SegResNet defaults, for older artifacts.
    model_name = "SegResNet"
    num_classes = 1
    meta = {}
    for cfg_name in ("result.json", "config.json"):
        cfg_path = os.path.join(model_dir, cfg_name)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                meta = json.load(f)
            model_name = meta.get("model_name", model_name)
            num_classes = int(meta.get("num_classes", 1) or 1)
            break

    multiclass = num_classes > 1
    out_channels = num_classes if multiclass else 1
    roi_size = tuple(int(x) for x in (meta.get("roi_size") or [96, 96, 96]))
    logger.info("Loading model: %s  num_classes=%d  mode=%s  roi_size=%s",
                model_name, num_classes, "multiclass" if multiclass else "binary", roi_size)

    # Initialize model — architecture + out_channels MUST match training.
    name = model_name.lower()
    if name == "swinunetr":
        model = SwinUNETR(
            img_size=(128, 128, 64),
            in_channels=1,
            out_channels=out_channels,
            feature_size=24,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
        )
    elif name == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    else:
        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_channels,
            init_filters=16,
            dropout_prob=0.2,
        )

    # Load weights - try multiple common filenames
    weight_files = ["best_model.pth", "final_model.pth", "model.pth", "model.pt"]
    loaded = False
    for weight_file in weight_files:
        weight_path = os.path.join(model_dir, weight_file)
        if os.path.exists(weight_path):
            # Use weights_only=True for security — prevents arbitrary code execution
            # during deserialization. Checkpoints from our training pipeline contain
            # only tensors and basic Python types (dict, list, int, str).
            try:
                checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
            except Exception as e:
                logger.warning(f"weights_only=True failed for {weight_file}: {e}. "
                               "Verify checkpoint was produced by a trusted training pipeline.")
                raise
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

    # Preprocessing — MUST mirror training (src/segmentation/utils.py).
    # Training applies only: channel-first, CT HU window -> [0,1], ensure-type —
    # and crops foreground patches at NATIVE resolution (no orientation/spacing/
    # resize). Inference therefore also keeps native geometry and lets the
    # SlidingWindowInferer tile the ROI over the full volume. The mask is already
    # in the input's geometry, so no Invertd is needed.
    preprocessing = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(
            keys="image", a_min=-100, a_max=500, b_min=0, b_max=1, clip=True
        ),
        EnsureTyped(keys="image"),
    ])

    # Post-processing.
    #
    # Preprocessing applies NO spatial transforms (no orientation/spacing/resize)
    # — patches were cropped at native resolution during training and the sliding
    # window infers over the native volume — so the prediction is ALREADY in the
    # input image's geometry. No Invertd needed; just activate + discretize.
    #
    # Binary: sigmoid -> threshold 0.5 -> {0,1} mask.
    # Multi-class: softmax -> argmax -> integer label map 0..N-1.
    if multiclass:
        post_transforms = Compose([
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
        ])
    else:
        post_transforms = Compose([
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ])

    # Sliding window inferer — roi_size MUST match the training patch size
    # (from result.json). Tiles the ROI over the full native volume.
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
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
        "num_classes": num_classes,
        "multiclass": multiclass,
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

        output_s3_uri = input_data.get("output_s3_uri", os.environ.get("OUTPUT_S3_URI", ""))
        output_pattern = input_data.get("output_pattern", "")

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
            return {"local_path": temp_path, "s3_uri": file_path, "output_s3_uri": output_s3_uri, "output_pattern": output_pattern}

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
    output_s3_uri = input_data.get("output_s3_uri", "")
    output_pattern_override = input_data.get("output_pattern", "")
    logger.info(f"Processing: {image_path}")

    try:
        # Load original image for metadata
        original_img = nib.load(image_path)
        original_affine = original_img.affine
        original_shape = original_img.shape

        # Preprocess (dictionary transforms track metadata needed for inversion)
        data_dict = {"image": image_path}
        preprocessed = preprocessing(data_dict)
        input_tensor = preprocessed["image"].unsqueeze(0).to(device)
        logger.info(f"Input tensor shape: {input_tensor.shape}")

        # Inference
        with torch.no_grad():
            prediction = inferer(input_tensor, model)

        # Post-process: activate + discretize. No geometry inversion needed —
        # preprocessing applied no spatial transforms, so the prediction is
        # already in the native input geometry (shape + affine match the source).
        preprocessed["pred"] = prediction[0]  # drop batch dim -> (C, H, W, D)
        processed = post_transforms(preprocessed)
        pred = processed["pred"]
        seg_array = pred.detach().cpu().numpy().squeeze()
        logger.info(f"Segmentation shape (native): {seg_array.shape} "
                    f"vs original {original_shape}")

        # Affine from the prediction MetaTensor (carries the input image's
        # affine, since no resampling occurred). Fall back to the original.
        try:
            out_affine = np.asarray(pred.affine.detach().cpu().numpy())
        except Exception:
            out_affine = original_affine

        # Compute statistics
        voxel_count = int(np.sum(seg_array > 0))
        total_voxels = int(seg_array.size)
        seg_percentage = (voxel_count / total_voxels) * 100

        # Per-class voxel counts (multi-class: one entry per non-background label).
        per_class_voxels = {}
        if model_components.get("multiclass"):
            import numpy as _np
            vals, counts = _np.unique(seg_array.astype(_np.int64), return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                if v == 0:
                    continue
                per_class_voxels[str(int(v))] = int(c)

        # Save segmentation as NIfTI in the ORIGINAL image geometry.
        seg_nifti = nib.Nifti1Image(
            seg_array.astype(np.float32), affine=out_affine
        )

        # Upload NIfTI mask to S3 at output_s3_uri using the output filename pattern
        mask_s3_uri = None
        if source_s3_uri:
            try:
                import boto3
                from datetime import datetime
                s3_client = boto3.client("s3")

                # Extract metadata from input path for pattern substitution
                # e.g., s3://bucket/segmentation/test/10090344/img.nii.gz
                source_parts = source_s3_uri.replace("s3://", "").split("/")
                patient_id = source_parts[-2] if len(source_parts) >= 3 else "unknown"
                input_filename = source_parts[-1].replace(".nii.gz", "").replace(".nii", "")
                timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

                # Output filename pattern — supports placeholders:
                #   {patient_id}    - parent folder of input (e.g., "10090344")
                #   {filename}      - input filename without extension (e.g., "img")
                #   {endpoint_name} - name of the endpoint (e.g., "segmentation")
                #   {timestamp}     - UTC timestamp (e.g., "20260731-163000")
                #   {model_name}    - model used (e.g., "SegResNet")
                output_pattern = output_pattern_override or os.environ.get(
                    "OUTPUT_FILENAME_PATTERN", "{patient_id}_segmentation.nii.gz"
                )
                endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "endpoint")
                output_filename = output_pattern.format(
                    patient_id=patient_id,
                    filename=input_filename,
                    endpoint_name=endpoint_name,
                    timestamp=timestamp,
                    model_name=model_components["model_name"],
                )

                # Determine output bucket and prefix
                if output_s3_uri:
                    out_parts = output_s3_uri.replace("s3://", "").rstrip("/").split("/", 1)
                    out_bucket = out_parts[0]
                    out_prefix = out_parts[1] if len(out_parts) > 1 else "segmentation-results"
                else:
                    out_bucket = source_parts[0]
                    out_prefix = "segmentation-results"

                mask_key = f"{out_prefix}/{output_filename}"

                with tempfile.NamedTemporaryFile(delete=False, suffix="_seg.nii.gz") as temp_out:
                    nib.save(seg_nifti, temp_out.name)
                    s3_client.upload_file(temp_out.name, out_bucket, mask_key)
                    os.unlink(temp_out.name)

                mask_s3_uri = f"s3://{out_bucket}/{mask_key}"
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
            "num_classes": model_components.get("num_classes", 1),
            "original_shape": list(original_shape),
            "segmentation_shape": list(seg_array.shape),
            "segmented_voxels": voxel_count,
            "total_voxels": total_voxels,
            "segmentation_percentage": round(seg_percentage, 2),
            "disclaimer": "Research use only. Not a clinical diagnosis. Requires review by qualified healthcare professionals.",
        }
        if per_class_voxels:
            result["per_class_voxels"] = per_class_voxels
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
