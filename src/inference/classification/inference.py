"""Inference script for medical image classification.

Loaded by the SageMaker PyTorch container via SAGEMAKER_PROGRAM=inference.py.
model.tar.gz is expected to contain:
  - best_model.pth / model.pth   (weights)
  - result.json                  (training provenance — model name, accuracy, etc.)
  - label_dict.json              (class index → label mapping)
  - model_def.py                 (model architecture definitions)
  - inference.py                 (this file)
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, Union

import boto3
import torch

from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Resize, ScaleIntensity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _s3_to_bucket_key(s3_uri: str):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _download_s3(s3_uri: str, suffix: str = ".dcm") -> str:
    bucket, key = _s3_to_bucket_key(s3_uri)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.flush()
        tmp_path = tmp.name
    boto3.client("s3").download_file(bucket, key, tmp_path)
    logger.info("Downloaded %s → %s", s3_uri, tmp_path)
    return tmp_path


# ── SageMaker handler functions ───────────────────────────────────────────────

def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load model using architecture from result.json + weights from best_model.pth."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s  |  model_dir contents: %s", device, os.listdir(model_dir))

    # --- result.json: model name ---
    result_path = os.path.join(model_dir, "result.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError("result.json not found in model_dir — was the model trained with the updated train.py?")
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)
    model_name = result["model_name"]
    logger.info("Model name from result.json: %s", model_name)

    # --- label_dict.json: num_classes ---
    label_dict_path = os.path.join(model_dir, "label_dict.json")
    if not os.path.exists(label_dict_path):
        raise FileNotFoundError("label_dict.json not found in model_dir")
    with open(label_dict_path, encoding="utf-8") as f:
        label_dict = json.load(f)
    num_classes = len(label_dict)
    logger.info("num_classes from label_dict.json: %d  labels: %s", num_classes, list(label_dict.keys()))

    # --- build model from model_def.py ---
    from model_def import ModelDef
    model = ModelDef(num_classes=num_classes, model_name=model_name).get_model()

    # Load weights — priority: model_deploy.pth (weights-only) → best_model.pth → model.pth
    for candidate, is_full_checkpoint in (
        ("model_deploy.pth", False),
        ("best_model.pth",   True),
        ("model.pth",        True),
    ):
        weights_path = os.path.join(model_dir, candidate)
        if os.path.exists(weights_path):
            # weights_only=True prevents arbitrary code execution from untrusted model files
            checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
            state_dict = checkpoint["model"] if is_full_checkpoint and isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            logger.info("Loaded weights from %s", candidate)
            break
    else:
        raise FileNotFoundError("No model weights found (model_deploy.pth, best_model.pth or model.pth)")

    model.to(device).eval()

    transforms = Compose([
        LoadImage(),
        EnsureChannelFirst(),
        Resize(spatial_size=(256, 256, 1)),
        ScaleIntensity(),
    ])

    # Invert label_dict to idx → label for output_fn
    idx_to_label = {int(v): k for k, v in label_dict.items()}

    return {
        "model":        model,
        "device":       device,
        "transforms":   transforms,
        "idx_to_label": idx_to_label,
        "result":       result,
    }


def input_fn(request_body: Union[str, bytes], content_type: str) -> Dict[str, Any]:
    """Return a dict with local file path and metadata for output saving."""
    if content_type == "application/dicom":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            tmp.write(request_body)
            tmp.flush()
            tmp_path = tmp.name
        logger.info("Received raw DICOM bytes → %s", tmp_path)
        return {"local_path": tmp_path}

    if content_type == "application/json":
        data = json.loads(request_body)

        output_s3_uri = data.get("output_s3_uri", os.environ.get("OUTPUT_S3_URI", ""))
        output_pattern = data.get("output_pattern", "")
        s3_uri = data.get("s3_uri") or data.get("file_path", "")

        if s3_uri.startswith("s3://"):
            local_path = _download_s3(s3_uri)
            return {"local_path": local_path, "s3_uri": s3_uri, "output_s3_uri": output_s3_uri, "output_pattern": output_pattern}

        if os.path.exists(s3_uri):
            return {"local_path": s3_uri, "output_s3_uri": output_s3_uri, "output_pattern": output_pattern}

        raise ValueError(f"Cannot resolve input: {s3_uri}")

    raise ValueError(f"Unsupported content_type: {content_type}")


def predict_fn(input_data: Dict[str, Any], model_components: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on the image and return prediction + metadata for output saving."""
    model      = model_components["model"]
    device     = model_components["device"]
    transforms = model_components["transforms"]

    local_path = input_data["local_path"] if isinstance(input_data, dict) else input_data
    image_tensor = transforms(local_path).unsqueeze(0).to(device)

    # Handle 5-D output from some MONAI transforms (H,W,D) → drop depth dim
    if image_tensor.dim() == 5:
        image_tensor = image_tensor[..., 0]

    with torch.no_grad():
        output = model(image_tensor)

    # Some models (e.g. ViT) return a tuple
    if isinstance(output, (list, tuple)):
        output = output[0]

    # Pass metadata through for output_fn to save to S3
    return {
        "prediction": output,
        "s3_uri": input_data.get("s3_uri", "") if isinstance(input_data, dict) else "",
        "output_s3_uri": input_data.get("output_s3_uri", "") if isinstance(input_data, dict) else "",
        "output_pattern": input_data.get("output_pattern", "") if isinstance(input_data, dict) else "",
    }


def output_fn(prediction: Dict[str, Any], accept: str) -> str:
    """Format prediction as JSON with human-readable labels. Save to S3 if output_s3_uri provided."""
    pred_tensor   = prediction["prediction"]
    source_s3_uri = prediction.get("s3_uri", "")
    output_s3_uri = prediction.get("output_s3_uri", "")
    output_pattern_override = prediction.get("output_pattern", "")

    probs         = torch.softmax(pred_tensor, dim=1)
    predicted_idx = int(probs.argmax(dim=1).item())
    confidence    = float(probs.max(dim=1)[0].item())

    idx_to_label = _label_cache.get("idx_to_label", {})
    result_meta  = _label_cache.get("result", {})

    prob_list = probs.squeeze(0).cpu().tolist()
    prob_dict = {idx_to_label.get(i, str(i)): round(p, 4) for i, p in enumerate(prob_list)}

    response = {
        "predicted_class": idx_to_label.get(predicted_idx, str(predicted_idx)),
        "confidence":      round(confidence, 4),
        "probabilities":   prob_dict,
        "model_name":      result_meta.get("model_name"),
        "num_classes":     len(prob_list),
    }

    if accept == "application/json;verbose":
        response["training_info"] = {
            "final_epoch":      result_meta.get("final_epoch"),
            "train_accuracy":   result_meta.get("train_accuracy"),
            "val_accuracy":     result_meta.get("val_accuracy"),
            "best_val_accuracy":result_meta.get("best_val_accuracy"),
            "data_s3_uri":      result_meta.get("data_s3_uri"),
        }

    # Save result to S3 with configurable filename pattern
    if output_s3_uri and source_s3_uri:
        try:
            from datetime import datetime
            s3_client = boto3.client("s3")

            # Extract metadata from input path
            source_parts = source_s3_uri.replace("s3://", "").split("/")
            patient_id = source_parts[-2] if len(source_parts) >= 3 else "unknown"
            input_filename = source_parts[-1].split(".")[0]
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "endpoint")

            output_pattern = output_pattern_override or os.environ.get(
                "OUTPUT_FILENAME_PATTERN", "{patient_id}_classification_{timestamp}.json"
            )
            output_filename = output_pattern.format(
                patient_id=patient_id,
                filename=input_filename,
                endpoint_name=endpoint_name,
                timestamp=timestamp,
                model_name=result_meta.get("model_name", "classification"),
            )

            out_parts = output_s3_uri.replace("s3://", "").rstrip("/").split("/", 1)
            out_bucket = out_parts[0]
            out_prefix = out_parts[1] if len(out_parts) > 1 else "classification-results"

            result_key = f"{out_prefix}/{output_filename}"
            s3_client.put_object(
                Bucket=out_bucket,
                Key=result_key,
                Body=json.dumps(response, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            response["saved_to"] = f"s3://{out_bucket}/{result_key}"
            logger.info("Saved classification result to: s3://%s/%s", out_bucket, result_key)
        except Exception as e:
            logger.warning("Failed to save result to S3: %s", e)

    return json.dumps(response)


# ── Module-level cache so output_fn can access label map ─────────────────────
# SageMaker calls model_fn once then output_fn per request; we bridge via this.
_label_cache: Dict[str, Any] = {}

_original_model_fn = model_fn

def model_fn(model_dir: str) -> Dict[str, Any]:  # noqa: F811
    components = _original_model_fn(model_dir)
    _label_cache["idx_to_label"] = components["idx_to_label"]
    _label_cache["result"]       = components["result"]
    return components
