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
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    boto3.client("s3").download_file(bucket, key, tmp.name)
    logger.info("Downloaded %s → %s", s3_uri, tmp.name)
    return tmp.name


# ── SageMaker handler functions ───────────────────────────────────────────────

def model_fn(model_dir: str) -> Dict[str, Any]:
    """Load model using architecture from result.json + weights from best_model.pth."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s  |  model_dir contents: %s", device, os.listdir(model_dir))

    # --- result.json: model name ---
    result_path = os.path.join(model_dir, "result.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError("result.json not found in model_dir — was the model trained with the updated train.py?")
    with open(result_path) as f:
        result = json.load(f)
    model_name = result["model_name"]
    logger.info("Model name from result.json: %s", model_name)

    # --- label_dict.json: num_classes ---
    label_dict_path = os.path.join(model_dir, "label_dict.json")
    if not os.path.exists(label_dict_path):
        raise FileNotFoundError("label_dict.json not found in model_dir")
    with open(label_dict_path) as f:
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
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
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


def input_fn(request_body: Union[str, bytes], content_type: str) -> str:
    """Return a local file path to the DICOM image."""
    if content_type == "application/dicom":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
        tmp.write(request_body)
        tmp.flush()
        logger.info("Received raw DICOM bytes → %s", tmp.name)
        return tmp.name

    if content_type == "application/json":
        data = json.loads(request_body)

        if "file_path" in data:
            path = data["file_path"]
            if path.startswith("s3://"):
                return _download_s3(path)
            if os.path.exists(path):
                return path
            raise ValueError(f"file_path not found: {path}")

        if "s3_uri" in data:
            return _download_s3(data["s3_uri"])

    raise ValueError(f"Unsupported content_type: {content_type}")


def predict_fn(input_data: str, model_components: Dict[str, Any]) -> torch.Tensor:
    """Run inference on the image at input_data path."""
    model      = model_components["model"]
    device     = model_components["device"]
    transforms = model_components["transforms"]

    image_tensor = transforms(input_data).unsqueeze(0).to(device)

    # Handle 5-D output from some MONAI transforms (H,W,D) → drop depth dim
    if image_tensor.dim() == 5:
        image_tensor = image_tensor[..., 0]

    with torch.no_grad():
        output = model(image_tensor)

    # Some models (e.g. ViT) return a tuple
    if isinstance(output, (list, tuple)):
        output = output[0]

    return output


def output_fn(prediction: torch.Tensor, accept: str) -> str:
    """Format prediction as JSON with human-readable labels."""
    probs         = torch.softmax(prediction, dim=1)
    predicted_idx = int(probs.argmax(dim=1).item())
    confidence    = float(probs.max(dim=1)[0].item())

    # model_components not directly available here, but idx_to_label is
    # injected via a module-level cache set during model_fn
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
