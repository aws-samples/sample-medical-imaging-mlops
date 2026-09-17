"""
Generic MONAI Bundle inference handler for SageMaker.

Runs ANY MONAI bundle's inference config via `python -m monai.bundle run`,
using the packaged checkpoint. Accepts a NIfTI image via S3 URI, local path,
or base64 payload, and returns the produced segmentation as base64 NIfTI.

⚠️ RESEARCH USE ONLY — This inference handler produces AI predictions for
research and development only. Outputs must NOT be used as the sole basis for
clinical decisions. Clinical deployment requires regulatory clearance, clinical
validation, qualified professional review, and fairness/bias evaluation.
"""
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Dict, Union

from bundle_utils import inspect_bundle, bundle_base_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DISCLAIMER = "RESEARCH USE ONLY — not for clinical diagnosis; requires qualified review."


def model_fn(model_dir: str) -> Dict[str, Any]:
    """Resolve the bundle: inference config + checkpoint."""
    bundle_dir = bundle_base_dir(model_dir)
    info = inspect_bundle(bundle_dir)
    if not info.is_valid:
        raise ValueError(f"No valid MONAI bundle found in {model_dir}")
    inference_config = info.default_inference_config()
    if not inference_config:
        raise ValueError(
            f"No inference config found in bundle. Configs: {info.config_files}"
        )
    checkpoint = info.checkpoints[0] if info.checkpoints else None
    logger.info("Bundle ready: config=%s checkpoint=%s", inference_config, checkpoint)
    return {"bundle_dir": bundle_dir, "inference_config": inference_config, "checkpoint": checkpoint}


def _materialize_image(payload: Dict[str, Any], dest_dir: str) -> str:
    """Write the request image to a local .nii.gz and return its absolute path."""
    image = payload.get("image")
    if not image:
        raise ValueError("Request must include 'image' (s3 uri, local path, or base64 NIfTI).")

    out = os.path.join(dest_dir, "input.nii.gz")
    if isinstance(image, str) and image.startswith("s3://"):
        import boto3
        from urllib.parse import urlparse
        p = urlparse(image)
        boto3.client("s3").download_file(p.netloc, p.path.lstrip("/"), out)
    elif isinstance(image, str) and os.path.isfile(image):
        out = image
    else:
        # base64-encoded NIfTI bytes
        with open(out, "wb") as f:
            f.write(base64.b64decode(image))
    return os.path.abspath(out)


def input_fn(request_body: Union[str, bytes], content_type: str = "application/json") -> Dict[str, Any]:
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def predict_fn(payload: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    bundle_dir = model["bundle_dir"]
    with tempfile.TemporaryDirectory() as tmp:
        # MONAI inference bundles typically build their input list by globbing a
        # dataset_dir (e.g. spleen: glob(dataset_dir + '/imagesTs/*.nii.gz')),
        # rather than taking a single --input_file. So we materialize the request
        # image into an imagesTs/ layout under a temp dataset_dir and point the
        # bundle at it via config overrides.
        dataset_dir = os.path.join(tmp, "dataset")
        images_dir = os.path.join(dataset_dir, "imagesTs")
        os.makedirs(images_dir, exist_ok=True)
        _materialize_image(payload, images_dir)   # writes <images_dir>/input.nii.gz

        output_dir = os.path.join(tmp, "eval")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable, "-m", "monai.bundle", "run",
            "--config_file", os.path.join("configs", model["inference_config"]),
            "--bundle_root", bundle_dir,
            "--dataset_dir", dataset_dir,
            "--output_dir", output_dir,
        ]
        if model.get("checkpoint"):
            cmd += ["--ckpt_path", model["checkpoint"]]

        # Select device to match the host: bundles often default to cuda, which
        # fails to even load the checkpoint on a CPU-only host. On a GPU endpoint
        # this resolves to cuda. Overriding --device keeps one image working on
        # both. (torch import is cheap here.)
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
        cmd += ["--device", f"$torch.device('{device}')"]

        # Pass through supported optional inference params (e.g. VISTA3D prompts);
        # bundles that don't declare these ignore them.
        for key in ("label_prompt", "points", "point_labels"):
            if key in payload:
                cmd += [f"--{key}", json.dumps(payload[key])]

        logger.info("Running inference: %s (cwd=%s)", " ".join(cmd), bundle_dir)
        proc = subprocess.run(cmd, cwd=bundle_dir, capture_output=True, text=True)
        if proc.returncode != 0:
            # Surface MONAI's full stderr (root cause is often near the top).
            err = (proc.stderr or proc.stdout or "").strip()
            logger.error("monai.bundle run failed:\n%s", err)
            # Return a trimmed-from-the-END message so the true first exception is kept.
            msg = "monai.bundle run failed:\n" + err[:4000]
            raise RuntimeError(msg)

        # Find the produced segmentation NIfTI (search both output_dir and the
        # bundle's own eval dir, since some configs write under bundle_root/eval).
        produced = None
        for search in (output_dir, os.path.join(bundle_dir, "eval")):
            if not os.path.isdir(search):
                continue
            for cur, _, files in os.walk(search):
                for fn in files:
                    if fn.endswith((".nii", ".nii.gz")):
                        produced = os.path.join(cur, fn)
                        break
                if produced:
                    break
            if produced:
                break
        if not produced:
            raise RuntimeError("Inference produced no NIfTI output")

        # Upload the produced mask NIfTI to S3 and return its URI (cleaner than
        # base64 for large 3D volumes). Destination from the request or the
        # OUTPUT_S3_URI env var (set on the endpoint). Falls back to base64 only
        # if no S3 destination is configured.
        out_s3 = payload.get("output_s3_uri") or os.environ.get("OUTPUT_S3_URI", "")
        if out_s3:
            import boto3
            from urllib.parse import urlparse
            p = urlparse(out_s3)
            out_bucket = p.netloc
            out_prefix = p.path.strip("/") or "monai-bundle-results"
            fname = payload.get("output_filename") or f"{uuid.uuid4().hex}_seg.nii.gz"
            key = f"{out_prefix}/{fname}"
            boto3.client("s3").upload_file(produced, out_bucket, key)
            mask_s3_uri = f"s3://{out_bucket}/{key}"
            logger.info("Saved segmentation mask to %s", mask_s3_uri)
            return {"mask_s3_uri": mask_s3_uri, "disclaimer": _DISCLAIMER}

        # No S3 destination configured — fall back to base64 in the response.
        with open(produced, "rb") as f:
            mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"mask_nifti_b64": mask_b64, "disclaimer": _DISCLAIMER}


def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    return json.dumps(prediction)
