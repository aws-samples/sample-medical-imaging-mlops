"""
NV-Segment-CT (VISTA-3D) SageMaker inference handler (async endpoints).

VISTA-3D's inference interface differs from typical MONAI Zoo bundles, so the
generic handler cannot drive it:
  - input is a single-image `input_dict` ({'image': path, 'label_prompt': [...]})
    instead of a dataset_dir glob;
  - the checkpoint key is `checkpoint_path` (not `ckpt_path`);
  - prompts (label_prompt / points / point_labels) select which of the 127
    classes to segment (see configs/label_dict.json; e.g. spleen = 3).

If the artifact contains a fine-tuned checkpoint (models/model_finetune.pt,
produced by src/nvmedtech/train.py) it is preferred over the pretrained
models/model.pt; override with request key "checkpoint": "models/model.pt".

Request JSON:
  {
    "image": "s3://bucket/case.nii.gz" | "<base64 NIfTI>",
    "label_prompt": [3],                # optional; omitted -> segment everything
    "points": [[x,y,z], ...],           # optional interactive prompts
    "point_labels": [1, ...],           # optional
    "output_s3_uri": "s3://bucket/masks/",   # optional; else OUTPUT_S3_URI env
    "output_filename": "case_seg.nii.gz"     # optional
  }
Response: {"mask_s3_uri": ...} (or base64 fallback when no S3 destination).

⚠️ RESEARCH USE ONLY — outputs are not clinical diagnoses.
"""
import base64
import json
import logging
import os
import shutil
import subprocess  # nosec B404 - argv-list invocation of `python -m monai.bundle` only (no shell)
import sys
import tempfile
import uuid
from typing import Any, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DISCLAIMER = "RESEARCH USE ONLY — not for clinical diagnosis; requires qualified review."
_CHECKPOINT_EXTS = (".pt", ".pth", ".ts")


# ALLOWED_INPUT_BUCKETS (comma-separated) optionally restricts input buckets.
def _s3_bucket_key(uri: str):
    """Parse s3://bucket/key and enforce ALLOWED_INPUT_BUCKETS when set."""
    from urllib.parse import urlparse
    p = urlparse(str(uri))
    key = p.path.lstrip("/")
    if p.scheme != "s3" or not p.netloc or not key:
        raise ValueError("Expected an s3://bucket/key URI")
    allowed = {b.strip() for b in os.environ.get("ALLOWED_INPUT_BUCKETS", "").split(",") if b.strip()}
    if allowed and p.netloc not in allowed:
        raise ValueError(f"S3 bucket not allowed for inference input: {p.netloc}")
    return p.netloc, key



def _check_numeric(value: Any, name: str, depth: int = 0) -> None:
    """Prompts are forwarded into MONAI's config (which EVALUATES "$..." strings),
    so only (nested lists of) numbers are accepted."""
    if depth > 4:
        raise ValueError(f"'{name}' is nested too deeply")
    if isinstance(value, bool):
        raise ValueError(f"'{name}' must contain numbers only")
    if isinstance(value, (int, float)):
        return
    if isinstance(value, list) and len(value) <= 1024:
        for v in value:
            _check_numeric(v, name, depth + 1)
        return
    raise ValueError(f"'{name}' must be a number or a list of numbers")


def _resolve_checkpoint(bundle_dir: str, requested: str) -> str:
    """Resolve a per-request checkpoint name strictly inside the bundle's models/."""
    models_dir = os.path.realpath(os.path.join(bundle_dir, "models"))
    cand = os.path.realpath(os.path.join(models_dir, os.path.basename(str(requested))))
    if (os.path.dirname(cand) != models_dir or not cand.endswith(_CHECKPOINT_EXTS)
            or not os.path.isfile(cand)):
        raise ValueError(f"Requested checkpoint not found in bundle models/: {requested}")
    return cand


def _find_bundle_dir(model_dir: str) -> str:
    for root, dirs, _ in os.walk(model_dir):
        if "configs" in dirs and os.path.isfile(os.path.join(root, "configs", "metadata.json")):
            return root
    raise ValueError(f"No MONAI bundle (configs/metadata.json) found under {model_dir}")


def model_fn(model_dir: str) -> Dict[str, Any]:
    src_bundle = _find_bundle_dir(model_dir)
    # SageMaker mounts /opt/ml/model READ-ONLY, but this bundle writes into its
    # own root at runtime (configs/logging.conf appends training.log; the HF
    # download counter touches config.json). Copy the bundle to a writable dir
    # once at model load and run from there. A private mkdtemp dir (0700, random
    # name) rather than a fixed /tmp path, so nothing else can pre-create/swap it.
    bundle_dir = os.path.join(tempfile.mkdtemp(prefix="nvsegct_"), "bundle")
    logger.info("Copying bundle to writable dir %s ...", bundle_dir)
    shutil.copytree(src_bundle, bundle_dir)
    models_dir = os.path.join(bundle_dir, "models")
    # Prefer the fine-tuned checkpoint when the artifact carries one.
    checkpoint = None
    for name in ("model_finetune.pt", "model.pt"):
        cand = os.path.join(models_dir, name)
        if os.path.isfile(cand):
            checkpoint = cand
            break
    if not checkpoint:
        raise ValueError(f"No checkpoint found under {models_dir}")
    logger.info("NV-Segment-CT ready: bundle=%s checkpoint=%s", bundle_dir, checkpoint)
    return {"bundle_dir": bundle_dir, "checkpoint": checkpoint}


def input_fn(request_body: str, content_type: str = "application/json"):
    return json.loads(request_body)


def _materialize_image(payload: Dict[str, Any], dest_dir: str) -> str:
    image = payload.get("image")
    if not image:
        raise ValueError("Request must include 'image' (s3 uri or base64 NIfTI).")
    out = os.path.join(dest_dir, "input.nii.gz")
    if isinstance(image, str) and image.startswith("s3://"):
        import boto3
        bucket, key = _s3_bucket_key(image)
        boto3.client("s3").download_file(bucket, key, out)
    else:
        with open(out, "wb") as f:
            f.write(base64.b64decode(image))
    return out


def predict_fn(payload: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    bundle_dir = model["bundle_dir"]
    with tempfile.TemporaryDirectory() as tmp:
        image_path = _materialize_image(payload, tmp)
        output_dir = os.path.join(tmp, "eval")
        os.makedirs(output_dir, exist_ok=True)

        # VISTA takes a single-image input_dict (image + optional prompts).
        input_dict: Dict[str, Any] = {"image": image_path}
        for key in ("label_prompt", "points", "point_labels"):
            if key in payload:
                _check_numeric(payload[key], key)
                input_dict[key] = payload[key]

        # Allow overriding which checkpoint to serve per request (bundle models/ only).
        checkpoint = model["checkpoint"]
        if payload.get("checkpoint"):
            checkpoint = _resolve_checkpoint(bundle_dir, payload["checkpoint"])

        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

        cmd = [
            sys.executable, "-m", "monai.bundle", "run",
            "--config_file", os.path.join("configs", "inference.json"),
            "--bundle_root", bundle_dir,
            "--input_dict", repr(input_dict),
            "--checkpoint_path", checkpoint,
            "--output_dir", output_dir,
            "--separate_folder", "false",
            "--device", f"$torch.device('{device}')",
        ]
        logger.info("Running VISTA inference: %s (cwd=%s)", " ".join(cmd), bundle_dir)
        # argv list (no shell): fixed flags, server-side paths, numeric prompts.
        proc = subprocess.run(cmd, cwd=bundle_dir, capture_output=True, text=True)  # nosec B603 - argv list, no shell  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            # Raised (not also logged): SageMaker's serving stack logs it once.
            raise RuntimeError("monai.bundle run failed:\n" + err[:4000])

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

        out_s3 = payload.get("output_s3_uri") or os.environ.get("OUTPUT_S3_URI", "")
        if out_s3:
            import boto3
            from urllib.parse import urlparse
            p = urlparse(out_s3)
            out_prefix = p.path.strip("/") or "nvmedtech-results"
            fname = payload.get("output_filename") or f"{uuid.uuid4().hex}_seg.nii.gz"
            key = f"{out_prefix}/{fname}"
            boto3.client("s3").upload_file(produced, p.netloc, key)
            mask_s3_uri = f"s3://{p.netloc}/{key}"
            logger.info("Saved segmentation mask to %s", mask_s3_uri)
            return {"mask_s3_uri": mask_s3_uri,
                    "checkpoint_used": os.path.relpath(checkpoint, bundle_dir),
                    "disclaimer": _DISCLAIMER}

        with open(produced, "rb") as f:
            mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"mask_nifti_b64": mask_b64,
            "checkpoint_used": os.path.relpath(checkpoint, bundle_dir),
            "disclaimer": _DISCLAIMER}


def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    return json.dumps(prediction)
