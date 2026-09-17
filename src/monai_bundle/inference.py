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
    # Class label map for classification bundles (e.g. {'0':'A','1':'B',...}).
    meta = info.metadata or {}
    channel_def = (((meta.get("network_data_format") or {}).get("outputs") or {})
                   .get("pred") or {}).get("channel_def")
    # Does the inference config read its inputs from a JSON datalist (a 'data'
    # component with a 'filename' arg, e.g. scripts.createList.CreateImageLabelList)?
    # And which run id does it execute? (monai.bundle defaults to 'run', but e.g.
    # breast_density_classification names its section 'evaluating'.)
    needs_datalist = False
    run_id = "run"
    try:
        with open(os.path.join(bundle_dir, "configs", inference_config), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        data_cfg = cfg.get("data")
        needs_datalist = isinstance(data_cfg, dict) and "filename" in data_cfg
        for cand in ("run", "evaluating", "inferring", "infer", "evaluate"):
            if cand in cfg:
                run_id = cand
                break
    except Exception:  # noqa: BLE001 - YAML or unusual config; keep defaults
        pass
    logger.info("Bundle ready: config=%s checkpoint=%s datalist=%s run_id=%s",
                inference_config, checkpoint, needs_datalist, run_id)
    return {"bundle_dir": bundle_dir, "inference_config": inference_config,
            "checkpoint": checkpoint, "channel_def": channel_def,
            "needs_datalist": needs_datalist, "run_id": run_id}


_IMAGE_EXTS = (".nii.gz", ".nii", ".jpg", ".jpeg", ".png", ".bmp", ".dcm")


def _materialize_image(payload: Dict[str, Any], dest_dir: str) -> str:
    """Write the request image to a local file (preserving its format extension)
    and return its absolute path. NIfTI is the default; 2D bundles (e.g. JPEG
    mammography classification) are supported via the S3 key's extension or an
    explicit 'image_format' field for base64 payloads."""
    image = payload.get("image")
    if not image:
        raise ValueError("Request must include 'image' (s3 uri, local path, or base64).")

    ext = ".nii.gz"
    fmt = str(payload.get("image_format", "")).lower().lstrip(".")
    if fmt:
        ext = "." + ("nii.gz" if fmt == "nii.gz" else fmt)
    if isinstance(image, str) and image.startswith("s3://"):
        low = image.lower()
        for e in _IMAGE_EXTS:
            if low.endswith(e):
                ext = e
                break

    out = os.path.join(dest_dir, "input" + ext)
    if isinstance(image, str) and image.startswith("s3://"):
        import boto3
        from urllib.parse import urlparse
        p = urlparse(image)
        boto3.client("s3").download_file(p.netloc, p.path.lstrip("/"), out)
    elif isinstance(image, str) and os.path.isfile(image):
        out = image
    else:
        # base64-encoded image bytes
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
        # MONAI inference bundles typically build their input list by globbing —
        # some under a subdir (spleen: dataset_dir/imagesTs/*.nii.gz), some at the
        # dataset root (HoVer-Net: dataset_dir/*.png). Materialize the request
        # image into imagesTs/ and hardlink it at the root so either style works.
        dataset_dir = os.path.join(tmp, "dataset")
        images_dir = os.path.join(dataset_dir, "imagesTs")
        os.makedirs(images_dir, exist_ok=True)
        image_path = _materialize_image(payload, images_dir)
        root_copy = os.path.join(dataset_dir, os.path.basename(image_path))
        if not os.path.exists(root_copy):
            try:
                os.link(image_path, root_copy)
            except OSError:
                import shutil as _sh
                _sh.copy2(image_path, root_copy)

        output_dir = os.path.join(tmp, "eval")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable, "-m", "monai.bundle", "run",
            "--config_file", os.path.join("configs", model["inference_config"]),
            "--bundle_root", bundle_dir,
            "--dataset_dir", dataset_dir,
            "--output_dir", output_dir,
        ]
        if model.get("run_id") and model["run_id"] != "run":
            cmd += ["--run_id", model["run_id"]]
        if model.get("checkpoint"):
            cmd += ["--ckpt_path", model["checkpoint"]]

        # Classification bundles (e.g. breast_density_classification) read their
        # input list from a JSON datalist via scripts.createList.CreateImageLabelList
        # ({"Test": [{"image": ..., "label": ...}]}) instead of globbing a
        # dataset_dir. Write a single-image datalist and point 'data#filename' at
        # it so the same handler drives them.
        if model.get("needs_datalist"):
            datalist_path = os.path.join(tmp, "datalist.json")
            with open(datalist_path, "w", encoding="utf-8") as f:
                json.dump({g: [{"image": image_path, "label": 0}]
                           for g in ("Test", "Train", "Val")}, f)
            cmd += ["--data#filename", datalist_path]

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

        # Arbitrary per-request config overrides (advanced escape hatch).
        for k, v in (payload.get("config_overrides") or {}).items():
            cmd += [f"--{k}", v if isinstance(v, str) else json.dumps(v)]

        logger.info("Running inference: %s (cwd=%s)", " ".join(cmd), bundle_dir)
        proc = subprocess.run(cmd, cwd=bundle_dir, capture_output=True, text=True)
        if proc.returncode != 0:
            # Surface MONAI's full stderr (root cause is often near the top).
            err = (proc.stderr or proc.stdout or "").strip()
            logger.error("monai.bundle run failed:\n%s", err)
            # Return a trimmed-from-the-END message so the true first exception is kept.
            msg = "monai.bundle run failed:\n" + err[:4000]
            raise RuntimeError(msg)

        # Find the produced output. Segmentation bundles write a NIfTI mask;
        # classification bundles (ClassificationSaver) write predictions.csv.
        produced, produced_csv = None, None
        for search in (output_dir, os.path.join(bundle_dir, "eval"),
                       os.path.join(bundle_dir, "output")):
            if not os.path.isdir(search):
                continue
            for cur, _, files in os.walk(search):
                for fn in files:
                    if fn.endswith((".nii", ".nii.gz")) and not produced:
                        produced = os.path.join(cur, fn)
                    elif fn.endswith(".csv") and not produced_csv:
                        produced_csv = os.path.join(cur, fn)

        # ── Classification output ────────────────────────────────────────────
        if produced_csv and not produced:
            import csv as _csv
            with open(produced_csv, newline="") as f:
                rows = list(_csv.reader(f))
            # ClassificationSaver rows: filename, prob_0, prob_1, ...
            probs = [float(x) for x in rows[-1][1:]] if rows and len(rows[-1]) > 1 else []
            channel_def = model.get("channel_def") or {}
            labels = [channel_def.get(str(i), str(i)) for i in range(len(probs))]
            result: Dict[str, Any] = {
                "task": "classification",
                "probabilities": dict(zip(labels, probs)),
                "disclaimer": _DISCLAIMER,
            }
            if probs:
                best = max(range(len(probs)), key=lambda i: probs[i])
                result["predicted_class"] = labels[best]
                result["confidence"] = probs[best]
            return result

        if not produced:
            raise RuntimeError("Inference produced no NIfTI or CSV output")

        # Multi-output bundles (e.g. HoVer-Net: instance_map + type_map) produce
        # several NIfTIs — collect them all.
        all_masks = []
        for search in (output_dir, os.path.join(bundle_dir, "eval"),
                       os.path.join(bundle_dir, "output")):
            if not os.path.isdir(search):
                continue
            for cur, _, files in os.walk(search):
                for fn in sorted(files):
                    if fn.endswith((".nii", ".nii.gz")):
                        all_masks.append(os.path.join(cur, fn))

        # Upload the produced mask NIfTI(s) to S3 and return URI(s) (cleaner than
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
            base = payload.get("output_filename") or f"{uuid.uuid4().hex}_seg.nii.gz"
            base_stem = base[:-7] if base.endswith(".nii.gz") else os.path.splitext(base)[0]
            s3 = boto3.client("s3")
            uris = []
            for i, m in enumerate(all_masks):
                # single output keeps the requested name; extra outputs get the
                # bundle's own postfix (e.g. _instance_map/_type_map) appended.
                if len(all_masks) == 1:
                    fname = base
                else:
                    tag = os.path.basename(m)
                    tag = tag[:-7] if tag.endswith(".nii.gz") else os.path.splitext(tag)[0]
                    fname = f"{base_stem}_{tag}.nii.gz"
                key = f"{out_prefix}/{fname}"
                s3.upload_file(m, out_bucket, key)
                uris.append(f"s3://{out_bucket}/{key}")
            logger.info("Saved %d mask(s) to %s", len(uris), out_s3)
            resp: Dict[str, Any] = {"mask_s3_uri": uris[0], "disclaimer": _DISCLAIMER}
            if len(uris) > 1:
                resp["mask_s3_uris"] = uris
            return resp

        # No S3 destination configured — fall back to base64 in the response.
        with open(produced, "rb") as f:
            mask_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"mask_nifti_b64": mask_b64, "disclaimer": _DISCLAIMER}


def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    return json.dumps(prediction)
