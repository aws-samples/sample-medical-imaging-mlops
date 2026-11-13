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
import re
import subprocess  # nosec B404 - argv-list invocation of `python -m monai.bundle` only (no shell)
import sys
import tempfile
import uuid
from typing import Any, Dict, Union

from bundle_utils import inspect_bundle, bundle_base_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DISCLAIMER = "RESEARCH USE ONLY — not for clinical diagnosis; requires qualified review."

# ── Request input confinement (Slingshot 2026-11-12) ─────────────────────────
# Requests normally send an S3 URI or raw bytes. A local path is honoured only
# when it resolves inside LOCAL_INPUT_ROOTS (comma-separated; default: the
# SageMaker data mounts — never the model directory). ALLOWED_INPUT_BUCKETS
# (comma-separated) optionally restricts which buckets the endpoint role reads.
_DEFAULT_LOCAL_INPUT_ROOTS = "/opt/ml/input/data,/opt/ml/processing/input"
_PATTERN_FIELDS = {"patient_id", "filename", "endpoint_name", "timestamp", "model_name"}


def _local_input_roots():
    raw = os.environ.get("LOCAL_INPUT_ROOTS", _DEFAULT_LOCAL_INPUT_ROOTS)
    return tuple(os.path.realpath(r.strip()) for r in raw.split(",") if r.strip())


def _is_confined_local_path(path) -> bool:
    """True only for an absolute path that resolves inside LOCAL_INPUT_ROOTS.
    Anything else (including base64 such as JPEG's '/9j/...') is not a path."""
    s = str(path)
    if not s.startswith("/"):
        return False
    real = os.path.realpath(s)
    return any(os.path.commonpath([root, real]) == root for root in _local_input_roots())


def _validated_local_path(path) -> str:
    real = os.path.realpath(str(path))
    if not any(os.path.commonpath([root, real]) == root for root in _local_input_roots()):
        raise ValueError("Local input paths are only accepted under LOCAL_INPUT_ROOTS")
    if not os.path.isfile(real):
        raise FileNotFoundError("Input file not found")
    return real


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


def _safe_output_pattern(pattern: str, default: str) -> str:
    """Only allow known {placeholders} (no attribute/index access, conversions or
    format specs) and no path traversal in a caller-supplied filename pattern."""
    import string
    try:
        for _lit, field, spec, conv in string.Formatter().parse(pattern):
            if field is None:
                continue
            if field not in _PATTERN_FIELDS or spec or conv:
                raise ValueError(field)
    except ValueError:
        logger.warning("Rejected output filename pattern; using default")
        return default
    if ".." in pattern or pattern.startswith("/"):
        logger.warning("Rejected output filename pattern with path traversal; using default")
        return default
    return pattern


# ── Request validation for values forwarded to `monai.bundle run` ─────────────
# MONAI's ConfigParser EVALUATES strings starting with "$" as Python expressions,
# resolves "@" references and "%" macros (which can load other files). Request
# payload values therefore must never reach the CLI unvalidated: prompts must be
# numeric, and config overrides are limited to plain literals on non-reserved keys.
_MONAI_EXPR_PREFIXES = ("$", "@", "%")
_OVERRIDE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(#[A-Za-z0-9_]+)*$")
_RESERVED_OVERRIDE_KEYS = {
    "config_file", "meta_file", "logging_file", "args_file", "bundle_root",
    "dataset_dir", "output_dir", "run_id", "ckpt_path", "checkpoint_path",
    "device", "data#filename", "init_id", "final_id",
}
_MAX_OVERRIDE_ITEMS = 32


def _check_numeric(value: Any, name: str, depth: int = 0) -> None:
    """Allow only (nested lists of) numbers — for label_prompt/points/point_labels."""
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


def _check_literal(value: Any, name: str, depth: int = 0) -> None:
    """Plain JSON literal with no MONAI expression/reference/macro strings."""
    if depth > 4:
        raise ValueError(f"override '{name}' is nested too deeply")
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        if len(value) > 1024 or value.lstrip().startswith(_MONAI_EXPR_PREFIXES):
            raise ValueError(f"override '{name}': expression/reference values are not allowed")
        return
    if isinstance(value, list) and len(value) <= 256:
        for v in value:
            _check_literal(v, name, depth + 1)
        return
    if isinstance(value, dict) and len(value) <= 64:
        for k, v in value.items():
            if not isinstance(k, str) or k.startswith(_MONAI_EXPR_PREFIXES):
                raise ValueError(f"override '{name}': invalid nested key")
            _check_literal(v, name, depth + 1)
        return
    raise ValueError(f"override '{name}' has an unsupported value type")


def _validated_overrides(raw: Any) -> Dict[str, Any]:
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict) or len(raw) > _MAX_OVERRIDE_ITEMS:
        raise ValueError("config_overrides must be an object with at most "
                         f"{_MAX_OVERRIDE_ITEMS} keys")
    for k, v in raw.items():
        if not isinstance(k, str) or not _OVERRIDE_KEY_RE.match(k) or k in _RESERVED_OVERRIDE_KEYS:
            raise ValueError(f"config override key not allowed: {k!r}")
        _check_literal(v, k)
    return raw


def model_fn(model_dir: str) -> Dict[str, Any]:
    """Resolve the bundle: inference config + checkpoint."""
    bundle_dir = bundle_base_dir(model_dir)
    info = inspect_bundle(bundle_dir)
    if not info.valid:
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
    except Exception as e:  # noqa: BLE001 - YAML or unusual config; keep defaults
        logger.debug("Could not inspect %s for datalist/run_id (using defaults): %s",
                     inference_config, e)
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
        bucket, key = _s3_bucket_key(image)
        boto3.client("s3").download_file(bucket, key, out)
    elif isinstance(image, str) and len(image) < 4096 and _is_confined_local_path(image):
        out = _validated_local_path(image)
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
        # bundles that don't declare these ignore them. Numeric-only (see above).
        for key in ("label_prompt", "points", "point_labels"):
            if key in payload:
                _check_numeric(payload[key], key)
                cmd += [f"--{key}", json.dumps(payload[key])]

        # Per-request config overrides (advanced escape hatch): plain literals on
        # non-reserved keys only — never MONAI expressions/references/macros.
        for k, v in _validated_overrides(payload.get("config_overrides")).items():
            cmd += [f"--{k}", v if isinstance(v, str) else json.dumps(v)]

        logger.info("Running inference: %s (cwd=%s)", " ".join(cmd), bundle_dir)
        # argv list (no shell); every element is a fixed flag, a server-side path,
        # or a value validated above.
        proc = subprocess.run(cmd, cwd=bundle_dir, capture_output=True, text=True)  # nosec B603 - argv list, no shell  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        if proc.returncode != 0:
            # Surface MONAI's full stderr (root cause is often near the top) in the
            # exception; SageMaker's serving stack logs the raised error once.
            err = (proc.stderr or proc.stdout or "").strip()
            # Return a trimmed-from-the-END message so the true first exception is kept.
            raise RuntimeError("monai.bundle run failed:\n" + err[:4000])

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
            with open(produced_csv, newline="", encoding="utf-8") as f:
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
