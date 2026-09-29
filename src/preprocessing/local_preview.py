"""Local single-image preview for the preprocessing pipeline.

Runs the pipeline on ONE image on the app server and returns before/after stats
plus a base64-encoded mid-axial slice PNG of the result. No S3 writes, no
SageMaker job — this is the cheap "does my pipeline look right?" loop.

Security (per secure-development-guidelines):
- User-supplied local paths are resolved and confined (guard against traversal).
- S3 URIs are validated against s3://bucket/key before use and downloaded to a
  confined temp dir that is always cleaned up.
- Errors returned to the client are sanitized (no internal absolute paths).

RESEARCH USE ONLY.
"""
from __future__ import annotations

import base64
import io
import os
import re
import shutil
import tempfile
import time
from typing import Any, Dict, Optional

_S3_RE = re.compile(r"^s3://([a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])/(.+)$")


class PreviewError(Exception):
    """Raised with a client-safe message (no internal paths)."""


# ── Input resolution ─────────────────────────────────────────────────────────

def _allowed_roots() -> tuple:
    """Directories a local preview input may live under.

    Defaults to the temp dir, the project dir, and ./data. Additional roots can
    be allowed for local development via the PREPROCESS_PREVIEW_ALLOWED_DIRS env
    var (os.pathsep-separated, e.g. "/Users/me/Documents/Data").

    This guard exists so a served endpoint cannot be coerced into reading
    arbitrary files (e.g. /etc/passwd). Widen it deliberately, not by default.
    """
    roots = [
        os.path.realpath(tempfile.gettempdir()),
        os.path.realpath("data"),
        os.path.realpath(os.getcwd()),
    ]
    extra = os.environ.get("PREPROCESS_PREVIEW_ALLOWED_DIRS", "")
    for d in extra.split(os.pathsep):
        d = d.strip()
        if d:
            roots.append(os.path.realpath(os.path.expanduser(d)))
    return tuple(roots)


def _validate_local_path(user_path: str, allowed_roots: tuple) -> str:
    resolved = os.path.realpath(os.path.expanduser(user_path))
    # Check existence first so a genuinely missing file gets a clear message.
    if not os.path.exists(resolved):
        raise PreviewError("Input path not found")
    if not any(resolved == r or resolved.startswith(r + os.sep) for r in allowed_roots):
        raise PreviewError(
            "Access denied: input path is outside the allowed directories. "
            "Set PREPROCESS_PREVIEW_ALLOWED_DIRS to include its parent folder, "
            "or copy the file under ./data."
        )
    return resolved


def _download_s3(input_ref: str, dest_dir: str, boto_session=None) -> str:
    """Download an S3 object or prefix into dest_dir. Returns local path.

    A single object → the local file. A prefix (or DICOM series) → the local dir.

    Uses the app's authenticated boto3 session when provided so the download runs
    with the SAME credentials the dashboard connected with. A bare boto3.client()
    would resolve credentials independently and can pick up a DIFFERENT identity
    (e.g. ~/.aws/credentials) that lacks access to the object → 403 Forbidden.
    """
    m = _S3_RE.match(input_ref)
    if not m:
        raise PreviewError("Invalid S3 URI (expected s3://bucket/key)")
    bucket, key = m.group(1), m.group(2)

    import boto3  # lazy
    s3 = boto_session.client("s3") if boto_session is not None else boto3.client("s3")

    keys = []
    is_prefix = key.endswith("/")
    if is_prefix:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=key):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith("/"):
                    keys.append(obj["Key"])
        if not keys:
            raise PreviewError("No objects found at the S3 prefix")
    else:
        keys = [key]

    for k in keys:
        rel = k[len(key):] if is_prefix else os.path.basename(k)
        # confine within dest_dir
        target = os.path.realpath(os.path.join(dest_dir, rel))
        if not target.startswith(os.path.realpath(dest_dir)):
            raise PreviewError("Refusing to write outside the temp directory")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        s3.download_file(bucket, k, target)

    return dest_dir if is_prefix else os.path.join(dest_dir, os.path.basename(key))


# ── Stats + rendering ─────────────────────────────────────────────────────────

def _to_numpy(x):
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _volume_stats(arr) -> Dict[str, Any]:
    import numpy as np
    a = _to_numpy(arr)
    # drop channel dim for reporting if present (C, H, W, D) -> spatial
    spatial = a[0] if a.ndim == 4 else a
    return {
        "shape": list(a.shape),
        "intensity_min": float(np.nanmin(a)) if a.size else None,
        "intensity_max": float(np.nanmax(a)) if a.size else None,
        "intensity_mean": float(np.nanmean(a)) if a.size else None,
        "_spatial_shape": list(spatial.shape),
    }


def _spacing_of(img_tensor) -> Optional[list]:
    """Best-effort voxel spacing from a MONAI MetaTensor affine."""
    try:
        import numpy as np
        affine = getattr(img_tensor, "affine", None)
        if affine is None:
            meta = getattr(img_tensor, "meta", {}) or {}
            affine = meta.get("affine")
        if affine is None:
            return None
        aff = _to_numpy(affine)
        sp = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
        return [round(float(s), 4) for s in sp]
    except Exception:
        return None


def _render_mid_axial_png(img_tensor) -> Optional[str]:
    """Render the middle axial slice to a base64 PNG. Returns None on failure."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        a = _to_numpy(img_tensor)
        vol = a[0] if a.ndim == 4 else a          # (H, W, D) expected
        if vol.ndim != 3:
            # 2D image: just show it
            slice2d = vol
        else:
            mid = vol.shape[2] // 2               # axial = last axis
            slice2d = vol[:, :, mid]

        fig, ax = plt.subplots(figsize=(4, 4))
        # rotate for conventional radiological display
        ax.imshow(np.rot90(slice2d), cmap="gray")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def preview(spec, input_ref: str, boto_session=None) -> Dict[str, Any]:
    """Run the pipeline on one image and return stats + a mid-axial slice PNG.

    Args:
        spec: a PipelineSpec.
        input_ref: local path or s3:// URI to a single image (NIfTI) or a
                   directory/prefix (DICOM series).

    Returns a dict:
        { input: {...}, output: {...}, preview_image_base64: str|None,
          elapsed_seconds: float, disclaimer: str }
    """
    from src.preprocessing.pipeline_builder import build_transforms

    tmp_dir = None
    try:
        if input_ref.startswith("s3://"):
            tmp_dir = tempfile.mkdtemp(prefix="pp_preview_")
            local_input = _download_s3(input_ref, tmp_dir, boto_session=boto_session)
        else:
            local_input = _validate_local_path(input_ref, _allowed_roots())

        keys = list(spec.keys)
        # For preview we only have the image; map every key to the same input so
        # the dictionary transforms run (label preview is out of scope here).
        sample = {k: local_input for k in keys}

        transforms = build_transforms(spec, for_preview=True)

        t0 = time.time()
        # Capture "before" by running just the load step for stats.
        loaded = None
        try:
            from src.preprocessing.pipeline_builder import build_transforms as _bt
            from app.models.preprocessing import PipelineSpec, Step
            load_only = PipelineSpec(
                task_type=spec.task_type, keys=keys,
                steps=[Step(type="load", params={}),
                       Step(type="ensure_channel_first", params={})],
            )
            loaded = _bt(load_only, for_preview=True)({k: local_input for k in keys})
        except Exception:
            loaded = None

        result = transforms(sample)
        elapsed = time.time() - t0

        out_img = result["image"] if isinstance(result, dict) else result
        in_stats = {}
        if loaded is not None:
            in_img = loaded["image"] if isinstance(loaded, dict) else loaded
            in_stats = _volume_stats(in_img)
            in_stats["spacing"] = _spacing_of(in_img)

        out_stats = _volume_stats(out_img)
        out_stats["spacing"] = _spacing_of(out_img)

        png = _render_mid_axial_png(out_img)

        return {
            "input": in_stats,
            "output": out_stats,
            "preview_image_base64": png,
            "elapsed_seconds": round(elapsed, 3),
            "disclaimer": "RESEARCH USE ONLY — preprocessing performs no diagnosis.",
        }
    except PreviewError:
        raise
    except Exception as e:
        # Sanitize: strip anything that looks like an absolute path.
        msg = re.sub(r"(/[^\s'\"]+)", "<path>", str(e))
        raise PreviewError(f"Preview failed: {msg}") from None
    finally:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
