"""Viewer API — presigned access to NIfTI volumes/masks for the NiiVue viewer.

RESEARCH USE ONLY — rendered predictions are not clinical diagnoses.
"""
import logging
import os
import re
import tarfile
from typing import Any, Dict
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.routers.aws_router import aws_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Prefixes where MedAnvil's async endpoints write prediction masks.
MASK_PREFIXES = [
    "mldashboard-output/monai-bundle-test/masks/",
    "mldashboard-output/nvmedtech/masks/",
]
DEFAULT_BUCKET = os.environ.get("VIEWER_MASKS_BUCKET", "")
PRESIGN_EXPIRY_SECONDS = 900  # 15 min — viewer loads immediately

# SageMaker endpoint names: alphanumerics and hyphens, max 63 chars.
_ENDPOINT_NAME_RE = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")
# Label-map lookup streams the model.tar.gz from the start and stops at the
# label file. Bundles are packaged with configs/ first, so this is normally a
# few KB; the cap stops us streaming a multi-GB checkpoint to find nothing.
_LABEL_SCAN_MAX_BYTES = 64 * 1024 * 1024
_LABEL_FILE_MAX_BYTES = 2 * 1024 * 1024
_label_cache: Dict[str, Dict[str, Any]] = {}   # ModelDataUrl -> response


def _labels_from_metadata(meta: Any) -> Dict[str, str]:
    """channel_def from a MONAI bundle metadata.json: {"0": "background", "1": "spleen"}."""
    if not isinstance(meta, dict):
        return {}
    ndf = meta.get("network_data_format") or {}
    outputs = ndf.get("outputs") if isinstance(ndf, dict) else None
    pred = (outputs or {}).get("pred") if isinstance(outputs, dict) else None
    cd = (pred or {}).get("channel_def") if isinstance(pred, dict) else None
    if not isinstance(cd, dict):
        return {}
    return {str(int(k)): str(v)[:200] for k, v in cd.items()
            if str(k).isdigit() and int(k) > 0}


def _labels_from_label_dict(ld: Any) -> Dict[str, str]:
    """NV-Segment-CT style label_dict.json: {"liver": 1, "spleen": 3}."""
    if not isinstance(ld, dict):
        return {}
    out = {}
    for name, idx in ld.items():
        if isinstance(idx, int) and not isinstance(idx, bool) and idx > 0:
            out[str(idx)] = str(name)[:200]
    return out


def _model_data_url(endpoint_name: str) -> str:
    sm = aws_service.sagemaker_client
    cfg_name = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointConfigName"]
    variants = sm.describe_endpoint_config(EndpointConfigName=cfg_name)["ProductionVariants"]
    model = sm.describe_model(ModelName=variants[0]["ModelName"])
    container = model.get("PrimaryContainer") or (model.get("Containers") or [{}])[0]
    url = container.get("ModelDataUrl", "")
    if not url:
        raise ValueError("Endpoint's model has no packaged ModelDataUrl (model.tar.gz)")
    return url


def _parse_s3(uri: str):
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI (expected s3://bucket/key): {uri}")
    return p.netloc, p.path.lstrip("/")


def _dicom_to_nifti_gz(raw: bytes) -> bytes:
    """Decode a single-file DICOM and re-encode as gzipped NIfTI.

    Browser-side DICOM decoding (daikon) chokes on some JPEG2000 codestream
    options; pydicom + Pillow (OpenJPEG) handle them, so we convert on the
    server and let NiiVue parse plain NIfTI.
    """
    import gzip
    import io

    import nibabel as nib
    import numpy as np
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(raw))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = float(arr.max()) - arr  # inverted grayscale → normal polarity
    if arr.ndim == 2:
        # rows/cols → NIfTI x (right) / y (up): flip rows, then transpose
        arr = np.flip(arr, axis=0).T
        arr = arr[:, :, np.newaxis]  # single-slice 3D volume
    spacing = getattr(ds, "PixelSpacing", None) or [1.0, 1.0]
    affine = np.diag([float(spacing[1]), float(spacing[0]), 1.0, 1.0])
    nii = nib.Nifti1Image(arr, affine)
    return gzip.compress(nii.to_bytes(), compresslevel=1)


@router.get("/presign")
async def presign(s3_uri: str):
    """Short-lived presigned GET URL for a NIfTI object, for NiiVue to load."""
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        bucket, key = _parse_s3(s3_uri)
        if not key.lower().endswith((".nii", ".nii.gz")):
            raise ValueError("Viewer presign is limited to NIfTI objects (.nii/.nii.gz)")
        url = aws_service.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=PRESIGN_EXPIRY_SECONDS,
        )
        return {"url": url, "expires_in": PRESIGN_EXPIRY_SECONDS}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Presign failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fetch")
def fetch(s3_uri: str):
    """Stream a NIfTI or single-file DICOM object through the backend.

    Same-origin proxy for the NiiVue viewer — avoids requiring CORS
    configuration on the source S3 bucket (browsers block cross-origin
    fetches of presigned URLs unless the bucket allows the origin).
    Sync endpoint: FastAPI runs it in the threadpool so the blocking
    boto3 read doesn't stall the event loop.
    """
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        bucket, key = _parse_s3(s3_uri)
        if not key.lower().endswith((".nii", ".nii.gz", ".dcm")):
            raise ValueError(
                "Viewer fetch is limited to NIfTI/DICOM objects (.nii/.nii.gz/.dcm)")
        obj = aws_service.s3_client.get_object(Bucket=bucket, Key=key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "NoSuchBucket", "404"):
            raise HTTPException(status_code=404, detail=f"Object not found: {s3_uri}")
        if code in ("AccessDenied", "ExpiredToken", "InvalidAccessKeyId"):
            raise HTTPException(
                status_code=403,
                detail=f"S3 access failed ({code}) — check AWS credentials/permissions")
        logger.error("Viewer fetch failed for %s: %s", s3_uri, e)
        raise HTTPException(status_code=500, detail=str(e))

    filename = key.rsplit("/", 1)[-1]
    lk = key.lower()

    if lk.endswith(".dcm"):
        # Convert server-side: browsers get gzipped NIfTI they can always parse.
        from fastapi.responses import Response
        try:
            nii_gz = _dicom_to_nifti_gz(obj["Body"].read())
        except Exception as e:
            # FastAPI does not log HTTPException, so this is the only server-side record.
            logger.error("DICOM→NIfTI conversion failed for %s: %s", s3_uri, e)  # nosemgrep: python.lang.best-practice.logging-error-without-handling.logging-error-without-handling
            raise HTTPException(
                status_code=422,
                detail=f"Could not decode DICOM {filename}: {e}")
        return Response(
            content=nii_gz,
            media_type="application/gzip",
            headers={
                "Content-Disposition": f'inline; filename="{filename}.nii.gz"',
                "Cache-Control": "private, max-age=300",
            },
        )

    headers = {
        "Content-Length": str(obj["ContentLength"]),
        "Content-Disposition": f'inline; filename="{filename}"',
        "Cache-Control": "private, max-age=300",
    }
    body = obj["Body"]  # botocore StreamingBody
    media_type = "application/gzip" if lk.endswith(".gz") else "application/octet-stream"
    return StreamingResponse(
        body.iter_chunks(chunk_size=1024 * 256),
        media_type=media_type,
        headers=headers,
    )


@router.get("/recent-masks")
async def recent_masks(limit: int = 25):
    """Most recent prediction masks from the known output prefixes."""
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        bucket = os.environ.get("VIEWER_MASKS_BUCKET", DEFAULT_BUCKET)
        if not bucket:
            return {"masks": [],
                    "note": "Set VIEWER_MASKS_BUCKET in .env.local to list recent prediction masks."}
        items = []
        for prefix in MASK_PREFIXES:
            resp = aws_service.s3_client.list_objects_v2(
                Bucket=bucket, Prefix=prefix, MaxKeys=200)
            for obj in resp.get("Contents", []):
                if obj["Key"].lower().endswith((".nii", ".nii.gz")):
                    items.append({
                        "s3_uri": f"s3://{bucket}/{obj['Key']}",
                        "name": obj["Key"].rsplit("/", 1)[-1],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                    })
        items.sort(key=lambda x: x["last_modified"], reverse=True)
        return {"masks": items[:limit]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "AccessDenied":
            raise HTTPException(
                status_code=403,
                detail=f"Access denied listing s3://{bucket} — check VIEWER_MASKS_BUCKET and IAM permissions")
        logger.error("recent-masks failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("recent-masks failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _endpoint_for_mask(mask_s3_uri: str) -> str:
    """Endpoint whose async output location contains this mask (longest match).
    Uses the dashboard's tracked deployments — no AWS calls."""
    from app.routers.deployment_router import deployment_service

    bucket, key = _parse_s3(mask_s3_uri)
    target = f"s3://{bucket}/{key}"
    best, best_len = "", 0
    for rec in deployment_service.list_deployments():
        out = (rec.get("output_s3_uri") or "").rstrip("/")
        name = rec.get("endpoint_name") or ""
        if out and name and target.startswith(out + "/") and len(out) > best_len:
            best, best_len = name, len(out)
    return best


@router.get("/labels")
async def segmentation_labels(endpoint_name: str = "", mask_s3_uri: str = ""):
    """Label names for a segmentation endpoint's mask values, for the viewer legend.

    Read from the bundle files inside the endpoint's deployed model.tar.gz
    (configs/metadata.json channel_def, or configs/label_dict.json) — so no
    image rebuild or redeploy is needed. Returns {"labels": {"1": "spleen", ...}};
    an empty map means the viewer falls back to "Label N". When endpoint_name is
    omitted, it is resolved from mask_s3_uri via the endpoints' output locations.
    """
    import json

    if not endpoint_name and mask_s3_uri:
        try:
            endpoint_name = _endpoint_for_mask(mask_s3_uri)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not endpoint_name:
            return {"endpoint_name": "", "labels": {},
                    "note": "No tracked endpoint writes to this mask's location — enter the endpoint name."}
    if not _ENDPOINT_NAME_RE.match(endpoint_name or ""):
        raise HTTPException(status_code=400, detail="Invalid endpoint name")
    try:
        if not aws_service.sagemaker_client or not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        model_url = _model_data_url(endpoint_name)
        if model_url in _label_cache:
            return _label_cache[model_url]
        bucket, key = _parse_s3(model_url)
        if not key.endswith((".tar.gz", ".tgz", ".tar")):
            raise ValueError("Model artifact is not a tar archive")

        body = aws_service.s3_client.get_object(Bucket=bucket, Key=key)["Body"]
        labels: Dict[str, str] = {}
        source = ""
        scanned = 0
        try:
            # Streaming mode: reads sequentially, never seeks or extracts to disk.
            with tarfile.open(fileobj=body, mode="r|*") as tar:
                for member in tar:
                    scanned += member.size
                    name = member.name.replace("\\", "/")
                    is_meta = name.endswith("configs/metadata.json")
                    is_dict = name.endswith("configs/label_dict.json")
                    if (is_meta or is_dict) and member.isfile() \
                            and member.size <= _LABEL_FILE_MAX_BYTES:
                        f = tar.extractfile(member)
                        data = json.loads(f.read().decode("utf-8")) if f else None
                        found = (_labels_from_metadata(data) if is_meta
                                 else _labels_from_label_dict(data))
                        # metadata channel_def wins; label_dict fills in when absent
                        if found and (is_meta or not labels):
                            labels, source = found, name
                        if is_meta and found:
                            break
                    if scanned > _LABEL_SCAN_MAX_BYTES:
                        break
        finally:
            body.close()

        resp: Dict[str, Any] = {"endpoint_name": endpoint_name, "labels": labels}
        if source:
            resp["source"] = source
        else:
            resp["note"] = "No label names found in the model's bundle configs."
        _label_cache[model_url] = resp
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("ValidationException", "ResourceNotFound"):
            raise HTTPException(status_code=404, detail=f"Endpoint '{endpoint_name}' not found")
        if code == "AccessDenied":
            raise HTTPException(status_code=403, detail="Access denied reading the endpoint's model artifact")
        logger.error("labels lookup failed: %s", e)
        raise HTTPException(status_code=500, detail="Label lookup failed")
    except (tarfile.TarError, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("labels: unreadable model artifact for %s: %s", endpoint_name, e)
        return {"endpoint_name": endpoint_name, "labels": {},
                "note": "Model artifact could not be read for label names."}
    except Exception as e:
        logger.error("labels lookup failed: %s", e)
        raise HTTPException(status_code=500, detail="Label lookup failed")
