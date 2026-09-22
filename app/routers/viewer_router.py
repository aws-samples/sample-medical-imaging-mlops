"""Viewer API — presigned access to NIfTI volumes/masks for the NiiVue viewer.

RESEARCH USE ONLY — rendered predictions are not clinical diagnoses.
"""
import logging
import os
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
            logger.error("DICOM→NIfTI conversion failed for %s: %s", s3_uri, e)
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
