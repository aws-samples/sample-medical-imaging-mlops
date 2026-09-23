"""Router for the preprocessing pipeline: step catalog + local single-image preview.

The batch (SageMaker) submit/jobs endpoints are part of the spec
(specs/preprocessing/) but not implemented yet; they return HTTP 501 so the UI
can degrade gracefully instead of 404-ing.

RESEARCH USE ONLY.
"""
import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException

from app.models.preprocessing import PreviewRequest, PipelineSpec, SubmitRequest
from src.preprocessing.pipeline_builder import get_catalog
from src.preprocessing.local_preview import preview as run_preview, PreviewError
from app.services.preprocessing_service import PreprocessingService

logger = logging.getLogger(__name__)
router = APIRouter()

preprocessing_service = PreprocessingService()


def _ensure_aws():
    """Return (sagemaker_client, s3_client, region), lazily configuring from
    environment credentials if the UI hasn't connected AWS yet."""
    from app.routers.aws_router import aws_service
    if not getattr(aws_service, "s3_client", None) or not getattr(aws_service, "sagemaker_client", None):
        try:
            aws_service.configure_from_environment()
        except Exception:
            pass
    if not getattr(aws_service, "s3_client", None) or not getattr(aws_service, "sagemaker_client", None):
        raise HTTPException(
            status_code=400,
            detail="AWS credentials not configured. Connect AWS on the AWS tab, "
                   "or configure environment credentials (~/.aws or env vars).",
        )
    region = getattr(aws_service._session, "region_name", None) or "us-east-1"
    return aws_service.sagemaker_client, aws_service.s3_client, region


@router.get("/catalog")
async def catalog():
    """Return the supported preprocessing steps with default params + docs."""
    return {"catalog": get_catalog()}


@router.post("/validate")
async def validate(spec: PipelineSpec):
    """Validate a user-supplied pipeline spec (e.g. an uploaded JSON file).

    Returns the normalized spec on success; FastAPI returns 422 with field-level
    errors if the spec is invalid. This lets the UI import a JSON spec safely.
    """
    return {"valid": True, "spec": spec.model_dump()}


@router.get("/resolve-dataset")
async def resolve_dataset(input_s3_uri: str, sample_n: int = 5):
    """Dry-run: read ONLY <input_s3_uri>/dataset.json and summarize it.

    Returns dataset name, train/val/test counts, label map, the derived task
    type (segmentation if entries carry a 'label', else classification), and the
    first N resolved (image, label) pairs — so the pairing can be confirmed
    before submitting a batch job. Reads one small object, no listing.
    """
    from app.routers.aws_router import aws_service
    from src.preprocessing.dataset_resolver import (
        read_dataset_json_from_s3, discover_imagefolder_s3,
        DatasetResolveError, DatasetJsonMissingError, example_dataset_json_text,
    )
    try:
        s3 = aws_service.s3_client if getattr(aws_service, "s3_client", None) else None
        if s3 is None:
            # Lazily fall back to ambient environment credentials (env vars,
            # ~/.aws, IAM role) for this read-only dataset.json fetch, so the user
            # doesn't have to visit the AWS tab first just to resolve a dataset.
            try:
                if aws_service.configure_from_environment():
                    s3 = aws_service.s3_client
            except Exception:
                s3 = None
        if s3 is None:
            raise HTTPException(
                status_code=400,
                detail="AWS credentials not configured. Connect AWS on the AWS tab, "
                       "or configure environment credentials (~/.aws or env vars).",
            )
        resolved = read_dataset_json_from_s3(input_s3_uri, s3_client=s3)
        return resolved.summary(sample_n=max(1, min(sample_n, 50)))
    except DatasetJsonMissingError:
        # No dataset.json. For classification, fall back to ImageFolder discovery
        # (class = subfolder). Segmentation still requires a dataset.json.
        try:
            resolved = discover_imagefolder_s3(input_s3_uri, s3_client=s3)
            return resolved.summary(sample_n=max(1, min(sample_n, 50)))
        except DatasetResolveError:
            pass  # neither a manifest nor class folders → actionable error below

        expected = input_s3_uri.rstrip("/") + "/dataset.json"
        raise HTTPException(status_code=404, detail={
            "error": "dataset_json_missing",
            "message": (
                "No dataset.json found, and no classification class-subfolders "
                "were detected either. For segmentation, create a dataset.json at "
                f"{expected} (MSD/Decathlon format). For classification, either add "
                "a dataset.json or organise images as <root>/<class-name>/<images>."
            ),
            "expected_location": expected,
            "example": example_dataset_json_text(),
        })
    except DatasetResolveError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("resolve-dataset failed: %s", e)
        raise HTTPException(status_code=500, detail="Could not resolve dataset")


@router.post("/preview")
async def preview(request: PreviewRequest):
    """Run the pipeline on a single image locally and return stats + a slice PNG.

    No S3 writes, no SageMaker job. Runs off the event loop since MONAI/IO is
    blocking.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(run_preview, request.spec, request.input_ref)
        )
        return result
    except PreviewError as e:
        # Already sanitized, client-safe.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Preview failed unexpectedly: %s", e)
        raise HTTPException(status_code=500, detail="Preview failed due to an internal error")


@router.post("/submit")
async def submit_batch(request: SubmitRequest):
    """Submit a SageMaker Processing Job that runs the pipeline over the dataset."""
    import asyncio
    from functools import partial
    try:
        sm, s3, region = _ensure_aws()
        loop = asyncio.get_event_loop()
        record = await loop.run_in_executor(None, partial(
            preprocessing_service.submit_batch,
            input_s3_uri=request.input_s3_uri,
            output_s3_uri=request.output_s3_uri,
            spec=request.spec.model_dump(),
            image_uri=request.image_uri,
            role_arn=request.role_arn,
            sagemaker_client=sm,
            s3_client=s3,
            instance_type=request.instance_type or "ml.m5.2xlarge",
            region=region,
        ))
        return record
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to submit preprocessing job: %s", e)
        raise HTTPException(status_code=500, detail="Failed to submit preprocessing job")


def _sagemaker_client_or_none():
    """Best-effort SageMaker client for status refresh — lazily configures from
    environment credentials, returns None if unavailable (never raises)."""
    from app.routers.aws_router import aws_service
    if not getattr(aws_service, "sagemaker_client", None):
        try:
            aws_service.configure_from_environment()
        except Exception:
            pass
    return getattr(aws_service, "sagemaker_client", None)


@router.get("/jobs")
async def list_jobs():
    """List all preprocessing jobs, refreshing status for in-progress ones.

    Lazily configures AWS from environment creds so a completed job's status is
    updated even if the user hasn't connected AWS on the AWS tab.
    """
    try:
        sm = _sagemaker_client_or_none()
        return {"jobs": preprocessing_service.list_jobs(sagemaker_client=sm)}
    except Exception as e:
        logger.error("Failed to list preprocessing jobs: %s", e)
        return {"jobs": []}


@router.get("/jobs/{job_name}")
async def get_job(job_name: str):
    try:
        preprocessing_service._load()
        if job_name not in preprocessing_service.jobs:
            raise ValueError(f"Job {job_name!r} not found")
        record = preprocessing_service.jobs[job_name]
        if record.get("status") in ("Completed", "Failed", "Stopped"):
            return record
        sm = _sagemaker_client_or_none()
        if sm:
            return preprocessing_service.get_status(job_name, sm)
        return record
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_name}")
async def delete_job(job_name: str):
    """Remove a job record. Does not delete S3 outputs."""
    try:
        return preprocessing_service.delete_job(job_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
async def sync_jobs(name_contains: str = None):
    """Pull preprocessing processing jobs from AWS into the local store."""
    try:
        sm, _s3, _region = _ensure_aws()
        return preprocessing_service.sync_from_aws(sm, name_contains=name_contains)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to sync preprocessing jobs: %s", e)
        raise HTTPException(status_code=500, detail="Failed to sync preprocessing jobs")
