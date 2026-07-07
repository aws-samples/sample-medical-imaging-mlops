"""Router for classification endpoint evaluation via SageMaker Processing Jobs."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import logging

from app.routers.aws_router import aws_service
from app.services.evaluation_service import EvaluationService

logger = logging.getLogger(__name__)
router = APIRouter()

evaluation_service = EvaluationService()


class EvaluationRequest(BaseModel):
    endpoint_name: str
    data_s3_uri:   str          # s3://bucket/test-data/  (class subfolders inside)
    output_s3_uri: str          # s3://bucket/eval-results/job-name
    image_uri:     str          # ECR URI of the inference container
    role_arn:      str
    instance_type: str = "ml.m5.xlarge"
    task_type:     str = "classification"   # classification | segmentation


@router.post("/submit")
async def submit_evaluation(request: EvaluationRequest):
    """Submit a SageMaker Processing Job to evaluate a classification endpoint."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        region = aws_service._session.region_name or "us-east-1"
        record = evaluation_service.submit_evaluation(
            endpoint_name=request.endpoint_name,
            data_s3_uri=request.data_s3_uri,
            output_s3_uri=request.output_s3_uri,
            image_uri=request.image_uri,
            role_arn=request.role_arn,
            instance_type=request.instance_type,
            sagemaker_client=aws_service.sagemaker_client,
            region=region,
            task_type=request.task_type,
        )
        return record
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to submit evaluation job: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_evaluation_jobs():
    """List all evaluation jobs, refreshing status for in-progress ones."""
    try:
        client = aws_service.sagemaker_client if aws_service.sagemaker_client else None
        return {"jobs": evaluation_service.list_jobs(sagemaker_client=client)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_name}/subgroup-metrics")
async def get_subgroup_metrics(job_name: str):
    """Fetch and return subgroup_metrics.json from the job's S3 output."""
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        evaluation_service._load()
        if job_name not in evaluation_service.jobs:
            raise ValueError(f"Job {job_name!r} not found")

        record     = evaluation_service.jobs[job_name]
        output_uri = record.get("output_s3_uri", "")
        if not output_uri:
            raise ValueError("No output S3 URI on this job record")

        bucket, prefix = output_uri.replace("s3://", "").split("/", 1)
        key = prefix.rstrip("/") + "/subgroup_metrics.json"

        obj  = aws_service.s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        from botocore.exceptions import ClientError
        if isinstance(e, ClientError) and e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise HTTPException(status_code=404,
                detail="subgroup_metrics.json not found — job may not have DICOM files with demographic tags")
        logger.error("Failed to fetch subgroup metrics: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_name}")
async def get_evaluation_job(job_name: str):
    """Get status and metrics for a specific evaluation job."""
    try:
        evaluation_service._load()
        if job_name not in evaluation_service.jobs:
            raise ValueError(f"Job {job_name!r} not found")
        record = evaluation_service.jobs[job_name]
        if record.get("status") in ("Completed", "Failed", "Stopped"):
            return record
        if aws_service.sagemaker_client:
            return evaluation_service.get_status(job_name, aws_service.sagemaker_client)
        return record
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_name}")
async def delete_evaluation_job(job_name: str):
    """Remove a job record and its local cache. Does not delete S3 outputs."""
    try:
        return evaluation_service.delete_job(job_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_name}/cache")
async def clear_evaluation_cache(job_name: str):
    """Delete locally cached result files for a job."""
    try:
        return evaluation_service.clear_cache(job_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_name}/download-results")
async def download_evaluation_results(job_name: str, plots_only: bool = False):
    """Download output files for a job into the local cache.
    plots_only=true downloads only PNGs (fast); omit for full download."""
    import asyncio
    from functools import partial
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        loop   = asyncio.get_event_loop()
        cached = await loop.run_in_executor(
            None,
            partial(evaluation_service.download_results,
                    job_name, aws_service.s3_client, plots_only),
        )
        return {"cached": cached, "job_name": job_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to download results: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_name}/plots")
async def get_evaluation_plots(job_name: str):
    """Return local cache URLs for plot files. Returns empty if not yet downloaded."""
    try:
        evaluation_service._load()
        if job_name not in evaluation_service.jobs:
            raise ValueError(f"Job {job_name!r} not found")
        record    = evaluation_service.jobs[job_name]
        task_type = record.get("task_type", "classification")
        plot_files = (
            ["confusion_matrix.png", "roc_auc.png", "subgroup_accuracy.png"]
            if task_type == "classification"
            else ["dice_distribution.png"]
        )
        cached = record.get("cached_results", {})
        urls   = {f: cached[f] for f in plot_files if f in cached}
        return {"plots": urls, "task_type": task_type,
                "output_s3_uri": record.get("output_s3_uri"), "source": "cache"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to get plot URLs: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
async def sync_evaluation_jobs_from_aws(name_contains: str = None):
    """Pull evaluation processing jobs from AWS and merge into the local store."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        result = evaluation_service.sync_from_aws(aws_service.sagemaker_client, name_contains=name_contains)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to sync evaluation jobs from AWS: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
