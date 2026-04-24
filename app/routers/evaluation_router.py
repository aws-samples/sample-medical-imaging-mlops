"""Router for classification endpoint evaluation via SageMaker Processing Jobs."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
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


@router.get("/jobs/{job_name}/plots")
async def get_evaluation_plots(job_name: str):
    """Return presigned S3 URLs for the plot files of a completed evaluation job."""
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        evaluation_service._load()   # re-sync from disk
        if job_name not in evaluation_service.jobs:
            raise ValueError(f"Job {job_name!r} not found")

        record     = evaluation_service.jobs[job_name]
        output_uri = record.get("output_s3_uri", "")
        task_type  = record.get("task_type", "classification")

        if not output_uri:
            raise ValueError("No output S3 URI on this job record")

        bucket, prefix = output_uri.replace("s3://", "").split("/", 1)
        prefix = prefix.rstrip("/")

        plot_files = (
            ["confusion_matrix.png", "roc_auc.png"]
            if task_type == "classification"
            else ["dice_distribution.png"]
        )

        urls = {}
        for fname in plot_files:
            key = f"{prefix}/{fname}"
            url = aws_service.generate_presigned_url(bucket, key)
            if url:
                urls[fname] = url

        return {"plots": urls, "task_type": task_type, "output_s3_uri": output_uri}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to generate plot URLs: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_name}")
async def get_evaluation_job(job_name: str):
    """Get status and metrics for a specific evaluation job."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        return evaluation_service.get_status(job_name, aws_service.sagemaker_client)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
