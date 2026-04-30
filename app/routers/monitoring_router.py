"""Router for SageMaker Model Monitor — data capture, baselines, and monitoring schedules."""
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.routers.aws_router import aws_service
from app.services.monitoring_service import MonitoringService

logger = logging.getLogger(__name__)
router = APIRouter()

monitoring_service = MonitoringService()


# ── Request models ────────────────────────────────────────────────────────────

class DataCaptureRequest(BaseModel):
    endpoint_name:       str
    capture_s3_uri:      str          # s3://bucket/capture-data/
    sampling_percentage: int = 100    # 1-100


class DataQualityBaselineRequest(BaseModel):
    baseline_data_s3_uri: str         # s3://bucket/baseline-data/
    output_s3_uri:        str         # s3://bucket/baseline-output/
    role_arn:             str
    instance_type:        str = "ml.m5.xlarge"
    job_name:             Optional[str] = None


class DataQualityScheduleRequest(BaseModel):
    schedule_name:               str
    endpoint_name:               str
    baseline_constraints_s3_uri: str   # constraints.json from baseline output
    baseline_statistics_s3_uri:  str   # statistics.json from baseline output
    output_s3_uri:               str
    role_arn:                    str
    instance_type:               str = "ml.m5.xlarge"
    schedule_expression:         str = "cron(0 * ? * * *)"  # hourly


class ModelQualityScheduleRequest(BaseModel):
    schedule_name:          str
    endpoint_name:          str
    ground_truth_s3_uri:    str        # s3://bucket/ground-truth/
    output_s3_uri:          str
    role_arn:               str
    instance_type:          str = "ml.m5.xlarge"
    schedule_expression:    str = "cron(0 * ? * * *)"
    problem_type:           str = "MulticlassClassification"  # BinaryClassification | MulticlassClassification | Regression
    inference_attribute:    str = "predicted_label"
    probability_attribute:  Optional[str] = None
    ground_truth_attribute: str = "label"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/capture-summary")
async def get_capture_summary(
    endpoint_name: str,
    capture_s3_uri: str,
    max_files: int = 20,
):
    """Read latest capture files from S3 and return aggregated prediction stats."""
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        result = monitoring_service.get_capture_summary(
            capture_s3_uri=capture_s3_uri,
            endpoint_name=endpoint_name,
            s3_client=aws_service.s3_client,
            max_files=max_files,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to get capture summary: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-capture")
async def enable_data_capture(request: DataCaptureRequest):
    """Enable data capture on an existing SageMaker endpoint."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        result = monitoring_service.enable_data_capture(
            endpoint_name=request.endpoint_name,
            capture_s3_uri=request.capture_s3_uri,
            sampling_percentage=request.sampling_percentage,
            sagemaker_client=aws_service.sagemaker_client,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to enable data capture: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/baseline/data-quality")
async def create_data_quality_baseline(request: DataQualityBaselineRequest):
    """Submit a data quality baseline job."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        job_name = request.job_name or f"dq-baseline-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        result = monitoring_service.create_data_quality_baseline(
            job_name=job_name,
            baseline_data_s3_uri=request.baseline_data_s3_uri,
            output_s3_uri=request.output_s3_uri,
            role_arn=request.role_arn,
            instance_type=request.instance_type,
            sagemaker_client=aws_service.sagemaker_client,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create baseline: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedules/data-quality")
async def create_data_quality_schedule(request: DataQualityScheduleRequest):
    """Create a data quality monitoring schedule."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        result = monitoring_service.create_data_quality_schedule(
            schedule_name=request.schedule_name,
            endpoint_name=request.endpoint_name,
            baseline_constraints_s3_uri=request.baseline_constraints_s3_uri,
            baseline_statistics_s3_uri=request.baseline_statistics_s3_uri,
            output_s3_uri=request.output_s3_uri,
            role_arn=request.role_arn,
            instance_type=request.instance_type,
            schedule_expression=request.schedule_expression,
            sagemaker_client=aws_service.sagemaker_client,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create data quality schedule: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedules/model-quality")
async def create_model_quality_schedule(request: ModelQualityScheduleRequest):
    """Create a model quality monitoring schedule."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        result = monitoring_service.create_model_quality_schedule(
            schedule_name=request.schedule_name,
            endpoint_name=request.endpoint_name,
            ground_truth_s3_uri=request.ground_truth_s3_uri,
            output_s3_uri=request.output_s3_uri,
            role_arn=request.role_arn,
            instance_type=request.instance_type,
            schedule_expression=request.schedule_expression,
            problem_type=request.problem_type,
            inference_attribute=request.inference_attribute,
            probability_attribute=request.probability_attribute,
            ground_truth_attribute=request.ground_truth_attribute,
            sagemaker_client=aws_service.sagemaker_client,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create model quality schedule: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedules")
async def list_schedules(local_only: bool = False):
    """List all monitoring schedules."""
    try:
        client = None if local_only else aws_service.sagemaker_client
        return {"schedules": monitoring_service.list_schedules(sagemaker_client=client)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/baselines")
async def list_baselines(local_only: bool = False):
    """List all baseline jobs."""
    try:
        monitoring_service._load()
        baselines = [r for r in monitoring_service.schedules.values() if r.get("type") == "baseline"]
        if not local_only and aws_service.sagemaker_client:
            for b in baselines:
                job_name = b.get("job_name")
                if job_name and b.get("status") in ("InProgress", "Stopping"):
                    try:
                        resp = aws_service.sagemaker_client.describe_processing_job(
                            ProcessingJobName=job_name
                        )
                        b["status"] = resp["ProcessingJobStatus"]
                        if resp.get("ProcessingEndTime"):
                            b["completed_at"] = resp["ProcessingEndTime"].isoformat()
                        if resp.get("FailureReason"):
                            b["failure_reason"] = resp["FailureReason"]
                        monitoring_service.schedules[job_name] = b
                    except Exception as e:
                        logger.warning("Could not refresh baseline %s: %s", job_name, e)
            monitoring_service._save()
        return {"baselines": sorted(baselines, key=lambda r: r.get("created_at", ""), reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/baselines/{job_name}")
async def delete_baseline(job_name: str):
    """Remove a baseline job record from local tracking."""
    monitoring_service._load()
    if job_name not in monitoring_service.schedules:
        raise HTTPException(status_code=404, detail=f"Baseline {job_name!r} not found")
    del monitoring_service.schedules[job_name]
    monitoring_service._save()
    return {"deleted": job_name}


@router.get("/schedules/{schedule_name}")
async def get_schedule(schedule_name: str):
    """Get status and latest execution for a monitoring schedule."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        return monitoring_service.refresh_schedule(schedule_name, aws_service.sagemaker_client)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedules/{schedule_name}")
async def delete_schedule(schedule_name: str):
    """Stop and delete a monitoring schedule."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        return monitoring_service.delete_schedule(schedule_name, aws_service.sagemaker_client)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
