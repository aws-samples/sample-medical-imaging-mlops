"""NVIDIA MedTech (NV-Segment-CT) API routes. RESEARCH USE ONLY."""
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from app.routers.aws_router import aws_service
from app.services.nvmedtech_service import nvmedtech_service, DEFAULT_TRAIN_CONFIG

logger = logging.getLogger(__name__)
router = APIRouter()


def _region() -> str:
    return os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _image(env_key: str, repo: str) -> str:
    uri = os.environ.get(env_key)
    if uri:
        return uri
    account = os.environ.get("AWS_ACCOUNT_ID", "")
    if not account:
        raise ValueError(f"{env_key} not set and AWS_ACCOUNT_ID missing")
    return f"{account}.dkr.ecr.{_region()}.amazonaws.com/{repo}:latest"


@router.get("/bundle")
async def bundle_card():
    """The NV-Segment-CT bundle card (papers/data org from its metadata.json)."""
    return nvmedtech_service.bundle_card()


class StageRequest(BaseModel):
    s3_uri: str


@router.post("/stage")
async def stage(request: StageRequest):
    """Package the local bundle + HF checkpoint + handler and upload to S3."""
    try:
        return nvmedtech_service.stage(request.s3_uri, _region())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("NV-Segment-CT staging failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class FinetuneRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    job_name: Optional[str] = None
    model_artifacts: str
    data_s3_uri: str
    role_arn: Optional[str] = None
    image_uri: Optional[str] = None
    instance_type: str = "ml.g5.4xlarge"
    instance_count: int = 1
    volume_size_gb: int = 100
    max_runtime_seconds: int = 3600
    train_config: Optional[str] = None      # default: train.json,train_continual.json
    generate_datalist: bool = False         # bundle ships its own datalists
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    extra_overrides: dict = {}
    output_s3_uri: Optional[str] = None


@router.post("/finetune")
async def finetune(request: FinetuneRequest):
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        role_arn = request.role_arn or os.environ.get("SAGEMAKER_ROLE_ARN")
        if not role_arn:
            raise ValueError("role_arn is required (set SAGEMAKER_ROLE_ARN or pass role_arn)")
        overrides = dict(request.extra_overrides or {})
        if request.epochs is not None:
            overrides["epochs"] = request.epochs
        if request.learning_rate is not None:
            overrides["learning_rate"] = request.learning_rate
        return nvmedtech_service.start_finetune(
            job_name=request.job_name,
            model_artifacts=request.model_artifacts,
            data_s3_uri=request.data_s3_uri,
            role_arn=role_arn,
            image_uri=request.image_uri or _image("NVMEDTECH_IMAGE_URI", "sm-nvmedtech"),
            sagemaker_client=aws_service.sagemaker_client,
            instance_type=request.instance_type,
            instance_count=request.instance_count,
            volume_size_gb=request.volume_size_gb,
            max_runtime_seconds=request.max_runtime_seconds,
            train_config=request.train_config or DEFAULT_TRAIN_CONFIG,
            config_overrides=overrides,
            generate_datalist=request.generate_datalist,
            output_s3_uri=request.output_s3_uri,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("NV-Segment-CT fine-tune failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_jobs():
    return {"jobs": nvmedtech_service.refresh_jobs(aws_service.sagemaker_client)}


@router.get("/artifacts")
async def deployable_artifacts():
    nvmedtech_service.refresh_jobs(aws_service.sagemaker_client)
    return nvmedtech_service.list_deployable_artifacts()


class DeployRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    endpoint_name: str
    model_artifacts: str
    output_s3_uri: str
    image_uri: Optional[str] = None
    role_arn: Optional[str] = None
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    max_concurrent: int = 4


@router.post("/deploy")
async def deploy(request: DeployRequest):
    """Deploy NV-Segment-CT (base or fine-tuned) to an ASYNC endpoint."""
    try:
        if not aws_service.sagemaker_session:
            raise ValueError("AWS credentials not configured")
        role_arn = request.role_arn or os.environ.get("SAGEMAKER_ROLE_ARN")
        if not role_arn:
            raise ValueError("role_arn is required (set SAGEMAKER_ROLE_ARN or pass role_arn)")
        return nvmedtech_service.deploy(
            endpoint_name=request.endpoint_name,
            model_artifacts=request.model_artifacts,
            image_uri=request.image_uri or _image("NVMEDTECH_INFERENCE_IMAGE_URI", "sm-nvmedtech-inference"),
            role_arn=role_arn,
            sagemaker_session=aws_service.sagemaker_session,
            output_s3_uri=request.output_s3_uri,
            instance_type=request.instance_type,
            instance_count=request.instance_count,
            max_concurrent=request.max_concurrent,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("NV-Segment-CT deploy failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
