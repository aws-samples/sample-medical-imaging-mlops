"""
MONAI Bundle router — stage a bundle to S3, fine-tune it on SageMaker, track
jobs, and deploy the resulting artifact.

RESEARCH USE ONLY — outputs are AI predictions for research/development, not
clinical diagnosis.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional
import logging
import os

from app.routers.aws_router import aws_service
from app.services.monai_bundle_service import MonaiBundleService, list_supported_bundles

logger = logging.getLogger(__name__)
router = APIRouter()

monai_bundle_service = MonaiBundleService()


@router.get("/supported")
async def supported_bundles():
    """List MONAI bundles supported by this integration (the 'MONAI Model Bundle' category)."""
    return {"bundles": list_supported_bundles()}


class StageBundleRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    repo_id: str = "https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation"
    s3_uri: str                       # destination model.tar.gz


class FinetuneRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    job_name: Optional[str] = None
    model_artifacts: str              # bundle model.tar.gz (from /stage)
    data_s3_uri: str                  # dataset (Decathlon)
    role_arn: Optional[str] = None
    image_uri: Optional[str] = None   # sm-monai-bundle image; falls back to env/config
    instance_type: str = "ml.g5.4xlarge"
    instance_count: int = 1
    volume_size_gb: int = 100
    max_runtime_seconds: int = 3600
    train_config: Optional[str] = None
    # ── user-input hyperparameters (only sent values are applied) ──
    epochs: Optional[int] = None
    cache_rate: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    # arbitrary extra bundle config keys (merged last)
    extra_overrides: dict = {}
    output_s3_uri: Optional[str] = None


@router.post("/stage")
async def stage_bundle(request: StageBundleRequest):
    """Download a MONAI Model Zoo bundle, package, and upload to S3."""
    try:
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        return monai_bundle_service.stage_bundle(request.repo_id, request.s3_uri, region)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Bundle staging failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finetune")
async def finetune_bundle(request: FinetuneRequest):
    """Launch a MONAI bundle fine-tuning job on SageMaker with user hyperparameters."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        role_arn = request.role_arn or os.environ.get("SAGEMAKER_ROLE_ARN")
        if not role_arn:
            raise ValueError("role_arn is required (set SAGEMAKER_ROLE_ARN or pass role_arn)")

        image_uri = request.image_uri or os.environ.get("MONAI_BUNDLE_IMAGE_URI")
        if not image_uri:
            account = os.environ.get("AWS_ACCOUNT_ID", "")
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            if account:
                image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/sm-monai-bundle:latest"
        if not image_uri:
            raise ValueError("image_uri required (set MONAI_BUNDLE_IMAGE_URI / AWS_ACCOUNT_ID or pass image_uri)")

        # Assemble config_overrides from the discrete hyperparameter fields
        # (only include the ones the user set) plus any extra keys.
        overrides = {}
        if request.epochs is not None:
            overrides["epochs"] = request.epochs
        if request.cache_rate is not None:
            overrides["cache_rate"] = request.cache_rate
        if request.learning_rate is not None:
            overrides["learning_rate"] = request.learning_rate
        if request.batch_size is not None:
            overrides["batch_size"] = request.batch_size
        overrides.update(request.extra_overrides or {})

        return monai_bundle_service.start_finetune(
            job_name=request.job_name,
            model_artifacts=request.model_artifacts,
            data_s3_uri=request.data_s3_uri,
            role_arn=role_arn,
            image_uri=image_uri,
            sagemaker_client=aws_service.sagemaker_client,
            region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            instance_type=request.instance_type,
            instance_count=request.instance_count,
            volume_size_gb=request.volume_size_gb,
            max_runtime_seconds=request.max_runtime_seconds,
            train_config=request.train_config,
            config_overrides=overrides,
            output_s3_uri=request.output_s3_uri,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Bundle fine-tune failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/artifacts")
async def deployable_artifacts():
    """List deployable artifacts (base staged bundles + fine-tuned jobs) for the deploy page."""
    # Refresh non-terminal jobs so freshly-completed fine-tunes appear as deployable.
    monai_bundle_service.refresh_jobs(aws_service.sagemaker_client)
    return monai_bundle_service.list_deployable_artifacts()


@router.get("/jobs")
async def list_jobs():
    """List tracked MONAI bundle jobs, refreshing non-terminal statuses from AWS."""
    return {"jobs": monai_bundle_service.refresh_jobs(aws_service.sagemaker_client)}


@router.delete("/staged/{name}")
async def delete_staged(name: str, delete_artifact: bool = True):
    """Remove a staged bundle (record + optionally its S3 model.tar.gz)."""
    try:
        return monai_bundle_service.delete_staged(
            name, delete_artifact=delete_artifact,
            region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Delete staged failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_name}")
async def get_job(job_name: str):
    """Get status of a MONAI bundle job."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        return monai_bundle_service.get_job_status(job_name, aws_service.sagemaker_client)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeployRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    endpoint_name: str
    model_artifacts: str              # trained bundle model.tar.gz (from a completed job)
    output_s3_uri: str                # REQUIRED — async endpoints write results here
    image_uri: Optional[str] = None   # sm-monai-bundle-inference image; falls back to env/config
    role_arn: Optional[str] = None
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    max_concurrent: int = 4


@router.post("/deploy")
async def deploy_bundle(request: DeployRequest):
    """Deploy a trained MONAI bundle artifact to a SageMaker ASYNC endpoint."""
    try:
        if not aws_service._session:
            raise ValueError("AWS credentials not configured")
        role_arn = request.role_arn or os.environ.get("SAGEMAKER_ROLE_ARN")
        if not role_arn:
            raise ValueError("role_arn is required (set SAGEMAKER_ROLE_ARN or pass role_arn)")

        image_uri = request.image_uri or os.environ.get("MONAI_BUNDLE_INFERENCE_IMAGE_URI")
        if not image_uri:
            account = os.environ.get("AWS_ACCOUNT_ID", "")
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            if account:
                image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/sm-monai-bundle-inference:latest"
        if not image_uri:
            raise ValueError("image_uri required (set MONAI_BUNDLE_INFERENCE_IMAGE_URI / AWS_ACCOUNT_ID or pass image_uri)")

        return monai_bundle_service.deploy_bundle(
            endpoint_name=request.endpoint_name,
            model_artifacts=request.model_artifacts,
            image_uri=image_uri,
            role_arn=role_arn,
            sagemaker_session=aws_service._get_sagemaker_session(),
            output_s3_uri=request.output_s3_uri,
            instance_type=request.instance_type,
            instance_count=request.instance_count,
            max_concurrent=request.max_concurrent,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Bundle deploy failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
