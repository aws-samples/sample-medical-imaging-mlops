from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.routers.aws_router import aws_service
from app.services.deployment_service import DeploymentService

logger = logging.getLogger(__name__)
router = APIRouter()

deployment_service = DeploymentService()


class EndpointDeployRequest(BaseModel):
    endpoint_name:        str
    model_artifacts:      Optional[str] = None
    image_uri:            Optional[str] = None
    role_arn:             Optional[str] = None
    instance_type:        str = "ml.m5.xlarge"
    instance_count:       int = 1
    inference_script:     str = "inference.py"
    environment:          dict = {}
    async_endpoint:       bool = False
    output_s3_uri:        Optional[str] = None
    max_concurrent:       int = 10
    output_pattern:       Optional[str] = None
    # Data capture
    enable_data_capture:  bool = False
    capture_s3_uri:       Optional[str] = None
    capture_sampling_pct: int = 100


@router.post("/deploy")
async def deploy_model(request: EndpointDeployRequest):
    """Create a SageMaker endpoint, or import/track an existing one by name only."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")

        # No model_artifacts → just register the existing endpoint for tracking
        if not request.model_artifacts:
            return deployment_service.import_endpoint(
                endpoint_name=request.endpoint_name,
                sagemaker_client=aws_service.sagemaker_client,
            )

        if not request.image_uri or not request.role_arn:
            raise ValueError("image_uri and role_arn are required when deploying a new endpoint")

        sagemaker_session = aws_service._get_sagemaker_session()
        result = deployment_service.create_endpoint(
            config=request.dict(),
            sagemaker_session=sagemaker_session,
            role_arn=request.role_arn,
        )
        return result
    except Exception as e:
        logger.warning("Deployment failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments")
async def list_deployments(local_only: bool = False):
    """List all deployments. If local_only=true, skip AWS status refresh."""
    deployments = deployment_service.list_deployments()
    if not local_only and aws_service.sagemaker_client:
        for d in deployments:
            if d.get("status") in {"Deleted"}:
                continue
            try:
                updated = deployment_service.get_endpoint_status(
                    d["endpoint_name"], aws_service.sagemaker_client
                )
                d.update(updated)
            except Exception:
                pass
        # Re-fetch after refresh — pruned deleted entries won't appear
        deployments = deployment_service.list_deployments()
    return {"deployments": [d for d in deployments if d.get("status") != "Deleted"]}


@router.get("/deployments/{endpoint_name}/status")
async def get_deployment_status(endpoint_name: str):
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        return deployment_service.get_endpoint_status(endpoint_name, aws_service.sagemaker_client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/deployments/{endpoint_name}")
async def delete_deployment(endpoint_name: str):
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        deployment_service.delete_endpoint(endpoint_name, aws_service.sagemaker_client)
        return {"endpoint_name": endpoint_name, "status": "Deleting"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AsyncInvokeRequest(BaseModel):
    endpoint_name: str
    input_s3_uri:  str
    output_s3_uri: str


class ScaleRequest(BaseModel):
    desired_count: int


@router.post("/deployments/{endpoint_name}/scale")
async def scale_endpoint(endpoint_name: str, request: ScaleRequest):
    """Scale an endpoint up or down (including to 0)."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        result = deployment_service.scale_endpoint(
            endpoint_name=endpoint_name,
            desired_count=request.desired_count,
            sagemaker_client=aws_service.sagemaker_client,
        )
        return result
    except Exception as e:
        logger.error("Scale failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invoke-async")
async def invoke_async(request: AsyncInvokeRequest):
    """Submit an async inference request to a SageMaker async endpoint."""
    try:
        if not aws_service._session:
            raise ValueError("AWS credentials not configured")
        result = deployment_service.invoke_async(
            endpoint_name=request.endpoint_name,
            input_s3_uri=request.input_s3_uri,
            output_s3_uri=request.output_s3_uri,
            boto_session=aws_service._session,
        )
        return result
    except Exception as e:
        logger.error("Async invocation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
async def sync_deployments_from_aws(name_contains: str = None):
    """Pull endpoints from AWS and merge into the local store."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        result = deployment_service.sync_from_aws(aws_service.sagemaker_client, name_contains=name_contains)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to sync deployments from AWS: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
