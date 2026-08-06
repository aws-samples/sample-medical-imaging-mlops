from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import logging

from app.routers.aws_router import aws_service
from app.services.inference_service import InferenceService

logger = logging.getLogger(__name__)
router = APIRouter()
inference_service = InferenceService()


class InferenceRequest(BaseModel):
    endpoint_name: str
    task_type:     Optional[str] = "classification"   # classification | segmentation
    input_type:    str = "s3_uri"           # s3_uri | file_path
    input_value:   str                      # S3 URI or local path
    output_s3_uri: Optional[str] = None     # where to save result
    output_pattern: Optional[str] = None    # filename pattern for segmentation masks


@router.post("/invoke")
async def invoke_endpoint(request: InferenceRequest):
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        session = aws_service._get_boto_session()
        result  = inference_service.invoke_endpoint(
            endpoint_name=request.endpoint_name,
            input_data={
                request.input_type: request.input_value,
                "output_s3_uri": request.output_s3_uri,
                "output_pattern": request.output_pattern,
            },
            task_type=request.task_type,
            output_s3_uri=request.output_s3_uri,
            boto_session=session,
        )
        return result
    except Exception as e:
        logger.warning("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invoke-local")
async def invoke_endpoint_local(
    request: Request,
    endpoint_name: str,
    task_type: str = "classification",
):
    """Invoke endpoint with raw file bytes uploaded from the browser."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        body = await request.body()
        session = aws_service._get_boto_session()
        result = inference_service.invoke_endpoint(
            endpoint_name=endpoint_name,
            input_data={"bytes": body},
            task_type=task_type,
            output_s3_uri=None,
            boto_session=session,
        )
        return result
    except Exception as e:
        logger.warning("Local file inference failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/endpoints")
async def list_active_endpoints():
    """List InService SageMaker endpoints, enriched with task_type from local deployment store."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")

        # Import here to avoid circular import
        from app.services.deployment_service import DeploymentService
        deployment_service = DeploymentService()
        local_store = {d["endpoint_name"]: d for d in deployment_service.list_deployments()}

        response  = aws_service.sagemaker_client.list_endpoints(StatusEquals="InService")
        endpoints = []
        for e in response.get("Endpoints", []):
            name = e["EndpointName"]
            local = local_store.get(name, {})
            # Infer task_type from local store, then fall back to name heuristic
            task_type = local.get("task_type")
            if not task_type:
                task_type = "segmentation" if "segment" in name.lower() else "classification"
            instance_count = local.get("instance_count")
            endpoints.append({
                "name":           name,
                "status":         e["EndpointStatus"],
                "created":        e["CreationTime"].isoformat(),
                "task_type":      task_type,
                "endpoint_type":  local.get("endpoint_type", "realtime"),
                "output_s3_uri":  local.get("output_s3_uri", ""),
                "instance_count": instance_count,
                "scaled_to_zero": instance_count == 0,
            })
        return {"endpoints": endpoints}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/async-status")
async def check_async_status(output_location: str):
    """Check if an async inference result is available at the given S3 location."""
    try:
        if not aws_service._session:
            raise ValueError("AWS credentials not configured")
        import json as _json
        s3 = aws_service._session.client("s3")
        parts = output_location.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            result_str = obj["Body"].read().decode("utf-8")
            result = _json.loads(result_str)
            return {"completed": True, "result": result}
        except s3.exceptions.NoSuchKey:
            return {"completed": False}
        except Exception:
            return {"completed": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
