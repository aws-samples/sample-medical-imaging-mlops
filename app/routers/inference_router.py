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
    """List InService SageMaker endpoints."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        response  = aws_service.sagemaker_client.list_endpoints(StatusEquals="InService")
        endpoints = [
            {"name": e["EndpointName"], "status": e["EndpointStatus"],
             "created": e["CreationTime"].isoformat()}
            for e in response.get("Endpoints", [])
        ]
        return {"endpoints": endpoints}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
