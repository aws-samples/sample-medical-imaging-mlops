from fastapi import APIRouter, HTTPException
from app.models.schemas import DataImportRequest
from app.services.aws_service import AWSService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

from app.routers.aws_router import aws_service

@router.post("/import")
async def import_data_from_s3(import_request: DataImportRequest):
    """Import data from S3"""
    try:
        objects = aws_service.list_bucket_objects(
            import_request.bucket_name,
            import_request.prefix
        )
        return {
            "status": "success",
            "message": f"Found {len(objects)} objects in s3://{import_request.bucket_name}/{import_request.prefix}",
            "objects": objects
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Data import failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to import data from S3")