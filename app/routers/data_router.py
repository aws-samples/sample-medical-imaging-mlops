from fastapi import APIRouter, HTTPException
from app.models.schemas import DataImportRequest
from app.services.aws_service import AWSService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Use the same AWS service instance
from app.routers.aws_router import aws_service

@router.post("/import")
async def import_data_to_healthimaging(import_request: DataImportRequest):
    """Import data from S3 to AWS HealthImaging"""
    try:
        job_id = aws_service.import_to_healthimaging(
            import_request.bucket_name,
            import_request.prefix,
            import_request.destination_datastore_id
        )
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Data import job started successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Data import failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start data import job")

@router.get("/import/{job_id}/status")
async def get_import_status(job_id: str):
    """Get status of data import job"""
    # This would typically query the actual job status from AWS
    # For now, return a mock response
    return {
        "job_id": job_id,
        "status": "running",
        "progress": 45,
        "message": "Import in progress..."
    }