"""Router for fairness / demographic analysis of DICOM datasets."""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.routers.aws_router import aws_service
from app.services.fairness_service import FairnessService

logger = logging.getLogger(__name__)
router = APIRouter()

fairness_service = FairnessService()


class FairnessScanRequest(BaseModel):
    s3_uri:     str
    max_files:  int = 200


@router.post("/scan")
async def scan_demographics(request: FairnessScanRequest):
    """Scan DICOM files in an S3 prefix and return demographic distributions."""
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        result = fairness_service.scan_s3_demographics(
            s3_uri=request.s3_uri,
            s3_client=aws_service.s3_client,
            max_files=request.max_files,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Fairness scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
