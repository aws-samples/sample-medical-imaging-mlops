from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
from app.models.schemas import AWSCredentials, S3Bucket
from app.services.aws_service import AWSService

logger = logging.getLogger(__name__)
router = APIRouter()

# Global AWS service instance
aws_service = AWSService()

# Auto-configure from terminal/environment credentials on startup
_env_configured = aws_service.configure_from_environment()


@router.get("/status")
async def get_aws_status():
    """Check if AWS is configured"""
    configured = aws_service.s3_client is not None
    return {"configured": configured, "source": "environment" if _env_configured else "manual"}


@router.post("/configure")
async def configure_aws_credentials(credentials: AWSCredentials):
    """Configure AWS credentials"""
    try:
        success = aws_service.configure_credentials(credentials)
        if success:
            return {"status": "success", "message": "AWS credentials configured successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid AWS credentials")
    except Exception as e:
        logger.error(f"AWS configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/buckets", response_model=List[S3Bucket])
async def list_s3_buckets():
    """List all S3 buckets"""
    try:
        buckets = aws_service.list_s3_buckets()
        return buckets
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to list buckets: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve S3 buckets")

@router.get("/buckets/{bucket_name}/objects")
async def list_bucket_objects(bucket_name: str, prefix: str = ""):
    """List objects in S3 bucket"""
    try:
        objects = aws_service.list_bucket_objects(bucket_name, prefix)
        return {"bucket": bucket_name, "objects": objects}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to list bucket objects: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve bucket objects")

@router.post("/healthimaging/datastore")
async def create_datastore(datastore_name: str):
    """Create HealthImaging datastore"""
    try:
        datastore_id = aws_service.create_healthimaging_datastore(datastore_name)
        return {"datastore_id": datastore_id, "name": datastore_name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create datastore: {e}")
        raise HTTPException(status_code=500, detail="Failed to create HealthImaging datastore")