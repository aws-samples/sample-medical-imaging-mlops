from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    TrainingConfig, TrainingJob, ModelType, JobStatus,
    ModelImportRequest, ImportedModel, SageMakerTrainingConfig,
)
from app.services.training_service import TrainingService
from app.services.sagemaker_training_service import SageMakerTrainingService
from app.routers.aws_router import aws_service
from typing import List
import logging
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter()

training_jobs = {}
training_service = TrainingService()
sm_training_service = SageMakerTrainingService()

SRC_DIR = Path("src")

# Metadata for known models — used when no model_info.json exists in the folder
MODEL_METADATA = {
    "densenet121": {"description": "DenseNet-121 for medical image classification", "parameters": "8M", "input_size": "256x256", "pretrained": True},
    "vit":         {"description": "Vision Transformer for medical imaging", "parameters": "86M", "input_size": "256x256", "pretrained": True},
    "resnet50":    {"description": "ResNet-50 for medical image classification", "parameters": "25.6M", "input_size": "224x224", "pretrained": True},
    "vgg16":       {"description": "VGG-16 for medical image analysis", "parameters": "138M", "input_size": "224x224", "pretrained": True},
    "efficientnet":{"description": "EfficientNet-B0 for efficient medical imaging", "parameters": "5.3M", "input_size": "224x224", "pretrained": True},
    "segresnet":   {"description": "SegResNet for 3D medical image segmentation", "parameters": "5M", "input_size": "128x128x64", "pretrained": False},
    "swinunetr":   {"description": "SwinUNETR for 3D medical image segmentation", "parameters": "62M", "input_size": "128x128x64", "pretrained": False},
}

@router.get("/models")
async def get_available_models():
    """Return models discovered from src/ subdirectories."""
    models = []

    if SRC_DIR.exists():
        for folder in sorted(SRC_DIR.iterdir()):
            if not folder.is_dir():
                continue
            name = folder.name
            # Try loading model_info.json from the folder first
            info_file = folder / "model_info.json"
            if info_file.exists():
                import json
                with open(info_file) as f:
                    meta = json.load(f)
            else:
                meta = MODEL_METADATA.get(name.lower(), {
                    "description": f"{name} model",
                    "parameters": "N/A",
                    "input_size": "N/A",
                    "pretrained": False,
                })
            models.append({
                "name": name,
                "description": meta.get("description", ""),
                "parameters": meta.get("parameters", "N/A"),
                "input_size": meta.get("input_size", "N/A"),
                "pretrained": meta.get("pretrained", False),
                "source_dir": str(folder),
            })

    if not models:
        logger.warning("No model folders found under src/. Returning empty list.")

    return {"models": models}



@router.post("/models/import", response_model=ImportedModel)
async def import_model(request: ModelImportRequest):
    """Import a model from S3"""
    try:
        # Verify the S3 object exists
        metadata = aws_service.get_s3_object_metadata(request.s3_bucket, request.s3_key)

        model_id = str(uuid.uuid4())
        filename = Path(request.s3_key).name
        local_path = training_service.get_local_path_for_import(model_id, filename)

        # Download from S3
        aws_service.download_from_s3(request.s3_bucket, request.s3_key, local_path)

        # Validate the model file
        training_service.validate_model_file(local_path, request.model_format)

        # Register
        entry = training_service.register_model(
            model_id=model_id,
            model_name=request.model_name,
            local_path=local_path,
            s3_source=f"s3://{request.s3_bucket}/{request.s3_key}",
            model_format=request.model_format,
            num_classes=request.num_classes,
        )

        return ImportedModel(
            model_id=entry["model_id"],
            model_name=entry["model_name"],
            s3_source=entry["s3_source"],
            local_path=entry["local_path"],
            model_format=entry["model_format"],
            num_classes=entry["num_classes"],
            imported_at=datetime.now().isoformat(),
            status=entry["status"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Model import failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to import model")


@router.get("/models/imported")
async def list_imported_models():
    """List all imported models"""
    return {"models": training_service.list_imported_models()}


@router.get("/models/imported/{model_id}")
async def get_imported_model(model_id: str):
    """Get details of an imported model"""
    model = training_service.get_imported_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Imported model not found")
    return model


@router.post("/start", response_model=TrainingJob)
async def start_training(config: TrainingConfig, dataset_path: str):
    """Start model training job"""
    try:
        job_id = str(uuid.uuid4())
        
        job = TrainingJob(
            job_id=job_id,
            status=JobStatus.PENDING,
            model_type=config.model_type,
            created_at=datetime.now().isoformat()
        )
        
        training_jobs[job_id] = job
        
        # In a real implementation, this would start the actual training process
        logger.info(f"Started training job {job_id} with model {config.model_type}")
        
        return job
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start training job")

@router.get("/jobs", response_model=List[TrainingJob])
async def list_training_jobs():
    """List all training jobs"""
    return list(training_jobs.values())

@router.get("/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(job_id: str):
    """Get specific training job details"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Mock status update
    job = training_jobs[job_id]
    if job.status == JobStatus.PENDING:
        job.status = JobStatus.RUNNING
    
    return job

@router.get("/jobs/{job_id}/metrics")
async def get_training_metrics(job_id: str):
    """Get training metrics for TensorBoard/MLflow integration"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Mock metrics data
    metrics = {
        "epoch": 5,
        "train_loss": 0.234,
        "val_loss": 0.267,
        "train_accuracy": 0.892,
        "val_accuracy": 0.876,
        "learning_rate": 0.001,
        "tensorboard_url": f"http://localhost:6006/#scalars&run={job_id}",
        "mlflow_url": f"http://localhost:5000/#/experiments/1/runs/{job_id}"
    }
    
    return {"job_id": job_id, "metrics": metrics}

@router.post("/jobs/{job_id}/stop")
async def stop_training_job(job_id: str):
    """Stop a running training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    if job.status == JobStatus.RUNNING:
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now().isoformat()
    
    return {"job_id": job_id, "status": "stopped"}


# --- SageMaker BYOC Training ---

@router.post("/sagemaker/validate")
async def validate_sagemaker_config(config: SageMakerTrainingConfig):
    """Validate data, credentials, IAM role and image URI before launching a training job."""
    from app.services.validation_service import validate_training_data
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        iam_client = aws_service._session.client("iam")
        result = validate_training_data(
            s3_client=aws_service.s3_client,
            iam_client=iam_client,
            data_s3_uri=config.data_s3_uri,
            task_type=config.task_type,
            role_arn=config.role_arn,
            image_uri=config.image_uri,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sagemaker/start")
async def start_sagemaker_training(config: SageMakerTrainingConfig):
    """Start a SageMaker BYOC training job (mirrors the notebook workflow)"""
    try:
        import sagemaker
        session = sagemaker.Session(boto_session=aws_service._get_boto_session())
        job_info = sm_training_service.start_training_job(config, session)
        return {
            "job_name": job_info["job_name"],
            "status": job_info["status"],
            "message": "SageMaker training job started",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"SageMaker training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start SageMaker training: {e}")


@router.get("/sagemaker/jobs")
async def list_sagemaker_jobs():
    """List all SageMaker training jobs launched from this platform"""
    return {"jobs": sm_training_service.list_jobs()}


@router.get("/sagemaker/jobs/{job_name}/status")
async def get_sagemaker_job_status(job_name: str):
    """Get status of a SageMaker training job"""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        return sm_training_service.get_job_status(job_name, aws_service.sagemaker_client)
    except Exception as e:
        logger.error(f"Failed to get SageMaker job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sagemaker/jobs/{job_name}/stop")
async def stop_sagemaker_job(job_name: str):
    """Stop a running SageMaker training job"""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        sm_training_service.stop_training_job(job_name, aws_service.sagemaker_client)
        return {"job_name": job_name, "status": "Stopping"}
    except Exception as e:
        logger.error(f"Failed to stop SageMaker job: {e}")
        raise HTTPException(status_code=500, detail=str(e))