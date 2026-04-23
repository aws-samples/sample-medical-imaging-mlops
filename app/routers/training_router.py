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

# Task folders that contain a model_def.py with a MODELS registry dict
TASK_DIRS = {
    "classification": SRC_DIR / "classification",
    "segmentation":   SRC_DIR / "segmentation",
}


def _load_models_from_task(task: str) -> list:
    """Parse MODELS dict from a task's model_def.py using AST — no imports needed."""
    import ast
    task_dir = TASK_DIRS.get(task)
    if not task_dir or not task_dir.exists():
        return []
    model_def_path = task_dir / "model_def.py"
    if not model_def_path.exists():
        return []

    try:
        tree = ast.parse(model_def_path.read_text())
    except SyntaxError:
        logger.warning("Syntax error parsing %s", model_def_path)
        return []

    models_dict = {}
    for node in ast.walk(tree):
        # Find: MODELS = { ... }
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODELS":
                    try:
                        models_dict = ast.literal_eval(node.value)
                    except Exception:
                        pass

    return [
        {
            "name":        name,
            "description": meta.get("description", ""),
            "parameters":  meta.get("parameters", "N/A"),
            "input_size":  meta.get("input_size", "N/A"),
            "pretrained":  meta.get("pretrained", False),
            "task":        meta.get("task", task),
            "dimension":   meta.get("dimension", "2D"),
            "source_dir":  str(task_dir),
        }
        for name, meta in models_dict.items()
    ]

@router.get("/models")
async def get_available_models():
    """Return models from all task model_def.py registries."""
    models = []
    for task in TASK_DIRS:
        models.extend(_load_models_from_task(task))
    if not models:
        logger.warning("No models found in src/classification or src/segmentation.")
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
    """List all SageMaker training jobs, refreshing status from AWS"""
    jobs = sm_training_service.list_jobs()
    if aws_service.sagemaker_client:
        terminal_states = {"Completed", "Failed", "Stopped"}
        for job in jobs:
            if job.get("status") in terminal_states:
                # Auto-fetch result.json for completed jobs that don't have it yet
                if job.get("status") == "Completed" and not job.get("result") and job.get("model_artifacts") and not job.get("artifacts_deleted") and aws_service.s3_client:
                    try:
                        result = sm_training_service.fetch_result_json(job["job_name"], aws_service.s3_client)
                        job["result"] = result
                    except Exception as e:
                        logger.debug("Could not fetch result.json for %s: %s", job["job_name"], e)
                continue
            try:
                updated = sm_training_service.get_job_status(job["job_name"], aws_service.sagemaker_client)
                job["status"] = updated["status"]
                job["model_artifacts"] = updated.get("model_artifacts")
                job["failure_reason"]  = updated.get("failure_reason")
                sm_training_service.update_job_status(
                    job["job_name"], updated["status"],
                    model_artifacts=updated.get("model_artifacts"),
                    failure_reason=updated.get("failure_reason"),
                    creation_time=updated.get("creation_time"),
                    training_end_time=updated.get("training_end_time"),
                    billable_seconds=updated.get("billable_seconds"),
                    estimated_cost_usd=updated.get("estimated_cost_usd"),
                )
            except Exception:
                pass
    return {"jobs": jobs}


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


@router.get("/sagemaker/jobs/{job_name}/result")
async def get_job_result(job_name: str):
    """Fetch result.json from model.tar.gz for a completed job."""
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        result = sm_training_service.fetch_result_json(job_name, aws_service.s3_client)
        return {"job_name": job_name, "result": result}
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to fetch result.json for %s: %s", job_name, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sagemaker/jobs/{job_name}")
async def remove_sagemaker_job(job_name: str):
    """Remove a terminal job from local tracking."""
    try:
        sm_training_service.remove_job(job_name)
        return {"job_name": job_name, "removed": True}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sagemaker/jobs/{job_name}/archive")
async def archive_sagemaker_job(job_name: str):
    """Archive a job (hide from default list)."""
    try:
        sm_training_service.archive_job(job_name)
        return {"job_name": job_name, "archived": True}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/sagemaker/jobs/{job_name}/comment")
async def set_job_comment(job_name: str, body: dict):
    """Add or update a comment on a job."""
    try:
        sm_training_service.set_comment(job_name, body.get("comment", ""))
        return {"job_name": job_name, "comment": body.get("comment", "")}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/sagemaker/jobs/{job_name}/artifacts")
async def delete_job_artifacts(job_name: str):
    """Delete model.tar.gz from S3 for a completed job."""
    try:
        if not aws_service.s3_client:
            raise ValueError("AWS credentials not configured")
        uri = sm_training_service.delete_model_artifacts(job_name, aws_service.s3_client)
        return {"job_name": job_name, "deleted_uri": uri}
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to delete artifacts for %s: %s", job_name, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sagemaker/mlflow-url")
async def get_mlflow_url(arn: str):
    """Resolve a SageMaker MLflow tracking server ARN to its HTTP URL."""
    try:
        if not aws_service.sagemaker_client:
            raise ValueError("AWS credentials not configured")
        url = aws_service.get_mlflow_tracking_url(arn)
        return {"url": url}
    except Exception as e:
        logger.error(f"Failed to resolve MLflow ARN: {e}")
        raise HTTPException(status_code=500, detail=str(e))
