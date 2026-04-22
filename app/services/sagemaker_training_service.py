"""Service for launching SageMaker BYOC training jobs"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from app.models.schemas import SageMakerTrainingConfig

logger = logging.getLogger(__name__)

JOBS_FILE = Path("data/sagemaker_jobs.json")


class SageMakerTrainingService:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self._load_jobs()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_jobs(self):
        """Load persisted jobs from disk on startup."""
        if JOBS_FILE.exists():
            try:
                self.jobs = json.loads(JOBS_FILE.read_text())
                logger.info("Loaded %d persisted SageMaker jobs", len(self.jobs))
            except Exception as e:
                logger.warning("Could not load jobs file: %s", e)

    def _save_jobs(self):
        """Persist current jobs dict to disk."""
        try:
            JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
            JOBS_FILE.write_text(json.dumps(self.jobs, indent=2, default=str))
        except Exception as e:
            logger.warning("Could not save jobs file: %s", e)

    # ── Job lifecycle ─────────────────────────────────────────────────────────

    def start_training_job(self, config: SageMakerTrainingConfig, sagemaker_session) -> dict:
        """Launch a SageMaker training job using PyTorch Estimator with BYOC"""
        from sagemaker.pytorch.estimator import PyTorch
        from sagemaker.inputs import TrainingInput

        hyperparameters = {
            "batch_size":    config.batch_size,
            "epochs":        config.epochs,
            "learning_rate": config.learning_rate,
            "model_name":    config.model_name,
            "val_interval":  config.val_interval,
            "task_type":     config.task_type,
            "dimension":     config.dimension,
        }

        env = {}
        if config.mlflow_tracking_uri:
            env["MLFLOW_TRACKING_URI"] = config.mlflow_tracking_uri

        estimator = PyTorch(
            source_dir=config.source_dir,
            entry_point=config.entry_point,
            image_uri=config.image_uri,
            instance_count=config.instance_count,
            instance_type=config.instance_type,
            role=config.role_arn,
            hyperparameters=hyperparameters,
            sagemaker_session=sagemaker_session,
            environment=env,
            output_path=config.output_s3_uri or None,
        )

        inputs = {"data": TrainingInput(s3_data=config.data_s3_uri)}
        estimator.fit(inputs, wait=False)

        job_name = estimator.latest_training_job.name
        job_info = {
            "job_name":        job_name,
            "status":          "InProgress",
            "model_artifacts": None,
            "failure_reason":  None,
            "config":          config.dict() if hasattr(config, "dict") else config.model_dump(),
        }
        self.jobs[job_name] = job_info
        self._save_jobs()
        logger.info("Started SageMaker training job: %s", job_name)
        return job_info

    def get_job_status(self, job_name: str, sagemaker_client) -> dict:
        """Get the status of a SageMaker training job from AWS."""
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        status   = response["TrainingJobStatus"]
        result   = {
            "job_name":          job_name,
            "status":            status,
            "creation_time":     response.get("CreationTime", "").isoformat() if response.get("CreationTime") else None,
            "training_end_time": response.get("TrainingEndTime", "").isoformat() if response.get("TrainingEndTime") else None,
            "model_artifacts":   response.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
        }
        if status == "Completed":
            result["billable_seconds"] = response.get("BillableTimeInSeconds")
        if status == "Failed":
            result["failure_reason"] = response.get("FailureReason")
        return result

    def stop_training_job(self, job_name: str, sagemaker_client):
        """Stop a running SageMaker training job."""
        sagemaker_client.stop_training_job(TrainingJobName=job_name)
        self.update_job_status(job_name, "Stopping")
        logger.info("Stopped SageMaker training job: %s", job_name)

    def update_job_status(self, job_name: str, status: str, **kwargs):
        """Persist refreshed status back to store and disk."""
        if job_name in self.jobs:
            self.jobs[job_name]["status"] = status
            self.jobs[job_name].update(kwargs)
            self._save_jobs()

    def list_jobs(self) -> list:
        return [
            {
                "job_name":        k,
                "status":          v.get("status"),
                "model_artifacts": v.get("model_artifacts"),
                "failure_reason":  v.get("failure_reason"),
                "config":          v.get("config"),
            }
            for k, v in self.jobs.items()
        ]
