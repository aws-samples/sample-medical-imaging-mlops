"""Service for launching SageMaker BYOC training jobs"""
import logging
from typing import Dict, Optional
from app.models.schemas import SageMakerTrainingConfig

logger = logging.getLogger(__name__)


class SageMakerTrainingService:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}

    def start_training_job(self, config: SageMakerTrainingConfig, sagemaker_session) -> dict:
        """Launch a SageMaker training job using PyTorch Estimator with BYOC"""
        from sagemaker.pytorch.estimator import PyTorch
        from sagemaker.inputs import TrainingInput

        hyperparameters = {
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "model_name": config.model_name,
            "num_classes": config.num_classes,
            "val_interval": config.val_interval,
        }

        env = {}
        if config.mlflow_tracking_uri:
            env["MLFLOW_TRACKING_URI"] = config.mlflow_tracking_uri

        estimator = PyTorch(
            source_dir=config.source_dir,
            entry_point=config.entry_point,
            framework_version=config.framework_version,
            py_version=config.py_version,
            image_uri=config.image_uri,
            instance_count=config.instance_count,
            instance_type=config.instance_type,
            role=config.role_arn,
            hyperparameters=hyperparameters,
            dependencies=[f"{config.source_dir}/requirements.txt"],
            sagemaker_session=sagemaker_session,
            environment=env,
        )

        inputs = {
            "train": TrainingInput(s3_data=config.train_s3_uri),
            "test": TrainingInput(s3_data=config.test_s3_uri),
        }

        estimator.fit(inputs, wait=False)

        job_name = estimator.latest_training_job.name
        job_info = {
            "job_name": job_name,
            "status": "InProgress",
            "config": config.model_dump(),
            "estimator": estimator,
        }
        self.jobs[job_name] = job_info
        logger.info(f"Started SageMaker training job: {job_name}")
        return job_info

    def get_job_status(self, job_name: str, sagemaker_client) -> dict:
        """Get the status of a SageMaker training job"""
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        status = response["TrainingJobStatus"]
        result = {
            "job_name": job_name,
            "status": status,
            "creation_time": response.get("CreationTime", "").isoformat() if response.get("CreationTime") else None,
            "training_end_time": response.get("TrainingEndTime", "").isoformat() if response.get("TrainingEndTime") else None,
            "model_artifacts": response.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
        }
        if status == "Completed":
            result["billable_seconds"] = response.get("BillableTimeInSeconds")
        if status == "Failed":
            result["failure_reason"] = response.get("FailureReason")
        return result

    def stop_training_job(self, job_name: str, sagemaker_client):
        """Stop a running SageMaker training job"""
        sagemaker_client.stop_training_job(TrainingJobName=job_name)
        logger.info(f"Stopped SageMaker training job: {job_name}")

    def list_jobs(self) -> list:
        return [
            {"job_name": k, "status": v.get("status"), "config": v.get("config")}
            for k, v in self.jobs.items()
        ]
