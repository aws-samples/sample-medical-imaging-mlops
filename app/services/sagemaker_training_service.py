"""Service for launching SageMaker BYOC training jobs"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
from app.models.schemas import SageMakerTrainingConfig

logger = logging.getLogger(__name__)

JOBS_FILE    = Path("data/sagemaker_jobs.json")
REPORTS_DIR  = Path("data/training_reports")

# On-demand pricing USD/hr (approximate, us-east-1)
INSTANCE_PRICING = {
    "ml.g5.xlarge":    1.408,
    "ml.g5.2xlarge":   1.515,
    "ml.g5.4xlarge":   2.030,
    "ml.g5.12xlarge":  7.090,
    "ml.g5.48xlarge": 24.840,
    "ml.p3.2xlarge":   3.825,
    "ml.p3.8xlarge":  14.688,
    "ml.p3.16xlarge": 28.152,
    "ml.m5.xlarge":    0.230,
    "ml.m5.2xlarge":   0.461,
    "ml.c5.xlarge":    0.204,
}


def _estimate_cost(instance_type: str, instance_count: int, billable_seconds: int) -> float:
    rate = INSTANCE_PRICING.get(instance_type, 0.0)
    return round(rate * instance_count * billable_seconds / 3600, 4)


class SageMakerTrainingService:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self._load_jobs()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_jobs(self):
        if JOBS_FILE.exists():
            try:
                self.jobs = json.loads(JOBS_FILE.read_text())
                logger.info("Loaded %d persisted SageMaker jobs", len(self.jobs))
            except Exception as e:
                logger.warning("Could not load jobs file: %s", e)
        else:
            # Bootstrap from training reports if jobs file is missing
            self._bootstrap_from_reports()

    def _bootstrap_from_reports(self):
        """Reconstruct jobs store from training report JSONs when jobs file is absent."""
        if not REPORTS_DIR.exists():
            return
        for report_file in sorted(REPORTS_DIR.glob("*.json")):
            try:
                report = json.loads(report_file.read_text())
                job_name = report.get("job_name")
                if not job_name:
                    continue
                hp = report.get("hyperparameters", {})
                self.jobs[job_name] = {
                    "job_name":          job_name,
                    "status":            report.get("status", "Completed"),
                    "model_artifacts":   report.get("model_artifacts"),
                    "failure_reason":    report.get("failure_reason"),
                    "billable_seconds":  report.get("billable_seconds"),
                    "estimated_cost_usd":report.get("estimated_cost_usd"),
                    "creation_time":     report.get("creation_time"),
                    "training_end_time": report.get("training_end_time"),
                    "archived":          False,
                    "config":            hp,
                    "result":            None,
                }
            except Exception as e:
                logger.warning("Could not parse report %s: %s", report_file, e)
        if self.jobs:
            logger.info("Bootstrapped %d jobs from training reports", len(self.jobs))
            self._save_jobs()

    def _save_jobs(self):
        try:
            JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
            JOBS_FILE.write_text(json.dumps(self.jobs, indent=2, default=str))
        except Exception as e:
            logger.warning("Could not save jobs file: %s", e)

    def _write_training_report(self, job_name: str):
        """Write a training report JSON for a completed job."""
        job = self.jobs.get(job_name)
        if not job:
            return
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report = {
                "job_name":         job_name,
                "status":           job.get("status"),
                "creation_time":    job.get("creation_time"),
                "training_end_time":job.get("training_end_time"),
                "billable_seconds": job.get("billable_seconds"),
                "estimated_cost_usd": job.get("estimated_cost_usd"),
                "model_artifacts":  job.get("model_artifacts"),
                "failure_reason":   job.get("failure_reason"),
                "hyperparameters":  {
                    "model_name":    job.get("config", {}).get("model_name"),
                    "task_type":     job.get("config", {}).get("task_type"),
                    "dimension":     job.get("config", {}).get("dimension"),
                    "epochs":        job.get("config", {}).get("epochs"),
                    "batch_size":    job.get("config", {}).get("batch_size"),
                    "learning_rate": job.get("config", {}).get("learning_rate"),
                    "val_interval":  job.get("config", {}).get("val_interval"),
                    "instance_type": job.get("config", {}).get("instance_type"),
                    "instance_count":job.get("config", {}).get("instance_count"),
                },
                "report_generated_at": datetime.utcnow().isoformat(),
            }
            report_path = REPORTS_DIR / f"{job_name}.json"
            report_path.write_text(json.dumps(report, indent=2, default=str))
            logger.info("Training report saved: %s", report_path)
        except Exception as e:
            logger.warning("Could not write training report: %s", e)

    # ── Job lifecycle ─────────────────────────────────────────────────────────

    def start_training_job(self, config: SageMakerTrainingConfig, sagemaker_session) -> dict:
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
            "data_s3_uri":   config.data_s3_uri,
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
        estimator.fit(inputs, wait=False, job_name=config.job_name or None)

        job_name = estimator.latest_training_job.name
        job_info = {
            "job_name":        job_name,
            "status":          "InProgress",
            "model_artifacts": None,
            "failure_reason":  None,
            "config":          config.model_dump(),
        }
        self.jobs[job_name] = job_info
        self._save_jobs()
        logger.info("Started SageMaker training job: %s", job_name)
        return job_info

    def get_job_status(self, job_name: str, sagemaker_client) -> dict:
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        status   = response["TrainingJobStatus"]
        config   = self.jobs.get(job_name, {}).get("config", {})

        billable_seconds = response.get("BillableTimeInSeconds") if status == "Completed" else None
        estimated_cost   = None
        if billable_seconds:
            estimated_cost = _estimate_cost(
                config.get("instance_type", ""),
                config.get("instance_count", 1),
                billable_seconds,
            )

        result = {
            "job_name":           job_name,
            "status":             status,
            "creation_time":      response.get("CreationTime", "").isoformat() if response.get("CreationTime") else None,
            "training_end_time":  response.get("TrainingEndTime", "").isoformat() if response.get("TrainingEndTime") else None,
            "model_artifacts":    response.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
            "billable_seconds":   billable_seconds,
            "estimated_cost_usd": estimated_cost,
        }
        if status == "Failed":
            result["failure_reason"] = response.get("FailureReason")
        return result

    def stop_training_job(self, job_name: str, sagemaker_client):
        sagemaker_client.stop_training_job(TrainingJobName=job_name)
        self.update_job_status(job_name, "Stopping")
        logger.info("Stopped SageMaker training job: %s", job_name)

    def update_job_status(self, job_name: str, status: str, **kwargs):
        if job_name in self.jobs:
            self.jobs[job_name]["status"] = status
            self.jobs[job_name].update(kwargs)
            self._save_jobs()
            if status == "Completed":
                self._write_training_report(job_name)

    def fetch_result_json(self, job_name: str, s3_client) -> dict:
        """Download model.tar.gz, extract result.json, store on job record.
        Safe to call multiple times — skips if already fetched."""
        job = self.jobs.get(job_name)
        if not job:
            raise KeyError(f"Job not found: {job_name}")
        if job.get("result"):
            return job["result"]   # already fetched

        uri = job.get("model_artifacts")
        if not uri:
            raise ValueError("No model_artifacts URI for this job")

        import io, tarfile, tempfile, os
        path   = uri.replace("s3://", "")
        bucket = path.split("/")[0]
        key    = "/".join(path.split("/")[1:])

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            s3_client.download_file(bucket, key, tmp_path)
            with tarfile.open(tmp_path, "r:gz") as tar:
                try:
                    member = tar.getmember("result.json")
                    f      = tar.extractfile(member)
                    result = json.loads(f.read().decode("utf-8"))
                except KeyError:
                    raise ValueError("result.json not found in model.tar.gz — was the model trained with the updated train.py?")
        finally:
            os.unlink(tmp_path)

        self.store_result(job_name, result)
        logger.info("Fetched result.json for job %s: %s", job_name, result)
        return result

    def list_jobs(self) -> list:
        return [
            {
                "job_name":           k,
                "status":             v.get("status"),
                "archived":           v.get("archived", False),
                "comment":            v.get("comment", ""),
                "model_artifacts":    v.get("model_artifacts"),
                "artifacts_deleted":  v.get("artifacts_deleted", False),
                "failure_reason":     v.get("failure_reason"),
                "billable_seconds":   v.get("billable_seconds"),
                "estimated_cost_usd": v.get("estimated_cost_usd"),
                "creation_time":      v.get("creation_time"),
                "training_end_time":  v.get("training_end_time"),
                "mlflow_tracking_uri": v.get("config", {}).get("mlflow_tracking_uri"),
                "config":             v.get("config"),
                "result":             v.get("result"),
            }
            for k, v in self.jobs.items()
        ]

    def remove_job(self, job_name: str):
        """Remove a job from local tracking (does not affect AWS)."""
        if job_name not in self.jobs:
            raise KeyError(f"Job not found: {job_name}")
        status = self.jobs[job_name].get("status")
        if status in {"InProgress", "Stopping"}:
            raise ValueError(f"Cannot remove an active job (status={status}). Stop it first.")
        self.jobs.pop(job_name)
        self._save_jobs()
        logger.info("Removed job from local store: %s", job_name)

    def archive_job(self, job_name: str):
        """Mark a job as archived — hidden from the default list."""
        if job_name not in self.jobs:
            raise KeyError(f"Job not found: {job_name}")
        self.jobs[job_name]["archived"] = True
        self._save_jobs()
        logger.info("Archived job: %s", job_name)

    def set_comment(self, job_name: str, comment: str):
        """Add or update a free-text comment on a job."""
        if job_name not in self.jobs:
            raise KeyError(f"Job not found: {job_name}")
        self.jobs[job_name]["comment"] = comment
        self._save_jobs()
        logger.info("Updated comment for job %s", job_name)

    def delete_model_artifacts(self, job_name: str, s3_client) -> str:
        """Delete model.tar.gz from S3 and mark the job accordingly."""
        job = self.jobs.get(job_name)
        if not job:
            raise KeyError(f"Job not found: {job_name}")
        uri = job.get("model_artifacts")
        if not uri:
            raise ValueError("No model_artifacts URI recorded for this job")
        if job.get("artifacts_deleted"):
            raise ValueError("Artifacts already deleted")

        path   = uri.replace("s3://", "")
        bucket = path.split("/")[0]
        key    = "/".join(path.split("/")[1:])
        s3_client.delete_object(Bucket=bucket, Key=key)
        logger.info("Deleted S3 object: %s", uri)

        self.jobs[job_name]["artifacts_deleted"] = True
        self._save_jobs()
        return uri

    def store_result(self, job_name: str, result: dict):
        """Store result.json contents against the job record."""
        if job_name in self.jobs:
            self.jobs[job_name]["result"] = result
            self._save_jobs()




