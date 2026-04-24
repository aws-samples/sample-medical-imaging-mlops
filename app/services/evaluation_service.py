"""Service for submitting and tracking SageMaker Processing Jobs for endpoint evaluation."""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

EVAL_JOBS_FILE = Path("data/evaluation_jobs.json")


class EvaluationService:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if EVAL_JOBS_FILE.exists():
            try:
                self.jobs = json.loads(EVAL_JOBS_FILE.read_text())
                logger.info("Loaded %d evaluation jobs", len(self.jobs))
            except Exception as e:
                logger.warning("Could not load evaluation jobs: %s", e)

    def _save(self):
        EVAL_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        EVAL_JOBS_FILE.write_text(json.dumps(self.jobs, indent=2, default=str))

    # ── Submit ────────────────────────────────────────────────────────────────

    def submit_evaluation(
        self,
        endpoint_name: str,
        data_s3_uri: str,
        output_s3_uri: str,
        image_uri: str,
        role_arn: str,
        instance_type: str,
        sagemaker_client,
        region: str = "us-east-1",
        task_type: str = "classification",
    ) -> dict:
        """Submit a SageMaker Processing Job that evaluates a classification endpoint."""
        job_name = f"eval-{endpoint_name[:30]}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        # Upload evaluate.py as the processing script
        import boto3, tempfile, os
        from pathlib import Path as _Path

        script_path = _Path(__file__).parent.parent.parent / "src" / "evaluation" / task_type / "evaluate.py"
        req_path    = _Path(__file__).parent.parent.parent / "src" / "evaluation" / task_type / "requirements.txt"

        # Derive S3 bucket from output_s3_uri for script upload
        out_bucket = output_s3_uri.replace("s3://", "").split("/")[0]
        script_key = f"evaluation-scripts/{job_name}/evaluate.py"
        req_key    = f"evaluation-scripts/{job_name}/requirements.txt"

        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(str(script_path), out_bucket, script_key)
        s3.upload_file(str(req_path),    out_bucket, req_key)

        code_s3_uri = f"s3://{out_bucket}/evaluation-scripts/{job_name}"

        processing_job_config = {
            "ProcessingJobName": job_name,
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount":  1,
                    "InstanceType":   instance_type,
                    "VolumeSizeInGB": 30,
                }
            },
            "AppSpecification": {
                "ImageUri":         image_uri,
                "ContainerEntrypoint": [
                    "bash", "-c",
                    "pip install -q -r /opt/ml/processing/code/requirements.txt && "
                    "python3 /opt/ml/processing/code/evaluate.py "
                    "--endpoint-name $EVAL_ENDPOINT_NAME "
                    "--data-s3-uri $EVAL_DATA_S3_URI "
                    "--output-s3-uri $EVAL_OUTPUT_S3_URI "
                    "--region $AWS_DEFAULT_REGION"
                ],
            },
            "ProcessingInputs": [
                {
                    "InputName": "code",
                    "S3Input": {
                        "S3Uri":            code_s3_uri,
                        "LocalPath":        "/opt/ml/processing/code",
                        "S3DataType":       "S3Prefix",
                        "S3InputMode":      "File",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                }
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "results",
                        "S3Output": {
                            "S3Uri":        output_s3_uri,
                            "LocalPath":    "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "RoleArn": role_arn,
            "Environment": {
                "AWS_DEFAULT_REGION":  region,
                "EVAL_ENDPOINT_NAME":  endpoint_name,
                "EVAL_DATA_S3_URI":    data_s3_uri,
                "EVAL_OUTPUT_S3_URI":  output_s3_uri,
            },
        }

        sagemaker_client.create_processing_job(**processing_job_config)
        logger.info("Submitted processing job: %s", job_name)

        record = {
            "job_name":      job_name,
            "endpoint_name": endpoint_name,
            "task_type":     task_type,
            "data_s3_uri":   data_s3_uri,
            "output_s3_uri": output_s3_uri,
            "instance_type": instance_type,
            "image_uri":     image_uri,
            "status":        "InProgress",
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "completed_at":  None,
            "metrics":       None,
            "failure_reason": None,
        }
        self.jobs[job_name] = record
        self._save()
        return record

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self, job_name: str, sagemaker_client) -> dict:
        """Refresh job status from AWS and return the updated record."""
        self._load()   # re-sync from disk in case another process wrote a new job
        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name!r} not found")

        resp   = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
        status = resp["ProcessingJobStatus"]   # InProgress | Completed | Failed | Stopping | Stopped

        record = self.jobs[job_name]
        record["status"] = status

        if status == "Completed":
            end = resp.get("ProcessingEndTime")
            record["completed_at"] = end.isoformat() if end else None
            # Try to fetch metrics.json from S3
            record["metrics"] = self._fetch_metrics(record["output_s3_uri"])

        if status == "Failed":
            record["failure_reason"] = resp.get("FailureReason", "Unknown")

        self._save()
        return record

    RESULTS_DIR = Path("data/evaluation_results")
    PLOT_FILES  = {"confusion_matrix.png", "roc_auc.png", "subgroup_accuracy.png", "dice_distribution.png"}

    def download_results(self, job_name: str, s3_client, plots_only: bool = False) -> dict:
        """
        Download output files for a completed job from S3 into
        data/evaluation_results/<job_name>/.
        If plots_only=True, only downloads PNG files (fast).
        Returns a dict of { filename: local_url_path }.
        """
        self._load()
        logger.info("download_results called for %r, known jobs: %s", job_name, list(self.jobs.keys()))
        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name!r} not found")

        record     = self.jobs[job_name]
        output_uri = record.get("output_s3_uri", "")
        if not output_uri:
            raise ValueError("No output S3 URI on this job record")

        bucket, prefix = output_uri.replace("s3://", "").split("/", 1)
        prefix = prefix.rstrip("/")

        local_dir = self.RESULTS_DIR / job_name
        local_dir.mkdir(parents=True, exist_ok=True)

        # List all objects at the output prefix
        paginator  = s3_client.get_paginator("list_objects_v2")
        downloaded = dict(record.get("cached_results", {}))  # preserve existing

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key      = obj["Key"]
                filename = key.split("/")[-1]
                if not filename:
                    continue
                if plots_only and filename not in self.PLOT_FILES:
                    continue
                local_path = local_dir / filename
                try:
                    s3_client.download_file(bucket, key, str(local_path))
                    downloaded[filename] = f"/evaluation-results/{job_name}/{filename}"
                    logger.info("Cached %s → %s", key, local_path)
                except Exception as e:
                    logger.warning("Could not download %s: %s", key, e)

        record["cached_results"] = downloaded
        self._save()
        return downloaded

    def delete_job(self, job_name: str) -> dict:
        """Remove a job record and its local cache."""
        import shutil
        self._load()
        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name!r} not found")
        local_dir = self.RESULTS_DIR / job_name
        if local_dir.exists():
            shutil.rmtree(local_dir)
        del self.jobs[job_name]
        self._save()
        return {"deleted": job_name}

    def clear_cache(self, job_name: str) -> dict:
        """Delete locally cached result files for a job."""
        local_dir = self.RESULTS_DIR / job_name
        removed = []
        if local_dir.exists():
            import shutil
            for f in local_dir.iterdir():
                removed.append(f.name)
            shutil.rmtree(local_dir)
            logger.info("Cleared cache for %s", job_name)

        # Remove cached_results from the job record
        self._load()
        if job_name in self.jobs:
            self.jobs[job_name].pop("cached_results", None)
            self._save()

        return {"removed": removed}

    def _fetch_metrics(self, output_s3_uri: str) -> Optional[dict]:
        """Download metrics.json from the job output prefix."""
        try:
            import boto3
            bucket, prefix = output_s3_uri.replace("s3://", "").split("/", 1)
            key = prefix.rstrip("/") + "/metrics.json"
            s3  = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read().decode("utf-8"))
        except Exception as e:
            logger.warning("Could not fetch metrics.json: %s", e)
            return None

    # ── List ──────────────────────────────────────────────────────────────────

    def list_jobs(self, sagemaker_client=None) -> list:
        """Return all jobs, refreshing status only for in-progress ones."""
        self._load()
        if sagemaker_client:
            for job_name, record in list(self.jobs.items()):
                if record.get("status") in ("InProgress", "Stopping"):
                    try:
                        self.get_status(job_name, sagemaker_client)
                    except Exception as e:
                        logger.warning("Could not refresh status for %s: %s", job_name, e)
        return sorted(self.jobs.values(), key=lambda j: j.get("created_at", ""), reverse=True)
