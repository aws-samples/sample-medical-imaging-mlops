"""Service for submitting and tracking SageMaker Processing Jobs that run the
MONAI preprocessing pipeline over a full dataset.

Mirrors EvaluationService: JSON job store + create_processing_job. The pipeline
is data-driven — a spec.json is uploaded alongside the static preprocess.py and
its helpers, so the batch job runs the SAME pipeline the local preview used.

RESEARCH USE ONLY.
"""
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

PREPROCESS_JOBS_FILE = Path("data/preprocessing_jobs.json")
_SRC = Path(__file__).parent.parent.parent / "src" / "preprocessing"
_S3_RE = re.compile(r"^s3://([a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])/(.+)$")


def _valid_s3_uri(uri: str) -> bool:
    """Accept s3://bucket/prefix or s3://bucket/prefix/ (bucket 3-63 chars)."""
    if not isinstance(uri, str):
        return False
    return bool(_S3_RE.match(uri.rstrip("/") + "/marker"))


class PreprocessingService:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────
    def _load(self):
        if PREPROCESS_JOBS_FILE.exists():
            try:
                self.jobs = json.loads(PREPROCESS_JOBS_FILE.read_text())
            except Exception as e:
                logger.warning("Could not load preprocessing jobs: %s", e)

    def _save(self):
        PREPROCESS_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PREPROCESS_JOBS_FILE.write_text(json.dumps(self.jobs, indent=2, default=str))

    # ── Submit ──────────────────────────────────────────────────────────────────
    def submit_batch(
        self,
        *,
        input_s3_uri: str,
        output_s3_uri: str,
        spec: dict,
        image_uri: str,
        role_arn: str,
        sagemaker_client,
        s3_client,
        instance_type: str = "ml.m5.2xlarge",
        volume_size_gb: int = 100,
        region: str = "us-east-1",
    ) -> dict:
        """Upload code + spec and create a SageMaker Processing Job."""
        for name, uri in (("input", input_s3_uri), ("output", output_s3_uri)):
            if not _valid_s3_uri(uri):
                raise ValueError(f"Invalid {name} S3 URI: {uri}")
        if not image_uri:
            raise ValueError("Processing image URI is required")
        if not role_arn:
            raise ValueError("SageMaker role ARN is required")

        job_name = f"preprocess-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        out_bucket = output_s3_uri.replace("s3://", "").split("/")[0]
        code_prefix = f"preprocessing-scripts/{job_name}"

        # Upload the static script + shared helpers + the data-driven spec.
        uploads = {
            "preprocess.py": _SRC / "preprocess.py",
            "pipeline_builder.py": _SRC / "pipeline_builder.py",
            "dataset_resolver.py": _SRC / "dataset_resolver.py",
            "requirements.txt": _SRC / "requirements.txt",
        }
        # The code prefix is mounted as the job's code channel: never upload it
        # to a bucket another account could own (bucket squatting).
        from app.services.s3_ownership import assert_bucket_owned, owner_extra_args
        assert_bucket_owned(s3_client, out_bucket)
        for key, path in uploads.items():
            s3_client.upload_file(str(path), out_bucket, f"{code_prefix}/{key}",
                                  ExtraArgs=owner_extra_args() or None)
        s3_client.put_object(Bucket=out_bucket, Key=f"{code_prefix}/spec.json",
                             Body=json.dumps(spec).encode("utf-8"), **owner_extra_args())

        code_s3_uri = f"s3://{out_bucket}/{code_prefix}"

        processing_job_config = {
            "ProcessingJobName": job_name,
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": instance_type,        # CPU instance
                    "VolumeSizeInGB": int(volume_size_gb),
                }
            },
            "AppSpecification": {
                "ImageUri": image_uri,
                "ContainerEntrypoint": [
                    "bash", "-c",
                    "pip install -q -r /opt/ml/processing/code/requirements.txt && "
                    "python3 /opt/ml/processing/code/preprocess.py "
                    "--spec-path /opt/ml/processing/code/spec.json "
                    "--dataset-input /opt/ml/processing/input "
                    "--output /opt/ml/processing/output"
                ],
            },
            "ProcessingInputs": [
                {
                    "InputName": "dataset",
                    "S3Input": {
                        "S3Uri": input_s3_uri,
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
                {
                    "InputName": "code",
                    "S3Input": {
                        "S3Uri": code_s3_uri,
                        "LocalPath": "/opt/ml/processing/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "processed",
                        "S3Output": {
                            "S3Uri": output_s3_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            "RoleArn": role_arn,
            "Environment": {
                "AWS_DEFAULT_REGION": region,
                "PP_INPUT_S3_URI": input_s3_uri,
                "PP_OUTPUT_S3_URI": output_s3_uri,
            },
        }

        sagemaker_client.create_processing_job(**processing_job_config)
        logger.info("Submitted preprocessing job: %s", job_name)

        record = {
            "job_name": job_name,
            "input_s3_uri": input_s3_uri,
            "output_s3_uri": output_s3_uri,
            "instance_type": instance_type,
            "volume_size_gb": int(volume_size_gb),
            "image_uri": image_uri,
            "spec": spec,
            "status": "InProgress",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "failure_reason": None,
        }
        self.jobs[job_name] = record
        self._save()
        return record

    # ── Status / list / delete ──────────────────────────────────────────────────
    def get_status(self, job_name: str, sagemaker_client) -> dict:
        self._load()
        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name!r} not found")
        resp = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
        status = resp["ProcessingJobStatus"]
        record = self.jobs[job_name]
        record["status"] = status
        if status == "Completed":
            end = resp.get("ProcessingEndTime")
            record["completed_at"] = end.isoformat() if hasattr(end, "isoformat") else str(end) if end else None
        if status == "Failed":
            record["failure_reason"] = resp.get("FailureReason", "Unknown")
        self._save()
        return record

    def list_jobs(self, sagemaker_client=None) -> list:
        self._load()
        if sagemaker_client:
            for job_name, record in list(self.jobs.items()):
                if record.get("status") in ("InProgress", "Stopping"):
                    try:
                        self.get_status(job_name, sagemaker_client)
                    except Exception as e:
                        logger.warning("Could not refresh %s: %s", job_name, e)
        return sorted(self.jobs.values(), key=lambda j: j.get("created_at", ""), reverse=True)

    def delete_job(self, job_name: str) -> dict:
        """Remove a job record. Does NOT delete S3 outputs."""
        self._load()
        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name!r} not found")
        del self.jobs[job_name]
        self._save()
        return {"deleted": job_name}

    def sync_from_aws(self, sagemaker_client, name_contains: str = None) -> dict:
        kwargs = {"SortBy": "CreationTime", "SortOrder": "Descending", "MaxResults": 100,
                  "NameContains": name_contains or "preprocess-"}
        response = sagemaker_client.list_processing_jobs(**kwargs)
        added, updated = [], []
        for summary in response.get("ProcessingJobSummaries", []):
            job_name = summary["ProcessingJobName"]
            status = summary["ProcessingJobStatus"]
            end_time = summary.get("ProcessingEndTime")
            if job_name in self.jobs:
                self.jobs[job_name]["status"] = status
                if end_time:
                    self.jobs[job_name]["completed_at"] = end_time.isoformat() if hasattr(end_time, "isoformat") else str(end_time)
                updated.append(job_name)
            else:
                creation = summary.get("CreationTime")
                self.jobs[job_name] = {
                    "job_name": job_name, "input_s3_uri": None, "output_s3_uri": None,
                    "instance_type": None, "image_uri": None, "spec": None,
                    "status": status,
                    "created_at": creation.isoformat() if hasattr(creation, "isoformat") else str(creation) if creation else None,
                    "completed_at": end_time.isoformat() if hasattr(end_time, "isoformat") else str(end_time) if end_time else None,
                    "failure_reason": None,
                }
                added.append(job_name)
        self._save()
        return {"added": added, "updated": updated}
