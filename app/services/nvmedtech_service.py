"""
NVIDIA MedTech (NV-Segment-CT / VISTA-3D) dashboard service.

Dedicated integration mirroring MonaiBundleService, but on the NV-Segment-CT
stack: dedicated images (sm-nvmedtech / sm-nvmedtech-inference, MONAI 1.4 on
PyTorch 2.4), the local bundle at NVIDIA-MedTech/NV-Segment-CT, and the
bundle's own composed fine-tune recipe (train.json + train_continual.json).

CT only — the MR sibling is commercially licensed and excluded.

⚠️ RESEARCH USE ONLY — outputs are not clinical diagnoses.
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

from app.services.monai_bundle_service import bundle_card_from_metadata, parse_s3_uri

logger = logging.getLogger(__name__)

JOBS_FILE = Path("data/nvmedtech_jobs.json")
BUNDLE_DIR = "NVIDIA-MedTech/NV-Segment-CT"
SOURCE_URL = "https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/main/NV-Segment-CT"
DEFAULT_TRAIN_CONFIG = "train.json,train_continual.json"

BUNDLE_INFO = {
    "name": "NV-Segment-CT",
    "task": "segmentation",
    "dimension": "3D",
    "description": ("VISTA-3D: multi-organ CT segmentation with zero-shot/interactive "
                    "prompts (127 classes). NVIDIA MedTech bundle; MONAI 1.4 / PyTorch 2.4 "
                    "dedicated images."),
    "source_url": SOURCE_URL,
}


class NvMedtechService:
    def __init__(self):
        self.jobs: dict = {}
        self.staged: dict = {}   # {"s3_uri", "papers", "data_org", ...}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self):
        if JOBS_FILE.exists():
            try:
                data = json.loads(JOBS_FILE.read_text())
                self.jobs = data.get("jobs", {})
                self.staged = data.get("staged", {})
            except Exception as e:
                logger.warning("Could not load nvmedtech jobs: %s", e)

    def _save(self):
        try:
            JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
            JOBS_FILE.write_text(json.dumps(
                {"jobs": self.jobs, "staged": self.staged}, indent=2, default=str))
        except Exception as e:
            logger.warning("Could not save nvmedtech jobs: %s", e)

    # ── bundle card (from the local bundle's own metadata.json) ──────────────
    def bundle_card(self) -> dict:
        meta = {}
        meta_path = os.path.join(BUNDLE_DIR, "configs", "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning("Could not read bundle metadata: %s", e)
        card = bundle_card_from_metadata(meta)
        # Make the training-data guidance VISTA-accurate (the generic helper's
        # MSD tree overstates dataset.json): the validated fine-tune consumes
        # imagesTr/labelsTr NIfTIs; the datalist comes from the bundle (Task09
        # spleen folds ship with it), a user-provided data_list_file_path, or —
        # optionally — a Decathlon dataset.json with generate_datalist=true.
        # label_mappings is the critical VISTA-specific prep step.
        data_org = dict(card.get("data_org") or {})
        data_org["layout"] = [
            "s3://<your-bucket>/<dataset>/",
            "├── imagesTr/        # CT volumes (.nii.gz)",
            "├── labelsTr/        # label volumes (.nii.gz), indices per your dataset",
            "└── dataset.json     # OPTIONAL — only needed with generate_datalist=true",
        ]
        data_org["training_data_note"] = (
            "Datalist: for MSD Task09_Spleen the bundle ships its own folds "
            "datalist (nothing extra needed). For other datasets provide a "
            "data_list_file_path override or a Decathlon dataset.json with "
            "generate_datalist=true. REQUIRED: label_mappings maps your label "
            "indices to VISTA-3D's global classes (e.g. spleen 1 → 3; full map "
            "in configs/label_dict.json)."
        )
        card["data_org"] = data_org
        return {**BUNDLE_INFO, **card,
                "monai_version": meta.get("monai_version"),
                "pytorch_version": meta.get("pytorch_version")}

    # ── staging ──────────────────────────────────────────────────────────────
    def stage(self, s3_uri: str, region: str = "us-east-1") -> dict:
        """Package the local bundle + HF checkpoint + serving handler to S3.
        Heavy (~900 MB upload) — run the dashboard on an AWS host for this."""
        parse_s3_uri(s3_uri)
        from src.nvmedtech.package import package_and_upload
        package_and_upload(BUNDLE_DIR, s3_uri, region)
        card = self.bundle_card()
        self.staged["NV-Segment-CT"] = {
            "s3_uri": s3_uri,
            "papers": card.get("papers", []),
            "data_org": card.get("data_org"),
            "source_url": SOURCE_URL,
        }
        self._save()
        return {"model_artifacts": s3_uri, **card}

    # ── fine-tuning ──────────────────────────────────────────────────────────
    def start_finetune(
        self,
        job_name: Optional[str],
        model_artifacts: str,
        data_s3_uri: str,
        role_arn: str,
        image_uri: str,
        sagemaker_client,
        instance_type: str = "ml.g5.4xlarge",
        instance_count: int = 1,
        volume_size_gb: int = 100,
        max_runtime_seconds: int = 3600,
        train_config: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        generate_datalist: bool = False,
        output_s3_uri: Optional[str] = None,
    ) -> dict:
        parse_s3_uri(model_artifacts)
        parse_s3_uri(data_s3_uri)
        overrides = dict(config_overrides or {})

        container_args = [
            "--train_config", train_config or DEFAULT_TRAIN_CONFIG,
            "--config_overrides", json.dumps(overrides),
            "--generate_datalist", "true" if generate_datalist else "false",
        ]
        if not job_name:
            from datetime import datetime
            job_name = "nvmedtech-" + datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        if not output_s3_uri:
            bucket, _ = parse_s3_uri(data_s3_uri)
            output_s3_uri = f"s3://{bucket}/nvmedtech-output/"

        request = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": image_uri,
                "TrainingInputMode": "File",
                "ContainerEntrypoint": ["python", "/opt/program/train.py"],
                "ContainerArguments": container_args,
            },
            "RoleArn": role_arn,
            "InputDataConfig": [
                {"ChannelName": "model",
                 "DataSource": {"S3DataSource": {
                     "S3DataType": "S3Prefix", "S3Uri": model_artifacts,
                     "S3DataDistributionType": "FullyReplicated"}}},
                {"ChannelName": "data",
                 "DataSource": {"S3DataSource": {
                     "S3DataType": "S3Prefix", "S3Uri": data_s3_uri,
                     "S3DataDistributionType": "FullyReplicated"}}},
            ],
            "OutputDataConfig": {"S3OutputPath": output_s3_uri},
            "ResourceConfig": {
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
                "VolumeSizeInGB": volume_size_gb,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_seconds},
        }
        sagemaker_client.create_training_job(**request)

        info = {
            "job_name": job_name,
            "status": "InProgress",
            "model_artifacts": None,
            "config": {
                "bundle_artifacts": model_artifacts,
                "data_s3_uri": data_s3_uri,
                "image_uri": image_uri,
                "train_config": train_config or DEFAULT_TRAIN_CONFIG,
                "config_overrides": overrides,
                "instance_type": instance_type,
                "instance_count": instance_count,
            },
        }
        self.jobs[job_name] = info
        self._save()
        logger.info("Started NV-Segment-CT fine-tune job: %s", job_name)
        return info

    # ── jobs ─────────────────────────────────────────────────────────────────
    def get_job_status(self, job_name: str, sagemaker_client) -> dict:
        resp = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        result = {
            "job_name": job_name,
            "status": status,
            "model_artifacts": resp.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
            "failure_reason": resp.get("FailureReason") if status == "Failed" else None,
        }
        if job_name in self.jobs:
            self.jobs[job_name].update(result)
            self._save()
        return result

    def list_jobs(self) -> list:
        return [{"job_name": k, **v} for k, v in self.jobs.items()]

    def refresh_jobs(self, sagemaker_client) -> list:
        TERMINAL = {"Completed", "Failed", "Stopped"}
        if sagemaker_client:
            for job_name, info in self.jobs.items():
                if info.get("status") in TERMINAL:
                    continue
                try:
                    self.get_job_status(job_name, sagemaker_client)
                except Exception as e:
                    logger.warning("Could not refresh status for %s: %s", job_name, e)
        return self.list_jobs()

    # ── deployable artifacts ─────────────────────────────────────────────────
    def list_deployable_artifacts(self) -> dict:
        card = self.bundle_card()
        staged = self.staged.get("NV-Segment-CT") or {}
        base = [{
            **card,
            "papers": staged.get("papers", card.get("papers", [])),
            "data_org": staged.get("data_org", card.get("data_org")),
            "model_artifacts": staged.get("s3_uri"),
            "deployable": bool(staged.get("s3_uri")),
        }]
        finetuned = []
        for name, j in self.jobs.items():
            art = j.get("model_artifacts")
            if art and j.get("status") == "Completed":
                cfg = j.get("config") or {}
                finetuned.append({
                    "name": name,
                    "task": "segmentation",
                    "model_artifacts": art,
                    "data_s3_uri": cfg.get("data_s3_uri"),
                    "instance_type": cfg.get("instance_type"),
                    "bundle_artifacts": cfg.get("bundle_artifacts"),
                    "deployable": True,
                })
        return {"base": base, "finetuned": finetuned}

    # ── deployment (async only) ──────────────────────────────────────────────
    def deploy(
        self,
        endpoint_name: str,
        model_artifacts: str,
        image_uri: str,
        role_arn: str,
        sagemaker_session,
        output_s3_uri: str,
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
        max_concurrent: int = 4,
    ) -> dict:
        from app.services.deployment_service import DeploymentService
        parse_s3_uri(model_artifacts)
        if not output_s3_uri:
            raise ValueError("output_s3_uri is required (async endpoints write results to S3)")
        parse_s3_uri(output_s3_uri)
        config = {
            "endpoint_name": endpoint_name,
            "model_artifacts": model_artifacts,
            "image_uri": image_uri,
            "inference_script": "inference.py",
            "instance_type": instance_type,
            "instance_count": instance_count,
            "async_endpoint": True,
            "output_s3_uri": output_s3_uri,
            "max_concurrent": max_concurrent,
        }
        return DeploymentService().create_endpoint(
            config=config, sagemaker_session=sagemaker_session, role_arn=role_arn
        )


nvmedtech_service = NvMedtechService()
