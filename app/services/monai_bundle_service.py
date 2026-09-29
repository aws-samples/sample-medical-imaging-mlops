"""
Service for MONAI Bundle operations in MedAnvil:
  - stage a MONAI Model Zoo bundle into S3 as model.tar.gz
  - launch a bundle fine-tuning job on SageMaker (source_dir approach — no custom
    image required; MONAI is installed at job start via requirements.txt)
  - track jobs / deploy artifacts by reusing existing services

Runs the generic entry points in src/monai_bundle/ inside a stock SageMaker
PyTorch container via `source_dir`, so it does not depend on a custom ECR image.

RESEARCH USE ONLY — outputs are for research/development, not clinical use.
"""
import json
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Path to the generic MONAI bundle entry points shipped with the repo.
BUNDLE_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src" / "monai_bundle")

JOBS_FILE = Path("data/monai_bundle_jobs.json")


# Registry of MONAI bundles known to work with this integration.
# These are MONAI Model Zoo bundles (configs/ + models/ driven by monai.bundle run).
SUPPORTED_BUNDLES = [
    {
        "name": "spleen_ct_segmentation",
        "repo_id": "MONAI/spleen_ct_segmentation",
        "task": "segmentation",
        "dimension": "3D",
        "description": "Spleen segmentation in CT (UNet). MONAI Model Zoo bundle.",
        "parameters": "~19 MB checkpoint",
        "input_size": "3D CT (NIfTI)",
        "default_train_config": "train.json",
        "default_inference_config": "inference.json",
    },
    {
        "name": "wholeBody_ct_segmentation",
        "repo_id": "MONAI/wholeBody_ct_segmentation",
        "task": "segmentation",
        "dimension": "3D",
        "description": "Whole-body multi-organ CT segmentation (SegResNet). MONAI Model Zoo bundle.",
        "parameters": "104 classes",
        "input_size": "3D CT (NIfTI)",
        "default_train_config": "train.json",
        "default_inference_config": "inference.json",
    },
    {
        "name": "breast_density_classification",
        "repo_id": "MONAI/breast_density_classification",
        "task": "classification",
        "dimension": "2D",
        "description": "Mammographic breast density classification, BI-RADS A-D (InceptionV3). MONAI Model Zoo bundle. Inference-only: the bundle ships no training config.",
        "parameters": "4 classes",
        "input_size": "2D mammogram JPEG (299x299)",
        "default_train_config": None,
        "default_inference_config": "inference.json",
    },
    {
        "name": "pathology_nuclei_segmentation_classification",
        "repo_id": "MONAI/pathology_nuclei_segmentation_classification",
        "task": "segmentation",
        "dimension": "2D",
        "description": "HoVer-Net nuclear segmentation + classification in H&E histology patches (CoNSeP). MONAI Model Zoo bundle. Requires MONAI 1.4 / torch 2.4 — deploy with the sm-nvmedtech-inference image via per-bundle image_uri, not the shared image.",
        "parameters": "instance + type maps",
        "input_size": "2D H&E patch PNG (256x256)",
        "default_train_config": "train.json",
        "default_inference_config": "inference.json",
    },
    {
        "name": "wholeBrainSeg_Large_UNEST_segmentation",
        "repo_id": "MONAI/wholeBrainSeg_Large_UNEST_segmentation",
        "task": "segmentation",
        "dimension": "3D",
        "description": "Whole-brain segmentation in T1-weighted MRI, 133 structures (UNEST Large). MONAI Model Zoo bundle. Metadata declares MONAI 1.4/torch 2.4, but it runs on the shared sm-monai-bundle images (verified: async inference + training) — its extra deps (timm/einops) install from the artifact requirements at runtime.",
        "parameters": "133 classes, ~349 MB checkpoint",
        "input_size": "3D T1w MRI (NIfTI)",
        "default_train_config": "train.json",
        "default_inference_config": "inference.json",
    },
]


def list_supported_bundles() -> list:
    """Return the registry of MONAI bundles this integration supports."""
    return [{**b, "source_type": "monai_bundle"} for b in SUPPORTED_BUNDLES]


def bundle_card_from_metadata(meta: dict) -> dict:
    """Derive model-card info (papers + data organization) from a bundle's own
    configs/metadata.json — the authoritative MONAI Model Zoo source. Nothing here
    is invented; every field is read from metadata (empty/None when absent)."""
    meta = meta or {}

    # references: metadata stores a list of free-text citation strings (often with URLs).
    import re as _re
    papers = []
    for ref in (meta.get("references") or []):
        if not isinstance(ref, str):
            continue
        m = _re.search(r"https?://\S+", ref)
        url = m.group(0).rstrip(".,);") if m else None
        title = ref.replace(url, "").strip().rstrip(".,; ") if url else ref.strip()
        papers.append({"title": title, "url": url})

    ndf = meta.get("network_data_format", {}) or {}
    img = ((ndf.get("inputs") or {}).get("image") or {}) if isinstance(ndf.get("inputs"), dict) else {}
    pred = ((ndf.get("outputs") or {}).get("pred") or {}) if isinstance(ndf.get("outputs"), dict) else {}
    channel_def = pred.get("channel_def") or img.get("channel_def")

    modality = img.get("modality")
    num_channels = img.get("num_channels")
    modality_str = None
    if modality:
        modality_str = f"{modality}, 3D (NIfTI)" + (f", {num_channels} channel(s)" if num_channels else "")

    labels_str = None
    if isinstance(channel_def, dict) and channel_def:
        items = list(channel_def.items())
        labels_str = ", ".join(f"{k}:{v}" for k, v in items[:12]) + (" …" if len(items) > 12 else "")

    data_org = {
        "modality": modality_str,
        "labels": labels_str,
        "data_source": meta.get("data_source"),
        "data_type": meta.get("data_type"),
        # The training data layout our fine-tune pipeline expects (MSD/Decathlon).
        "layout": [
            "s3://<your-bucket>/<dataset>/",
            "├── dataset.json     # MSD format: \"training\": [{\"image\": ..., \"label\": ...}, ...]",
            "├── imagesTr/        # image volumes (.nii.gz)",
            "└── labelsTr/        # matching label volumes (.nii.gz)",
        ],
    }
    # Drop empty values so the UI only shows what the bundle actually declares.
    data_org = {k: v for k, v in data_org.items() if v}
    return {"papers": papers, "data_org": data_org or None}


def parse_s3_uri(uri: str):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI (expected s3://bucket/key): {uri}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URI must include a key: {uri}")
    return parsed.netloc, key


MODEL_ZOO_BASE = "https://github.com/Project-MONAI/model-zoo/tree/dev/models"


def resolve_bundle_source(value: str):
    """Accept a full GitHub model-zoo link, a Hugging Face URL, or a plain repo id
    and return (hf_repo_id, source_url, short_name).

    Bundles are downloaded from the Hugging Face `MONAI/<name>` mirror; the
    source_url is the canonical MONAI model-zoo GitHub page for the bundle.
    """
    value = (value or "").strip()
    if not value:
        raise ValueError("Provide a MONAI bundle GitHub link or repo id.")

    short = None
    if value.startswith("http://") or value.startswith("https://"):
        p = urlparse(value)
        host = p.netloc.lower()
        parts = [seg for seg in p.path.split("/") if seg]
        if "github.com" in host:
            # .../Project-MONAI/model-zoo/tree/<branch>/models/<name>[/...]
            if "models" in parts:
                i = parts.index("models")
                if i + 1 < len(parts):
                    short = parts[i + 1]
            if not short:
                raise ValueError(
                    "GitHub link must point to a bundle under model-zoo/models/<name>, "
                    "e.g. https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation"
                )
        elif "huggingface.co" in host:
            # huggingface.co/MONAI/<name>
            short = parts[-1] if parts else None
            if not short:
                raise ValueError("Could not parse a bundle name from the Hugging Face URL.")
        else:
            raise ValueError("Unsupported URL host. Use a github.com/Project-MONAI/model-zoo link.")
    elif "/" in value:
        # plain repo id like MONAI/spleen_ct_segmentation
        short = value.split("/")[-1]
    else:
        # bare bundle name
        short = value

    hf_repo_id = f"MONAI/{short}"
    source_url = f"{MODEL_ZOO_BASE}/{short}"
    return hf_repo_id, source_url, short


class MonaiBundleService:
    def __init__(self):
        self.jobs: dict = {}
        self.staged: dict = {}   # bundle name -> staged model.tar.gz S3 URI
        self._load_jobs()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load_jobs(self):
        if JOBS_FILE.exists():
            try:
                data = json.loads(JOBS_FILE.read_text())
                # Backward compatible: file may be the plain jobs dict, or a
                # wrapper {"jobs": ..., "staged": ...}.
                if isinstance(data, dict) and "jobs" in data and "staged" in data:
                    self.jobs = data.get("jobs", {})
                    self.staged = data.get("staged", {})
                else:
                    self.jobs = data
            except Exception as e:
                logger.warning("Could not load monai bundle jobs: %s", e)

    def _save_jobs(self):
        try:
            JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
            JOBS_FILE.write_text(json.dumps(
                {"jobs": self.jobs, "staged": self.staged}, indent=2, default=str))
        except Exception as e:
            logger.warning("Could not save monai bundle jobs: %s", e)

    # ── staging ──────────────────────────────────────────────────────────────
    def stage_bundle(self, repo_id: str, s3_uri: str, region: str = "us-east-1", boto_session=None) -> dict:
        """Download a MONAI Model Zoo bundle, package it, upload to S3.

        Reuses the generic bundle_utils/package logic from src/monai_bundle.
        """
        import sys
        sys.path.insert(0, BUNDLE_SRC_DIR)
        import bundle_utils  # type: ignore
        from huggingface_hub import snapshot_download

        parse_s3_uri(s3_uri)  # validate early

        # Accept a full GitHub model-zoo link, a Hugging Face URL, or a repo id.
        hf_repo_id, source_url, short = resolve_bundle_source(repo_id)

        dest = tempfile.mkdtemp(prefix="bundle_")
        path = snapshot_download(repo_id=hf_repo_id, cache_dir=dest)
        info = bundle_utils.inspect_bundle(path)
        if not info.is_valid:
            raise ValueError(f"{hf_repo_id} is not a valid MONAI bundle (no configs/ found)")

        # Stage the serving handler into the bundle's code/ dir. The SageMaker
        # PyTorch inference DLC discovers the handler at /opt/ml/model/code/
        # (SAGEMAKER_PROGRAM) and pip-installs code/requirements.txt at endpoint
        # startup. So we ship the handler + the bundle's own required packages
        # there (Option 1: per-bundle deps installed at container start — no
        # per-bundle image needed for pip-installable dependencies).
        code_dir = os.path.join(path, "code")
        os.makedirs(code_dir, exist_ok=True)
        for fn in ("inference.py", "bundle_utils.py"):
            shutil.copy2(os.path.join(BUNDLE_SRC_DIR, fn), os.path.join(code_dir, fn))

        # Merge our base handler reqs with the bundle's declared required packages
        # (metadata.json: required_packages_version). Bundle pins win on overlap.
        base_reqs = {}
        base_reqs_file = os.path.join(BUNDLE_SRC_DIR, "requirements-inference.txt")
        if os.path.exists(base_reqs_file):
            for line in open(base_reqs_file):
                line = line.strip()
                if line and not line.startswith("#"):
                    name = line.split("==")[0].split(">=")[0].split("<")[0].strip()
                    base_reqs[name] = line

        meta = info.metadata or {}
        # Older zoo bundles (e.g. HoVer-Net) declare deps under the deprecated
        # 'optional_packages_version' key instead of 'required_packages_version'.
        required = dict(meta.get("optional_packages_version") or {})
        required.update(meta.get("required_packages_version") or {})
        # Never merge framework-coupled packages: their pins track the bundle's
        # torch generation (e.g. torchvision==0.14.1 → torch 1.13) and would make
        # pip downgrade/break the DLC's torch at endpoint startup. The DLC ships
        # matched torch/torchvision/torchaudio already. (Same lesson class as
        # cucim: some declared deps must not be pip-installed at runtime.)
        FRAMEWORK_SKIP = ("torch", "torchvision", "torchaudio", "cucim", "cupy")
        skipped = []
        for pkg, ver in required.items():
            if pkg.lower().startswith(FRAMEWORK_SKIP):
                skipped.append(f"{pkg}=={ver}")
                continue
            base_reqs[pkg] = f"{pkg}=={ver}"

        with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
            f.write("# Auto-generated by MedAnvil stage_bundle: base handler deps +\n")
            f.write("# bundle metadata required_packages_version. Installed by the\n")
            f.write("# inference DLC at endpoint startup (Option 1).\n")
            for line in sorted(base_reqs.values()):
                f.write(line + "\n")

        # Advisory: framework versions the bundle expects vs. what the image ships.
        # A hard mismatch here (torch/CUDA) may need a dedicated image, not pip.
        deps_report = {
            "extra_packages_installed_at_startup": sorted(
                p for p in required.keys() if not p.lower().startswith(FRAMEWORK_SKIP)
            ),
            "framework_packages_skipped": sorted(skipped),
            "bundle_framework_versions": {
                k: meta.get(k) for k in ("monai_version", "pytorch_version", "numpy_version")
                if meta.get(k)
            },
        }

        # Advisory: how the fine-tuning dataset (the `data` channel) must be laid
        # out. Fine-tune generates a MONAI datalist from a Decathlon-style
        # dataset.json (see src/monai_bundle/datalist_gen.py). Enrich with the
        # bundle's declared input format when available.
        ndf = meta.get("network_data_format", {}) or {}
        inputs = ndf.get("inputs", {}) or {}
        outputs = ndf.get("outputs", {}) or {}
        # Fine-tuning is only meaningful when the bundle ships a training config.
        # And the MSD layout below is what MEDANVIL'S generic fine-tune pipeline
        # (datalist_gen.py) consumes — it only applies to Decathlon-style NIfTI
        # bundles. Bundles with other training-data formats (e.g. HoVer-Net's
        # CoNSeP .npy patches) get an honest pointer to their own docs instead
        # of a misleading one-size-fits-all tree.
        trainable = bool(info.train_configs)
        msd_compatible = str(meta.get("data_type", "")).lower() in ("nibabel", "nifti")

        # Interpret the data organization from the bundle's OWN train config —
        # the glob patterns it applies to dataset_dir are the ground truth of how
        # training data must be arranged (e.g. spleen: imagesTr/*.nii.gz;
        # HoVer-Net: Train/*image.npy triplets). This beats any generic template.
        import re as _re
        train_globs, uses_datalist_file = [], False
        train_cfg_name = info.default_train_config()
        if trainable and train_cfg_name:
            try:
                with open(os.path.join(path, "configs", train_cfg_name), "r",
                          encoding="utf-8") as f:
                    raw_train_cfg = f.read()
                train_globs = sorted(set(_re.findall(
                    r"glob\.glob\(\s*@dataset_dir\s*\+\s*'([^']+)'", raw_train_cfg)))
                uses_datalist_file = "data_list_file_path" in raw_train_cfg
            except Exception as e:  # noqa: BLE001 - YAML/unusual config
                logger.warning("Could not interpret %s for data layout: %s",
                               train_cfg_name, e)

        training_data_note = None
        if trainable and not msd_compatible:
            training_data_note = (
                f"This bundle uses its own training-data format (data_type: "
                f"{meta.get('data_type', 'unknown')}) — MedAnvil's generic MSD "
                "fine-tune layout does not apply. See the bundle's documentation "
                "for dataset preparation."
            )
        data_layout = None
        if train_globs:
            # Layout derived directly from the bundle's train config.
            data_layout = {
                "format": f"Derived from the bundle's {train_cfg_name}",
                "description": (
                    "File patterns the bundle's own training config globs from "
                    "the dataset directory (the SageMaker data channel). Arrange "
                    "your dataset so these patterns match."
                ),
                "layout": ["s3://<your-bucket>/<dataset>/   (becomes dataset_dir)"] + [
                    f"{p.lstrip('/')}   # required by {train_cfg_name}"
                    for p in train_globs
                ],
                "expected_inputs": inputs or None,
                "expected_outputs": outputs or None,
                "notes": ([training_data_note] if training_data_note else []) + (
                    ["The config also reads a datalist file (data_list_file_path); "
                     "MedAnvil can generate one from a Decathlon dataset.json "
                     "(generate_datalist=true)."] if uses_datalist_file else []),
            }
        elif trainable and msd_compatible:
            data_layout = {
            "format": "Medical Segmentation Decathlon (MSD) — MedAnvil fine-tune pipeline",
            "description": (
                "This is the dataset layout MedAnvil's generic fine-tune pipeline "
                "expects (not bundle-specific): a dataset directory with a "
                "dataset.json plus the image and label volumes it references. "
                "Fine-tuning reads dataset.json and auto-splits train/validation."
            ),
            "layout": [
                "s3://<your-bucket>/<dataset>/",
                "├── dataset.json        # MSD format: {\"training\": [{\"image\": \"imagesTr/x.nii.gz\", \"label\": \"labelsTr/x.nii.gz\"}, ...], \"labels\": {...}}",
                "├── imagesTr/           # training image volumes (.nii.gz)",
                "└── labelsTr/           # matching label volumes (.nii.gz)",
            ],
            "dataset_json_keys": ["training (required)", "labels (optional)"],
            "pair_keys": {"image": "path relative to dataset dir", "label": "path relative to dataset dir"},
            "expected_inputs": inputs or None,
            "expected_outputs": outputs or None,
            "notes": [
                "Paths in dataset.json are relative to the dataset directory (leading ./ or / is stripped).",
                "Validation is auto-split from `training` (default 20%) unless the bundle provides its own.",
                "Public MSD tasks (e.g. Task09_Spleen) already ship a compatible dataset.json.",
            ],
        }

        tar_path = os.path.join(dest, "model.tar.gz")
        # dereference symlinks so HF-cache blobs are included as real bytes
        with tarfile.open(tar_path, "w:gz", dereference=True) as tar:
            for name in sorted(os.listdir(path)):
                tar.add(os.path.join(path, name), arcname=name)

        import boto3
        bucket, key = parse_s3_uri(s3_uri)
        # Use the app's authenticated boto3 session (the one the dashboard
        # connected with) so the upload uses the SAME credentials as the rest
        # of the app. Falling back to a bare boto3.client() would resolve
        # credentials independently and can pick up a DIFFERENT identity
        # (e.g. ~/.aws/credentials) that lacks write access to the target bucket.
        if boto_session is not None:
            s3_client = boto_session.client("s3", region_name=region)
        else:
            s3_client = boto3.client("s3", region_name=region)
        s3_client.upload_file(tar_path, bucket, key)
        logger.info("Staged bundle %s -> %s", repo_id, s3_uri)

        # Track the staged artifact (keyed by short bundle name) so the deploy
        # page can offer this base bundle for deployment. Store metadata-derived
        # model-card info (papers + data organization) read from the bundle's own
        # configs/metadata.json — the authoritative Model Zoo source.
        card = bundle_card_from_metadata(meta)
        # Card layout preference: (1) derived from the bundle's own train config
        # globs, (2) the MSD tree only when MedAnvil's pipeline consumes it,
        # (3) none — with the honest note instead.
        data_org = card["data_org"]
        if isinstance(data_org, dict):
            if train_globs:
                data_org["layout"] = data_layout["layout"]
                data_org["layout_source"] = train_cfg_name
            elif not (trainable and msd_compatible):
                data_org = {k: v for k, v in data_org.items() if k != "layout"}
            if training_data_note:
                data_org["training_data_note"] = training_data_note
            data_org = data_org or None
        self.staged[short] = {
            "s3_uri": s3_uri,
            "repo_id": hf_repo_id,
            "task": info.task,
            "trainable": trainable,
            "description": meta.get("description"),
            "papers": card["papers"],
            "data_org": data_org,
            "source_url": source_url,
        }
        self._save_jobs()

        return {
            "repo_id": hf_repo_id,
            "model_artifacts": s3_uri,
            "task": info.task,
            "trainable": trainable,
            "train_configs": info.train_configs,
            "inference_configs": info.inference_configs,
            "default_train_config": info.default_train_config(),
            "default_inference_config": info.default_inference_config(),
            "deps": deps_report,
            "data_layout": data_layout,
            "papers": card["papers"],
            "data_org": data_org,
            "source_url": source_url,
        }

    # ── fine-tuning ───────────────────────────────────────────────────────────
    def start_finetune(
        self,
        job_name: str,
        model_artifacts: str,      # bundle model.tar.gz OR unpacked bundle prefix (the `model` channel)
        data_s3_uri: str,          # dataset (the `data` channel, Decathlon layout)
        role_arn: str,
        image_uri: str,            # the sm-monai-bundle training image
        sagemaker_client,
        region: str = "us-east-1",
        instance_type: str = "ml.g5.4xlarge",
        instance_count: int = 1,
        volume_size_gb: int = 100,
        max_runtime_seconds: int = 3600,
        train_config: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        generate_datalist: bool = True,
        output_s3_uri: Optional[str] = None,
    ) -> dict:
        """Launch a bundle fine-tune on SageMaker using the custom MONAI-bundle image.

        Uses the validated launch contract: the image's baked-in entry point is run
        via ContainerEntrypoint (python /opt/program/train.py) with channel paths,
        config_overrides, and generate_datalist passed as ContainerArguments. The
        bundle is the `model` channel; the dataset is the `data` channel. train.py
        auto-extracts a model.tar.gz and runs `monai.bundle run` on the bundle.
        """
        parse_s3_uri(model_artifacts)
        parse_s3_uri(data_s3_uri)
        if output_s3_uri:
            parse_s3_uri(output_s3_uri)

        # Refuse fine-tune for artifacts we know are inference-only (the bundle
        # ships no training config) — otherwise SageMaker would start a billable
        # job that is guaranteed to fail in train.py minutes later.
        for name, s in self.staged.items():
            rec = s if isinstance(s, dict) else {"s3_uri": s}
            if rec.get("s3_uri") == model_artifacts and rec.get("trainable") is False:
                raise ValueError(
                    f"Bundle '{name}' is inference-only (it ships no training "
                    "config) — fine-tuning is not available for this artifact."
                )

        overrides = dict(config_overrides or {})

        container_args = [
            "--bundle_channel", "/opt/ml/input/data/model",
            "--data_dir", "/opt/ml/input/data/data",
            "--model_dir", "/opt/ml/model",
            "--config_overrides", json.dumps(overrides),
            "--generate_datalist", "true" if generate_datalist else "false",
        ]
        if train_config:
            container_args += ["--train_config", train_config]

        if not job_name:
            from datetime import datetime
            job_name = "monai-bundle-" + datetime.utcnow().strftime("%Y%m%d-%H%M%S")

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
                {
                    "ChannelName": "model",
                    "DataSource": {"S3DataSource": {
                        "S3DataType": "S3Prefix", "S3Uri": model_artifacts,
                        "S3DataDistributionType": "FullyReplicated"}},
                },
                {
                    "ChannelName": "data",
                    "DataSource": {"S3DataSource": {
                        "S3DataType": "S3Prefix", "S3Uri": data_s3_uri,
                        "S3DataDistributionType": "FullyReplicated"}},
                },
            ],
            "OutputDataConfig": {"S3OutputPath": output_s3_uri} if output_s3_uri else None,
            "ResourceConfig": {
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
                "VolumeSizeInGB": volume_size_gb,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_seconds},
        }
        # OutputDataConfig is required by the API; default to a sibling of the data.
        if request["OutputDataConfig"] is None:
            bucket, _ = parse_s3_uri(data_s3_uri)
            request["OutputDataConfig"] = {"S3OutputPath": f"s3://{bucket}/monai-bundle-output/"}

        sagemaker_client.create_training_job(**request)

        info = {
            "job_name": job_name,
            "status": "InProgress",
            "model_artifacts": None,
            "config": {
                "bundle_artifacts": model_artifacts,
                "data_s3_uri": data_s3_uri,
                "image_uri": image_uri,
                "train_config": train_config,
                "config_overrides": overrides,
                "instance_type": instance_type,
                "instance_count": instance_count,
            },
        }
        self.jobs[job_name] = info
        self._save_jobs()
        logger.info("Started MONAI bundle fine-tune job: %s", job_name)
        return info

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
            self._save_jobs()
        return result

    def list_jobs(self) -> list:
        return [{"job_name": k, **v} for k, v in self.jobs.items()]

    def delete_staged(self, name: str, delete_artifact: bool = True,
                      region: str = "us-east-1") -> dict:
        """Remove a staged bundle record and (optionally) its S3 artifact.

        The registry entry (if any) is code and remains — the bundle reverts to
        an unstaged placeholder. Fine-tuned job artifacts are not touched.
        NOTE: deleting the S3 artifact breaks scale-up-from-zero for any
        endpoint deployed from it (new instances re-pull ModelDataUrl).
        """
        rec = self.staged.get(name)
        if rec is None:
            raise KeyError(f"No staged bundle named '{name}'")
        rec = rec if isinstance(rec, dict) else {"s3_uri": rec}
        artifact_deleted = False
        s3_uri = rec.get("s3_uri")
        if delete_artifact and s3_uri:
            try:
                import boto3
                bucket, key = parse_s3_uri(s3_uri)
                boto3.client("s3", region_name=region).delete_object(Bucket=bucket, Key=key)
                artifact_deleted = True
            except Exception as e:  # noqa: BLE001 - record removal still proceeds
                logger.warning("Could not delete artifact %s: %s", s3_uri, e)
        del self.staged[name]
        self._save_jobs()
        logger.info("Removed staged bundle %s (artifact_deleted=%s)", name, artifact_deleted)
        return {"name": name, "removed": True,
                "artifact_deleted": artifact_deleted, "s3_uri": s3_uri}

    def refresh_jobs(self, sagemaker_client) -> list:
        """Refresh status for non-terminal jobs from AWS, then return all jobs.

        Stored records freeze status at 'InProgress' when submitted; without this,
        completed/failed jobs keep showing as active. Mirrors the training service.
        """
        TERMINAL = {"Completed", "Failed", "Stopped"}
        if sagemaker_client:
            for job_name, info in self.jobs.items():
                if info.get("status") in TERMINAL:
                    continue
                try:
                    self.get_job_status(job_name, sagemaker_client)
                except Exception as e:  # job may be gone/inaccessible — keep stored value
                    logger.warning("Could not refresh status for %s: %s", job_name, e)
        return self.list_jobs()

    def list_deployable_artifacts(self) -> dict:
        """Return deployable model.tar.gz artifacts, grouped by source:
          - base: staged pretrained bundles (from stage_bundle, tracked here)
          - finetuned: completed fine-tune jobs that produced an artifact
        Used by the dashboard deploy page to let the user pick what to deploy.
        """
        base = []
        seen = set()

        # Case-insensitive lookup of registry metadata by bundle short-name.
        registry_by_name = {b["name"].lower(): b for b in SUPPORTED_BUNDLES}

        def _norm(staged):
            # Back-compat: older records stored a bare S3 URI string.
            if isinstance(staged, str):
                return {"s3_uri": staged}
            return staged or {}

        # 1) Every actually-staged bundle — keyed by whatever name it was staged
        #    under (any case, and bundles outside the registry). This is the fix
        #    for staged bundles "disappearing" due to name/case mismatch.
        for name, staged in self.staged.items():
            s = _norm(staged)
            uri = s.get("s3_uri")
            if not uri:
                continue
            reg = registry_by_name.get(name.lower(), {})
            # trainable: from the staged record (metadata-derived); for records
            # staged before the flag existed, fall back to the registry (a null
            # default_train_config marks an inference-only bundle).
            trainable = s.get("trainable")
            if trainable is None:
                trainable = reg.get("default_train_config", "train.json") is not None
            base.append({
                "name": name,
                "task": s.get("task") or reg.get("task"),
                "description": s.get("description") or reg.get("description"),
                "trainable": trainable,
                # Papers + data organization from the bundle's own metadata.json.
                "papers": s.get("papers", []),
                "data_org": s.get("data_org"),
                "source_url": s.get("source_url"),
                "model_artifacts": uri,
                "deployable": True,
            })
            seen.add(name.lower())

        # 2) Registry bundles that have not been staged yet — shown as placeholders.
        for b in SUPPORTED_BUNDLES:
            if b["name"].lower() in seen:
                continue
            base.append({
                "name": b["name"],
                "task": b.get("task"),
                "description": b.get("description"),
                "papers": [],
                "data_org": None,
                "source_url": None,
                "model_artifacts": None,
                "deployable": False,
            })
        finetuned = []
        for name, j in self.jobs.items():
            art = j.get("model_artifacts")
            if art and j.get("status") == "Completed":
                cfg = j.get("config") or {}
                finetuned.append({
                    "name": name,
                    "task": cfg.get("task"),
                    "model_artifacts": art,
                    "data_s3_uri": cfg.get("data_s3_uri"),
                    "instance_type": cfg.get("instance_type"),
                    "bundle_artifacts": cfg.get("bundle_artifacts"),
                    "deployable": True,
                })
        return {"base": base, "finetuned": finetuned}

    # ── deployment (async only) ───────────────────────────────────────────────
    def deploy_bundle(
        self,
        endpoint_name: str,
        model_artifacts: str,      # trained bundle model.tar.gz (from a completed job)
        image_uri: str,            # the sm-monai-bundle-INFERENCE image
        role_arn: str,
        sagemaker_session,
        output_s3_uri: str,        # REQUIRED — async endpoints write results here
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
        max_concurrent: int = 4,
    ) -> dict:
        """Deploy a trained MONAI bundle artifact to a SageMaker ASYNC endpoint.

        MONAI bundle inference is 3D/volumetric — slow and large-payload — so
        deployment is async-only (no real-time). Reuses DeploymentService's async
        path. The dedicated inference image (Dockerfile.monai-bundle-inference)
        serves via inference.py -> `monai.bundle run` on the bundle's inference
        config; the servable model.tar.gz carries code/inference.py + the bundle.
        """
        from app.services.deployment_service import DeploymentService

        parse_s3_uri(model_artifacts)
        if not output_s3_uri:
            raise ValueError("output_s3_uri is required (async endpoints write results to S3)")
        parse_s3_uri(output_s3_uri)

        config = {
            "endpoint_name": endpoint_name,
            "model_artifacts": model_artifacts,
            "image_uri": image_uri,
            "inference_script": "inference.py",   # discovered at code/inference.py in the artifact
            "instance_type": instance_type,
            "instance_count": instance_count,
            "async_endpoint": True,
            "output_s3_uri": output_s3_uri,
            "max_concurrent": max_concurrent,
        }
        return DeploymentService().create_endpoint(
            config=config, sagemaker_session=sagemaker_session, role_arn=role_arn
        )
