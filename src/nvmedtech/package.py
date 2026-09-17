"""
Package and stage the local NV-Segment-CT bundle to S3 for SageMaker.

Assembles a complete, self-contained model.tar.gz from:
  - the local bundle folder (NVIDIA-MedTech/NV-Segment-CT: configs/ scripts/ ...)
  - the pretrained VISTA-3D checkpoint, fetched from Hugging Face
    (nvidia/NV-Segment-CT, as declared by the bundle's own inference.json)
    into models/model.pt — so training/inference never download at runtime.

Usage:
  python -m src.nvmedtech.package \
      --bundle_dir NVIDIA-MedTech/NV-Segment-CT \
      --s3_uri s3://<bucket>/<prefix>/NV-Segment-CT/model.tar.gz

⚠️ RESEARCH USE ONLY — not for clinical diagnosis.
"""
import argparse
import json
import logging
import os
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_s3_uri(uri: str):
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI (expected s3://bucket/key): {uri}")
    return p.netloc, p.path.lstrip("/")


def fetch_checkpoint(bundle_dir: str):
    """Download the pretrained checkpoint the bundle declares (HF) into
    models/model.pt, if not already present."""
    target = os.path.join(bundle_dir, "models", "model.pt")
    if os.path.isfile(target):
        logger.info("Checkpoint already present: %s", target)
        return
    cfg_path = os.path.join(bundle_dir, "configs", "inference.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    hf_repo = cfg.get("huggingface_repo_id")
    hf_file = cfg.get("huggingface_checkpoint_file")
    if not (hf_repo and hf_file):
        raise ValueError("Bundle config declares no Hugging Face checkpoint")
    from huggingface_hub import hf_hub_download
    logger.info("Fetching %s from %s ...", hf_file, hf_repo)
    got = hf_hub_download(repo_id=hf_repo, filename=hf_file)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(got, target)  # real bytes, not a cache symlink
    logger.info("Checkpoint -> models/model.pt")


def package_and_upload(bundle_dir: str, s3_uri: str, region: str = "us-east-1") -> str:
    bucket, key = parse_s3_uri(s3_uri)
    if not os.path.isfile(os.path.join(bundle_dir, "configs", "metadata.json")):
        raise ValueError(f"{bundle_dir} is not a MONAI bundle (no configs/metadata.json)")

    fetch_checkpoint(bundle_dir)

    with tempfile.TemporaryDirectory() as tmp:
        # Inject the serving handler at code/inference.py — the SageMaker
        # inference DLC discovers handlers at /opt/ml/model/code/. No
        # code/requirements.txt: all deps are baked in the dedicated image
        # (sm-nvmedtech-inference), so endpoint cold start skips pip entirely.
        staging = os.path.join(tmp, "bundle")
        shutil.copytree(bundle_dir, staging,
                        ignore=shutil.ignore_patterns("example"))
        code_dir = os.path.join(staging, "code")
        os.makedirs(code_dir, exist_ok=True)
        handler_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.py")
        shutil.copy2(handler_src, os.path.join(code_dir, "inference.py"))

        tar_path = os.path.join(tmp, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz", dereference=True) as tar:
            for name in sorted(os.listdir(staging)):
                tar.add(os.path.join(staging, name), arcname=name)
        import boto3
        boto3.client("s3", region_name=region).upload_file(tar_path, bucket, key)
    logger.info("Staged %s -> %s", bundle_dir, s3_uri)
    return s3_uri


def main():
    parser = argparse.ArgumentParser(description="Stage NV-Segment-CT to S3")
    parser.add_argument("--bundle_dir", default="NVIDIA-MedTech/NV-Segment-CT")
    parser.add_argument("--s3_uri", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    args = parser.parse_args()
    package_and_upload(args.bundle_dir, args.s3_uri, args.region)


if __name__ == "__main__":
    main()
