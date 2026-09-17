"""
Package a MONAI bundle directory into a SageMaker `model.tar.gz` and (optionally)
upload it to S3, so it can feed MedAnvil's training/deployment flows.

An existing S3 `model.tar.gz` can be used directly as the training `model`
channel / deployment artifact without repackaging — use this only for local
bundle directories.

RESEARCH USE ONLY.
"""
import argparse
import logging
import os
import tarfile
import tempfile
from urllib.parse import urlparse

from bundle_utils import inspect_bundle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_s3_uri(uri: str):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI (expected s3://bucket/key): {uri}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URI must include a key: {uri}")
    return parsed.netloc, key


def package_bundle(bundle_dir: str, tar_path: str) -> str:
    """Validate and tar a bundle directory, preserving its layout."""
    info = inspect_bundle(bundle_dir)
    if not info.is_valid:
        raise ValueError(
            f"{bundle_dir} is not a valid MONAI bundle "
            f"(no configs/ with config files found)."
        )
    logger.info("Packaging bundle %s -> %s", bundle_dir, tar_path)
    with tarfile.open(tar_path, "w:gz", dereference=True) as tar:
        for name in sorted(os.listdir(bundle_dir)):
            tar.add(os.path.join(bundle_dir, name), arcname=name)
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    logger.info("Created %s (%.1f MB)", tar_path, size_mb)
    return tar_path


def upload_s3(tar_path: str, s3_uri: str, region: str):
    import boto3

    bucket, key = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3", region_name=region)
    logger.info("Uploading -> %s", s3_uri)
    s3.upload_file(tar_path, bucket, key)
    logger.info("Upload complete: %s", s3_uri)


def main():
    parser = argparse.ArgumentParser(description="Package a MONAI bundle to model.tar.gz")
    parser.add_argument("--bundle_dir", required=True)
    parser.add_argument("--s3_uri", default=None, help="Optional destination model.tar.gz S3 URI")
    parser.add_argument("--out", default=None, help="Local output path (default: temp)")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    args = parser.parse_args()

    if args.s3_uri:
        parse_s3_uri(args.s3_uri)  # validate early

    tmp = None
    tar_path = args.out
    if not tar_path:
        tmp = tempfile.mkdtemp()
        tar_path = os.path.join(tmp, "model.tar.gz")

    try:
        package_bundle(args.bundle_dir, tar_path)
        if args.s3_uri:
            upload_s3(tar_path, args.s3_uri, args.region)
            logger.info("Done. model channel = %s", args.s3_uri)
        else:
            logger.info("Done. local artifact = %s", tar_path)
    except Exception:
        # Cleanup partial local artifact on failure.
        if tar_path and os.path.isfile(tar_path):
            os.unlink(tar_path)
        raise
    finally:
        if tmp and not args.s3_uri and args.out:
            pass  # keep user-requested local output


if __name__ == "__main__":
    main()
