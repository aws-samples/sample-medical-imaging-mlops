"""
Repackage model.tar.gz to include inference code.

SageMaker's standard PyTorch serving container discovers custom inference code
at code/inference.py inside the extracted model.tar.gz. This script downloads
the existing archive, adds the code/ directory, and re-uploads it.

Usage:
    python repackage_model.py \
        --model-s3-uri s3://medml-dashboard/model-output/segmentation/segmentation-segresnet-20260729-1559/output/model.tar.gz

    # Or specify a custom inference script location:
    python repackage_model.py \
        --model-s3-uri s3://bucket/path/model.tar.gz \
        --inference-dir src/inference/segmentation
"""
import argparse
import os
import shutil
import tarfile
import tempfile
import sys

import boto3
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local", override=True)

DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
DEFAULT_INFERENCE_DIR = "src/inference/segmentation"


def repackage(model_s3_uri: str, inference_dir: str, region: str):
    """Download model.tar.gz, add code/ directory, and re-upload."""
    s3 = boto3.client("s3", region_name=region)

    # Parse S3 URI
    parts = model_s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    print(f"{'='*60}")
    print(f"  Repackaging model.tar.gz")
    print(f"{'='*60}")
    print(f"  S3 URI:        {model_s3_uri}")
    print(f"  Inference dir: {inference_dir}")
    print(f"{'='*60}\n")

    # Verify inference files exist
    inference_py = os.path.join(inference_dir, "inference.py")
    requirements_txt = os.path.join(inference_dir, "requirements.txt")
    if not os.path.exists(inference_py):
        print(f"❌ Not found: {inference_py}")
        sys.exit(1)
    if not os.path.exists(requirements_txt):
        print(f"⚠️  No requirements.txt found at {requirements_txt} — skipping")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar = os.path.join(tmpdir, "model.tar.gz")
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir)

        # Download
        print("📋 Downloading model.tar.gz...")
        s3.download_file(bucket, key, local_tar)
        print(f"   ✅ Downloaded ({os.path.getsize(local_tar) / 1024 / 1024:.1f} MB)")

        # Extract
        print("📋 Extracting...")
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(extract_dir)

        # List existing contents
        contents = os.listdir(extract_dir)
        print(f"   Existing contents: {contents}")

        # Add code/ directory
        code_dir = os.path.join(extract_dir, "code")
        os.makedirs(code_dir, exist_ok=True)

        shutil.copy2(inference_py, os.path.join(code_dir, "inference.py"))
        print(f"   ✅ Added code/inference.py")

        if os.path.exists(requirements_txt):
            shutil.copy2(requirements_txt, os.path.join(code_dir, "requirements.txt"))
            print(f"   ✅ Added code/requirements.txt")

        # Repackage
        print("📋 Repackaging...")
        repackaged_tar = os.path.join(tmpdir, "model_repackaged.tar.gz")
        with tarfile.open(repackaged_tar, "w:gz") as tar:
            for item in os.listdir(extract_dir):
                tar.add(os.path.join(extract_dir, item), arcname=item)

        new_size = os.path.getsize(repackaged_tar) / 1024 / 1024
        print(f"   ✅ Repackaged ({new_size:.1f} MB)")

        # Verify code/ is in the archive
        with tarfile.open(repackaged_tar, "r:gz") as tar:
            names = tar.getnames()
            assert "code/inference.py" in names, "code/inference.py not in repackaged archive!"
            print(f"   ✅ Verified: code/inference.py present in archive")

        # Upload (overwrite original)
        print(f"📋 Uploading to {model_s3_uri}...")
        s3.upload_file(repackaged_tar, bucket, key)
        print(f"   ✅ Uploaded")

    print(f"\n{'='*60}")
    print(f"  ✅ Done! model.tar.gz now contains code/inference.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repackage model.tar.gz with inference code")
    parser.add_argument("--model-s3-uri", required=True, help="S3 URI of model.tar.gz")
    parser.add_argument("--inference-dir", default=DEFAULT_INFERENCE_DIR,
                        help=f"Local directory with inference.py and requirements.txt (default: {DEFAULT_INFERENCE_DIR})")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    args = parser.parse_args()

    repackage(args.model_s3_uri, args.inference_dir, args.region)
