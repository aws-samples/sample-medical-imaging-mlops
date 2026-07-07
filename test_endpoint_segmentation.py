
"""
Segmentation endpoint test script.
Invokes a deployed SageMaker segmentation endpoint with a NIfTI volume and
displays the returned binary mask statistics.

Supported input modes:
  --s3-uri   s3://bucket/subject/img.nii.gz   (JSON payload, recommended)
  --local    /path/to/volume.nii.gz           (raw bytes, application/octet-stream)
  --b64      /path/to/volume.nii.gz           (base64-encoded JSON payload)

Usage:
    # S3 URI input (default)
    python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
        --s3-uri s3://your-bucket/dataset/val/subject_001/img.nii.gz

    # Local file (raw bytes)
    python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
        --local /path/to/volume.nii.gz

    # Local file via base64 JSON
    python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
        --b64 /path/to/volume.nii.gz

    # Save the returned mask to disk
    python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
        --s3-uri s3://bucket/img.nii.gz --save-mask /tmp/mask.nii.gz
"""
import argparse
import base64
import json
import os
import sys
import tempfile
import time

import boto3
import numpy as np
from botocore.config import Config

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_ENDPOINT = "my-seg-endpoint"
DEFAULT_REGION   = "us-east-1"
# Override via TEST_SEGMENTATION_S3_URI env var or --s3-uri flag
DEFAULT_S3_URI   = os.environ.get(
    "TEST_SEGMENTATION_S3_URI",
    "s3://your-bucket/path/to/test/volume.nii.gz",
)

# 3D segmentation inference is slow — allow up to 5 minutes
INVOKE_TIMEOUT_S = 300


# ── Predictor ─────────────────────────────────────────────────────────────────
class SegmentationPredictor:
    def __init__(self, endpoint_name: str, region: str = "us-east-1"):
        self.endpoint_name  = endpoint_name
        self.runtime_client = boto3.client(
            "sagemaker-runtime",
            region_name=region,
            config=Config(
                read_timeout=INVOKE_TIMEOUT_S,
                connect_timeout=10,
                retries={"max_attempts": 1},  # don't retry on timeout
            ),
        )

    def predict_s3(self, s3_uri: str) -> dict:
        """Send S3 URI as JSON payload."""
        payload  = json.dumps({"s3_uri": s3_uri})
        t0       = time.time()
        response = self.runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        elapsed = time.time() - t0
        result  = json.loads(response["Body"].read().decode("utf-8"))
        result["_latency_s"] = round(elapsed, 3)
        return result

    def predict_local_raw(self, local_path: str) -> dict:
        """Send raw NIfTI bytes (application/octet-stream)."""
        with open(local_path, "rb") as f:
            raw = f.read()
        t0       = time.time()
        response = self.runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/octet-stream",
            Body=raw,
        )
        elapsed = time.time() - t0
        result  = json.loads(response["Body"].read().decode("utf-8"))
        result["_latency_s"] = round(elapsed, 3)
        return result

    def predict_local_b64(self, local_path: str) -> dict:
        """Send base64-encoded NIfTI as JSON payload."""
        with open(local_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        payload  = json.dumps({"file_path": encoded})
        t0       = time.time()
        response = self.runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        elapsed = time.time() - t0
        result  = json.loads(response["Body"].read().decode("utf-8"))
        result["_latency_s"] = round(elapsed, 3)
        return result


# ── Output helpers ────────────────────────────────────────────────────────────
def print_result(result: dict, title: str = ""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
    print(f"  Latency        : {result.get('_latency_s', '?')} s")
    print(f"  Mask shape     : {result.get('shape', '?')}")
    print(f"  Positive voxels: {result.get('voxels_positive', '?'):,}")
    print(f"  Total voxels   : {result.get('voxels_total', '?'):,}")
    vol_frac = result.get("volume_fraction", None)
    if vol_frac is not None:
        bar = "█" * int(vol_frac * 50)
        print(f"  Volume fraction: {bar} {vol_frac:.4f} ({vol_frac * 100:.2f}%)")
    if result.get("voxels_positive", 0) == 0:
        print("\n  ⚠  No positive voxels — model predicted empty mask.")
        print("     Check: correct model loaded? Input HU range correct?")
    print("=" * 60)


def save_mask(result: dict, output_path: str):
    """Decode base64 NIfTI mask and save to disk."""
    try:
        import nibabel as nib
        import io
        raw  = base64.b64decode(result["mask_nifti_b64"])
        # nibabel can load from a file-like object via a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        nii = nib.load(tmp_path)
        nib.save(nii, output_path)
        os.unlink(tmp_path)
        print(f"\n  Mask saved to: {output_path}")
    except ImportError:
        # nibabel not installed locally — just write raw bytes
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(result["mask_nifti_b64"]))
        print(f"\n  Raw mask bytes saved to: {output_path}")


def check_endpoint_status(endpoint_name: str, region: str):
    """Verify the endpoint is InService before invoking."""
    sm = boto3.client("sagemaker", region_name=region)
    try:
        resp   = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print(f"Endpoint status: {status}")
        if status != "InService":
            print(f"  ⚠  Endpoint is not InService (status={status}). Aborting.")
            sys.exit(1)
    except sm.exceptions.ClientError as e:
        print(f"  ✗  Could not describe endpoint '{endpoint_name}': {e}")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Test a SageMaker segmentation endpoint")
parser.add_argument("--endpoint",  default=DEFAULT_ENDPOINT, help="SageMaker endpoint name")
parser.add_argument("--region",    default=DEFAULT_REGION,   help="AWS region")
parser.add_argument("--s3-uri",    default=None,             help="S3 URI of a NIfTI volume")
parser.add_argument("--local",     default=None,             help="Local NIfTI file (raw bytes)")
parser.add_argument("--b64",       default=None,             help="Local NIfTI file (base64 JSON)")
parser.add_argument("--save-mask", default=None,             help="Save returned mask to this path (.nii.gz)")
parser.add_argument("--timeout",   default=INVOKE_TIMEOUT_S, type=int, help=f"Read timeout in seconds (default {INVOKE_TIMEOUT_S})")
args = parser.parse_args()

print(f"Endpoint : {args.endpoint}")
print(f"Region   : {args.region}")
print(f"Timeout  : {args.timeout}s")

# Verify endpoint is ready
check_endpoint_status(args.endpoint, args.region)

predictor = SegmentationPredictor(args.endpoint, region=args.region)
# Override timeout if specified
predictor.runtime_client = boto3.client(
    "sagemaker-runtime",
    region_name=args.region,
    config=Config(
        read_timeout=args.timeout,
        connect_timeout=10,
        retries={"max_attempts": 1},
    ),
)

# ── Run inference ─────────────────────────────────────────────────────────────
if args.local:
    if not os.path.exists(args.local):
        print(f"  ✗  File not found: {args.local}")
        sys.exit(1)
    print(f"\nInvoking with local file (raw bytes): {args.local}")
    result = predictor.predict_local_raw(args.local)
    print_result(result, "Local File — Raw Bytes")

elif args.b64:
    if not os.path.exists(args.b64):
        print(f"  ✗  File not found: {args.b64}")
        sys.exit(1)
    print(f"\nInvoking with local file (base64 JSON): {args.b64}")
    result = predictor.predict_local_b64(args.b64)
    print_result(result, "Local File — Base64 JSON")

else:
    s3_uri = args.s3_uri or DEFAULT_S3_URI
    print(f"\nInvoking with S3 URI: {s3_uri}")
    result = predictor.predict_s3(s3_uri)
    print_result(result, "S3 URI Inference")

# ── Save mask ─────────────────────────────────────────────────────────────────
if args.save_mask and result.get("mask_nifti_b64"):
    save_mask(result, args.save_mask)
elif args.save_mask:
    print("  ⚠  No mask_nifti_b64 in response — cannot save mask.")
