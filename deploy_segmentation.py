"""
SageMaker real-time deployment script for 3D medical image segmentation.
Mirrors the pattern from notebooks/05_sagemaker_deployment/deploy_realtime.ipynb
adapted for segmentation (NIfTI input → binary mask output).

Usage:
    # Deploy
    python deploy_segmentation.py --model-artifacts s3://bucket/path/model.tar.gz

    # Deploy with custom endpoint name
    python deploy_segmentation.py \
        --model-artifacts s3://bucket/path/model.tar.gz \
        --endpoint-name my-aorta-seg-endpoint

    # Test an already-deployed endpoint (skip deploy)
    python deploy_segmentation.py \
        --endpoint-name aorta-segmentation \
        --skip-deploy \
        --s3-uri s3://bucket/val/subject_001/img.nii.gz

    # Test with a local NIfTI file
    python deploy_segmentation.py \
        --endpoint-name aorta-segmentation \
        --skip-deploy \
        --local /path/to/volume.nii.gz \
        --save-mask /tmp/mask.nii.gz

    # Cleanup (delete endpoint, config, model)
    python deploy_segmentation.py \
        --endpoint-name aorta-segmentation \
        --cleanup
"""
import argparse
import base64
import json
import os
import sys
import tempfile
import time

from dotenv import load_dotenv
load_dotenv()
load_dotenv(".env.local", override=True)

import boto3
import sagemaker
from botocore.config import Config

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ENDPOINT      = "aorta-segmentation"
DEFAULT_REGION        = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
DEFAULT_INSTANCE_TYPE = "ml.g5.xlarge"
DEFAULT_ROLE_ARN      = os.environ.get("SAGEMAKER_ROLE_ARN", "")
INFERENCE_IMAGE_URI   = os.environ.get(
    "INFERENCE_IMAGE_SEGMENTATION",
    f"{os.environ.get('AWS_ACCOUNT_ID', 'YOUR_ACCOUNT_ID')}.dkr.ecr.{os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')}.amazonaws.com/sm-inference-byoc-segmentation:latest"
)
# SAGEMAKER_PROGRAM is set via env at deploy time — inference.py is bundled
# inside model.tar.gz by train.py, so no source_dir upload is needed.
INVOKE_TIMEOUT_S      = 300   # 3D segmentation can be slow on first call


# ── Deploy ────────────────────────────────────────────────────────────────────

def deploy(model_artifacts: str, endpoint_name: str, role_arn: str,
           instance_type: str, region: str):
    """Deploy a segmentation model.tar.gz to a SageMaker real-time endpoint."""
    sm = boto3.client("sagemaker", region_name=region)

    # Check if endpoint already exists
    existing = sm.list_endpoints()["Endpoints"]
    if any(ep["EndpointName"] == endpoint_name for ep in existing):
        print(f"Endpoint '{endpoint_name}' already exists — skipping deploy.")
        print("Use --skip-deploy to test it, or --cleanup to delete it first.")
        return

    model_name  = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    print(f"Deploying segmentation model...")
    print(f"  Model artifacts : {model_artifacts}")
    print(f"  Endpoint name   : {endpoint_name}")
    print(f"  Instance type   : {instance_type}")
    print(f"  Inference image : {INFERENCE_IMAGE_URI}")

    # Create model — inference.py is inside model.tar.gz, SAGEMAKER_PROGRAM tells
    # the container which script to use (same pattern as classification)
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image":        INFERENCE_IMAGE_URI,
            "ModelDataUrl": model_artifacts,
            "Environment": {
                "SAGEMAKER_PROGRAM":             "inference.py",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            },
        },
        ExecutionRoleArn=role_arn,
    )
    print(f"  Created model: {model_name}")

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName":          "AllTraffic",
            "ModelName":            model_name,
            "InstanceType":         instance_type,
            "InitialInstanceCount": 1,
        }],
    )
    print(f"  Created endpoint config: {config_name}")

    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    print(f"  Creating endpoint '{endpoint_name}' — waiting for InService...")

    # Wait for endpoint to be ready
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 30, "MaxAttempts": 40})
    print(f"\nEndpoint '{endpoint_name}' is InService.")


# ── Predictor ─────────────────────────────────────────────────────────────────

class SegmentationPredictor:
    def __init__(self, endpoint_name: str, region: str = DEFAULT_REGION,
                 timeout: int = INVOKE_TIMEOUT_S):
        self.endpoint_name  = endpoint_name
        self.runtime_client = boto3.client(
            "sagemaker-runtime",
            region_name=region,
            config=Config(
                read_timeout=timeout,
                connect_timeout=10,
                retries={"max_attempts": 1},
            ),
        )

    def predict_s3(self, s3_uri: str) -> dict:
        """Send S3 URI as JSON payload — endpoint downloads the NIfTI from S3."""
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
        """Send raw NIfTI bytes directly (application/octet-stream)."""
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
    print(f"  Positive voxels: {result.get('voxels_positive', 0):,}")
    print(f"  Total voxels   : {result.get('voxels_total', 0):,}")
    vol_frac = result.get("volume_fraction")
    if vol_frac is not None:
        bar = "█" * int(vol_frac * 50)
        print(f"  Volume fraction: {bar} {vol_frac:.4f} ({vol_frac * 100:.2f}%)")
    if result.get("voxels_positive", 0) == 0:
        print("\n  ⚠  Empty mask — check model weights and input HU range.")
    print("=" * 60)


def save_mask(result: dict, output_path: str):
    """Decode base64 NIfTI mask and save to disk."""
    if "mask_nifti_b64" not in result:
        print("  ⚠  No mask_nifti_b64 in response.")
        return
    raw = base64.b64decode(result["mask_nifti_b64"])
    try:
        import nibabel as nib
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        nii = nib.load(tmp_path)
        nib.save(nii, output_path)
        os.unlink(tmp_path)
    except ImportError:
        with open(output_path, "wb") as f:
            f.write(raw)
    print(f"  Mask saved → {output_path}")


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup(endpoint_name: str, region: str):
    """Delete endpoint, endpoint config, and model."""
    sm = boto3.client("sagemaker", region_name=region)
    try:
        desc        = sm.describe_endpoint(EndpointName=endpoint_name)
        config_name = desc["EndpointConfigName"]
        config_desc = sm.describe_endpoint_config(EndpointConfigName=config_name)
        model_name  = config_desc["ProductionVariants"][0]["ModelName"]

        sm.delete_endpoint(EndpointName=endpoint_name)
        print(f"Deleted endpoint       : {endpoint_name}")

        sm.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"Deleted endpoint config: {config_name}")

        sm.delete_model(ModelName=model_name)
        print(f"Deleted model          : {model_name}")
    except Exception as e:
        print(f"Cleanup error: {e}")


# ── Endpoint status ───────────────────────────────────────────────────────────

def check_endpoint_status(endpoint_name: str, region: str):
    sm = boto3.client("sagemaker", region_name=region)
    try:
        resp   = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print(f"Endpoint status: {status}")
        if status != "InService":
            print(f"  ⚠  Endpoint not InService (status={status}). Aborting.")
            sys.exit(1)
    except Exception as e:
        print(f"  ✗  Could not describe endpoint '{endpoint_name}': {e}")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Deploy and test a SageMaker segmentation endpoint")
parser.add_argument("--endpoint-name",   default=DEFAULT_ENDPOINT)
parser.add_argument("--region",          default=DEFAULT_REGION)
parser.add_argument("--model-artifacts", default=None,
                    help="S3 URI of model.tar.gz (required unless --skip-deploy or --cleanup)")
parser.add_argument("--role-arn",        default=DEFAULT_ROLE_ARN)
parser.add_argument("--instance-type",   default=DEFAULT_INSTANCE_TYPE)
parser.add_argument("--skip-deploy",     action="store_true",
                    help="Skip deployment and go straight to inference test")
parser.add_argument("--cleanup",         action="store_true",
                    help="Delete the endpoint, config, and model then exit")
parser.add_argument("--s3-uri",          default=None,
                    help="S3 URI of a NIfTI volume to test with")
parser.add_argument("--local",           default=None,
                    help="Local NIfTI file to test with (raw bytes)")
parser.add_argument("--b64",             default=None,
                    help="Local NIfTI file to test with (base64 JSON)")
parser.add_argument("--save-mask",       default=None,
                    help="Save returned mask to this .nii.gz path")
parser.add_argument("--timeout",         default=INVOKE_TIMEOUT_S, type=int,
                    help=f"Invoke read timeout in seconds (default {INVOKE_TIMEOUT_S})")
args = parser.parse_args()

print(f"Endpoint : {args.endpoint_name}")
print(f"Region   : {args.region}")

# ── Cleanup mode ──────────────────────────────────────────────────────────────
if args.cleanup:
    cleanup(args.endpoint_name, args.region)
    sys.exit(0)

# ── Deploy ────────────────────────────────────────────────────────────────────
if not args.skip_deploy:
    if not args.model_artifacts:
        print("Error: --model-artifacts is required unless --skip-deploy or --cleanup is set.")
        sys.exit(1)
    deploy(
        model_artifacts=args.model_artifacts,
        endpoint_name=args.endpoint_name,
        role_arn=args.role_arn,
        instance_type=args.instance_type,
        region=args.region,
    )

# ── Inference test ────────────────────────────────────────────────────────────
check_endpoint_status(args.endpoint_name, args.region)
predictor = SegmentationPredictor(args.endpoint_name, region=args.region, timeout=args.timeout)

result = None

if args.local:
    if not os.path.exists(args.local):
        print(f"  ✗  File not found: {args.local}")
        sys.exit(1)
    print(f"\nTesting with local file (raw bytes): {args.local}")
    result = predictor.predict_local_raw(args.local)
    print_result(result, "Local File — Raw Bytes")

elif args.b64:
    if not os.path.exists(args.b64):
        print(f"  ✗  File not found: {args.b64}")
        sys.exit(1)
    print(f"\nTesting with local file (base64 JSON): {args.b64}")
    result = predictor.predict_local_b64(args.b64)
    print_result(result, "Local File — Base64 JSON")

elif args.s3_uri:
    print(f"\nTesting with S3 URI: {args.s3_uri}")
    result = predictor.predict_s3(args.s3_uri)
    print_result(result, "S3 URI Inference")

else:
    print("\nNo test input provided (--s3-uri / --local / --b64). Deployment complete.")

if result and args.save_mask:
    save_mask(result, args.save_mask)
