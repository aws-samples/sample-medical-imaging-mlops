"""
Deploy and test a SageMaker segmentation endpoint.

This script:
  1. Deploys a segmentation model to a SageMaker async endpoint
  2. Waits for the endpoint to become InService
  3. Invokes the endpoint with a test NIfTI volume
  4. Polls for the async result
  5. Displays segmentation statistics

Usage:
    # Deploy and test
    python deploy_and_test_segmentation.py \
        --model-data s3://your-bucket/output/model.tar.gz \
        --test-image s3://test-imaging-workflows/segmentation/test/10090344/img.nii.gz

    # Test an existing endpoint (skip deployment)
    python deploy_and_test_segmentation.py \
        --endpoint segmentation \
        --test-image s3://test-imaging-workflows/segmentation/test/10090344/img.nii.gz \
        --skip-deploy

    # Deploy only (no test)
    python deploy_and_test_segmentation.py \
        --model-data s3://your-bucket/output/model.tar.gz \
        --skip-test
"""
import argparse
import json
import os
import sys
import time

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local", override=True)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
DEFAULT_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "")
DEFAULT_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
DEFAULT_ENDPOINT_NAME = "segmentation"
DEFAULT_INSTANCE_TYPE = "ml.g5.xlarge"
DEFAULT_TEST_IMAGE = os.environ.get(
    "TEST_SEGMENTATION_S3_URI",
    "s3://test-imaging-workflows/segmentation/test/10090344/img.nii.gz",
)
DEFAULT_OUTPUT_BUCKET = "medml-dashboard"
DEFAULT_OUTPUT_PREFIX = "model-output/segmentation"

POLL_INTERVAL = 30  # seconds between status checks
DEPLOY_TIMEOUT = 900  # 15 minutes max wait for endpoint creation (GPU instances need 8-12 min)
INFERENCE_TIMEOUT = 900  # 15 minutes max wait for async result (includes potential cold start)


# ── Deploy ────────────────────────────────────────────────────────────────────
def deploy_endpoint(
    endpoint_name: str,
    model_data: str,
    image_uri: str,
    role_arn: str,
    instance_type: str,
    region: str,
) -> str:
    """Deploy an async segmentation endpoint."""
    sm = boto3.client("sagemaker", region_name=region)

    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    print(f"\n{'='*60}")
    print(f"  Deploying Segmentation Endpoint")
    print(f"{'='*60}")
    print(f"  Endpoint:      {endpoint_name}")
    print(f"  Model data:    {model_data}")
    print(f"  Image URI:     {image_uri}")
    print(f"  Instance type: {instance_type}")
    print(f"  Role:          {role_arn}")
    print(f"{'='*60}\n")

    # Step 1: Create Model
    print("📋 Step 1: Creating model...")
    try:
        sm.delete_model(ModelName=model_name)
        print(f"   Deleted existing model '{model_name}'")
    except sm.exceptions.ClientError:
        pass

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "MODEL_NAME": "segresnet",
                "SAGEMAKER_MODEL_SERVER_TIMEOUT": "300",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
        },
        ExecutionRoleArn=role_arn,
    )
    print(f"   ✅ Model created: {model_name}")

    # Step 2: Create Endpoint Config (async)
    print("📋 Step 2: Creating endpoint config (async)...")
    try:
        sm.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"   Deleted existing config '{config_name}'")
    except sm.exceptions.ClientError:
        pass

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
            }
        ],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": f"s3://{DEFAULT_OUTPUT_BUCKET}/{DEFAULT_OUTPUT_PREFIX}",
            },
            "ClientConfig": {
                "MaxConcurrentInvocationsPerInstance": 2,
            },
        },
    )
    print(f"   ✅ Endpoint config created: {config_name}")

    # Step 3: Create or update endpoint
    print("📋 Step 3: Creating endpoint...")
    try:
        existing = sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"   Endpoint exists (status: {existing['EndpointStatus']}). Updating...")
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    except sm.exceptions.ClientError:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
        print(f"   ✅ Endpoint creation initiated: {endpoint_name}")

    # Step 4: Wait for InService
    print("📋 Step 4: Waiting for endpoint to become InService...")
    start = time.time()
    while time.time() - start < DEPLOY_TIMEOUT:
        resp = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        elapsed = int(time.time() - start)
        print(f"   [{elapsed}s] Status: {status}")

        if status == "InService":
            print(f"\n   ✅ Endpoint is InService! ({elapsed}s)")
            return endpoint_name
        elif status == "Failed":
            reason = resp.get("FailureReason", "Unknown")
            print(f"\n   ❌ Endpoint failed: {reason}")
            sys.exit(1)

        time.sleep(POLL_INTERVAL)

    print(f"\n   ❌ Timed out after {DEPLOY_TIMEOUT}s")
    sys.exit(1)


# ── Test (Async) ──────────────────────────────────────────────────────────────
def test_endpoint_async(endpoint_name: str, test_image_s3: str, region: str) -> dict:
    """Invoke the async endpoint and poll for results."""
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    print(f"\n{'='*60}")
    print(f"  Testing Async Segmentation Endpoint")
    print(f"{'='*60}")
    print(f"  Endpoint:   {endpoint_name}")
    print(f"  Test image: {test_image_s3}")
    print(f"{'='*60}\n")

    # Upload input JSON to S3
    # inference.py expects {"file_path": "<s3_uri_or_local_path>"}
    input_payload = json.dumps({"file_path": test_image_s3})
    input_key = f"{DEFAULT_OUTPUT_PREFIX}/test-input/request_{int(time.time())}.json"

    s3.put_object(
        Bucket=DEFAULT_OUTPUT_BUCKET,
        Key=input_key,
        Body=input_payload.encode("utf-8"),
        ContentType="application/json",
    )
    input_location = f"s3://{DEFAULT_OUTPUT_BUCKET}/{input_key}"
    print(f"📋 Uploaded input to: {input_location}")

    # Invoke async
    print("📋 Invoking endpoint (async)...")
    t0 = time.time()
    response = runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=input_location,
        ContentType="application/json",
        Accept="application/json",
    )

    inference_id = response.get("InferenceId")
    output_location = response.get("OutputLocation")
    print(f"   Inference ID: {inference_id}")
    print(f"   Output:       {output_location}")

    # Poll for result
    print("📋 Waiting for result...")
    output_parts = output_location.replace("s3://", "").split("/", 1)
    output_bucket = output_parts[0]
    output_key = output_parts[1]

    # Also check for failures — SageMaker writes errors to a parallel failure path
    failure_key = output_key.replace(
        DEFAULT_OUTPUT_PREFIX, f"{DEFAULT_OUTPUT_PREFIX}/failures"
    )

    while time.time() - t0 < INFERENCE_TIMEOUT:
        # Check for success
        try:
            obj = s3.get_object(Bucket=output_bucket, Key=output_key)
            result_str = obj["Body"].read().decode("utf-8")
            result = json.loads(result_str)
            elapsed = round(time.time() - t0, 1)
            print(f"   ✅ Result received ({elapsed}s)")
            return result
        except s3.exceptions.NoSuchKey:
            pass

        # Check for failure
        try:
            obj = s3.get_object(Bucket=output_bucket, Key=failure_key)
            error_str = obj["Body"].read().decode("utf-8")
            elapsed = round(time.time() - t0, 1)
            print(f"\n   ❌ Inference FAILED ({elapsed}s)")
            print(f"   Error: {error_str[:500]}")
            sys.exit(1)
        except s3.exceptions.NoSuchKey:
            pass

        elapsed = int(time.time() - t0)
        print(f"   [{elapsed}s] Waiting...")
        time.sleep(10)

    print(f"\n   ❌ Timed out after {INFERENCE_TIMEOUT}s waiting for result")
    print(f"   Check CloudWatch logs for endpoint '{endpoint_name}' for details.")
    sys.exit(1)


def test_endpoint_realtime(endpoint_name: str, test_image_s3: str, region: str) -> dict:
    """Try real-time invocation (for real-time endpoints)."""
    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=Config(read_timeout=300, connect_timeout=10, retries={"max_attempts": 1}),
    )

    print(f"\n{'='*60}")
    print(f"  Testing Real-Time Segmentation Endpoint")
    print(f"{'='*60}")
    print(f"  Endpoint:   {endpoint_name}")
    print(f"  Test image: {test_image_s3}")
    print(f"{'='*60}\n")

    payload = json.dumps({"file_path": test_image_s3})
    t0 = time.time()

    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
        elapsed = round(time.time() - t0, 1)
        result = json.loads(response["Body"].read().decode("utf-8"))
        print(f"   ✅ Result received ({elapsed}s)")
        return result
    except runtime.exceptions.ClientError as e:
        if "does not support this inference type" in str(e):
            print("   Endpoint is async — switching to async invocation...")
            return test_endpoint_async(endpoint_name, test_image_s3, region)
        raise


def print_result(result: dict):
    """Display segmentation results.

    Expected keys from inference.py:
      - model_name, original_shape, segmentation_shape,
        segmented_voxels, total_voxels, segmentation_percentage,
        segmentation_mask_base64
      - On error: error, model_name
    """
    print(f"\n{'='*60}")
    print(f"  Segmentation Results")
    print(f"{'='*60}")

    if "error" in result:
        print(f"  ❌ Inference error: {result['error']}")
        print(f"  Model: {result.get('model_name', 'unknown')}")
        print(f"{'='*60}")
        return

    print(f"  Model           : {result.get('model_name', '?')}")
    print(f"  Original shape  : {result.get('original_shape', '?')}")
    print(f"  Seg. shape      : {result.get('segmentation_shape', '?')}")
    print(f"  Segmented voxels: {result.get('segmented_voxels', '?'):,}")
    print(f"  Total voxels    : {result.get('total_voxels', '?'):,}")

    seg_pct = result.get("segmentation_percentage", None)
    if seg_pct is not None:
        bar = "█" * int(seg_pct / 2)
        print(f"  Segmentation %  : {bar} {seg_pct:.2f}%")

    if result.get("segmented_voxels", 0) == 0:
        print("\n  ⚠  No positive voxels — model predicted empty mask.")
        print("     Possible causes:")
        print("     - Model not fully trained")
        print("     - Input image outside training distribution")
        print("     - Wrong intensity range")
    else:
        print("\n  ✅ Segmentation successful!")

    if result.get("segmentation_mask_base64"):
        mask_size = len(result["segmentation_mask_base64"])
        print(f"  Mask payload    : {mask_size:,} chars (base64)")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy and test a SageMaker segmentation endpoint")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT_NAME, help="Endpoint name")
    parser.add_argument("--model-data", default=None, help="S3 URI of model.tar.gz")
    parser.add_argument("--image-uri", default=None, help="ECR inference image URI")
    parser.add_argument("--role-arn", default=DEFAULT_ROLE_ARN, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="Instance type")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--test-image", default=DEFAULT_TEST_IMAGE, help="S3 URI of test NIfTI volume")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip deployment, test existing endpoint")
    parser.add_argument("--skip-test", action="store_true", help="Deploy only, skip testing")
    args = parser.parse_args()

    # Resolve image URI
    if not args.image_uri and DEFAULT_ACCOUNT_ID:
        args.image_uri = f"{DEFAULT_ACCOUNT_ID}.dkr.ecr.{args.region}.amazonaws.com/sm-inference-byoc-segmentation:latest"

    # Deploy
    if not args.skip_deploy:
        if not args.model_data:
            print("❌ --model-data is required for deployment (S3 URI of model.tar.gz)")
            sys.exit(1)
        if not args.role_arn:
            print("❌ --role-arn is required (or set SAGEMAKER_ROLE_ARN in .env.local)")
            sys.exit(1)
        if not args.image_uri:
            print("❌ --image-uri is required (or set AWS_ACCOUNT_ID in .env.local)")
            sys.exit(1)

        deploy_endpoint(
            endpoint_name=args.endpoint,
            model_data=args.model_data,
            image_uri=args.image_uri,
            role_arn=args.role_arn,
            instance_type=args.instance_type,
            region=args.region,
        )

    # Test
    if not args.skip_test:
        result = test_endpoint_realtime(args.endpoint, args.test_image, args.region)
        print_result(result)
