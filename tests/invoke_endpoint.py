#!/usr/bin/env python3
"""
Invoke a SageMaker endpoint with a local image file.

Usage:
    python tests/invoke_endpoint.py --image /path/to/image.dcm
    python tests/invoke_endpoint.py --image /path/to/image.dcm --endpoint my-endpoint
    python tests/invoke_endpoint.py --image /path/to/image.dcm --endpoint my-endpoint --region us-east-1
"""
import argparse
import base64
import json
import os
import sys
import boto3


def invoke(endpoint_name: str, region: str, image_path: str = None, s3_uri: str = None) -> dict:
    if s3_uri:
        payload = json.dumps({"s3_uri": s3_uri})
        print(f"Using S3 URI: {s3_uri}")
    elif image_path:
        if not os.path.isfile(image_path):
            print(f"Error: file not found: {image_path}")
            sys.exit(1)
        print(f"Reading: {image_path}")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = json.dumps({
            "file_path": b64,
            "filename": os.path.basename(image_path),
        })
    else:
        print("Error: provide --image or --s3-uri")
        sys.exit(1)

    client = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=boto3.session.Config(connect_timeout=10, read_timeout=120),
    )

    print(f"Invoking endpoint: {endpoint_name} ({region})")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    return result


def print_result(result: dict):
    print("\n── Result ──────────────────────────────────────")
    print(f"Predicted class : {result.get('predicted_class', result.get('predicted_class_index', 'N/A'))}")
    print(f"Confidence      : {result.get('confidence', 0):.1%}")
    if "probabilities" in result:
        print("Probabilities   :")
        for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {cls:<30} {bar:<30} {prob:.1%}")
    print("────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="Invoke a SageMaker endpoint with a local image or S3 URI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",   help="Path to local image file (DICOM, PNG, JPG)")
    group.add_argument("--s3-uri",  help="S3 URI of the image (s3://bucket/key)")
    parser.add_argument("--endpoint", default=os.environ.get("ENDPOINT_NAME", "test-endp"), help="SageMaker endpoint name")
    parser.add_argument("--region",   default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"), help="AWS region")
    parser.add_argument("--output",   default=None, help="Optional path to save JSON result")
    parser.add_argument("--async",    dest="async_mode", action="store_true", help="Use async endpoint invocation")
    parser.add_argument("--bucket",   default=os.environ.get("OUTPUT_BUCKET"), help="S3 bucket for async input/output (required for --async with --image)")
    args = parser.parse_args()

    if args.async_mode:
        if args.s3_uri:
            result = invoke_async(args.s3_uri, args.endpoint, args.region)
        else:
            if not args.bucket:
                print("Error: --bucket is required for async invocation with a local file")
                sys.exit(1)
            result = upload_and_invoke_async(args.image, args.endpoint, args.region, args.bucket)
        print(f"\n── Async Result ─────────────────────────────────")
        print(f"Inference ID    : {result.get('inference_id')}")
        print(f"Output location : {result.get('output_location')}")
        print(f"Status          : {result.get('status')}")
        print(f"─────────────────────────────────────────────────")
    else:
        result = invoke(args.endpoint, args.region, image_path=args.image, s3_uri=args.s3_uri)
        print_result(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {args.output}")


if __name__ == "__main__":
    main()


def invoke_async(input_s3_uri: str, endpoint_name: str, region: str) -> dict:
    """Submit an async inference request and return the output S3 location."""
    client = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=boto3.session.Config(connect_timeout=10, read_timeout=30),
    )
    print(f"Submitting async request to endpoint: {endpoint_name} ({region})")
    print(f"Input: {input_s3_uri}")
    response = client.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=input_s3_uri,
        ContentType="application/json",
        Accept="application/json",
    )
    return {
        "inference_id":    response.get("InferenceId"),
        "output_location": response.get("OutputLocation"),
        "status":          "Submitted",
    }


def upload_and_invoke_async(image_path: str, endpoint_name: str, region: str, output_bucket: str) -> dict:
    """Upload a local file to S3 as JSON payload, then submit async inference."""
    import uuid
    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    print(f"Reading: {image_path}")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({
        "file_path": b64,
        "filename":  os.path.basename(image_path),
    })

    # Upload payload to S3
    s3 = boto3.client("s3", region_name=region)
    key = f"async-inputs/{endpoint_name}/{uuid.uuid4()}.json"
    s3.put_object(Bucket=output_bucket, Key=key, Body=payload.encode(), ContentType="application/json")
    input_s3_uri = f"s3://{output_bucket}/{key}"
    print(f"Uploaded input to: {input_s3_uri}")

    return invoke_async(input_s3_uri, endpoint_name, region)
