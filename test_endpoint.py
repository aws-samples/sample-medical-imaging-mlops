"""
Endpoint test script based on deploy_realtime.ipynb.
Uploads test image to SageMaker default bucket before invoking (matches notebook behaviour).

Usage:
    python test_endpoint.py
    python test_endpoint.py --endpoint test-endp-classification
    python test_endpoint.py --endpoint test-endp-classification --local path/to/image.dcm
"""
import json
import argparse
import os
import tempfile
import boto3
import sagemaker
from botocore.config import Config

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_ENDPOINT   = "test-endp"
DEFAULT_SOURCE_IMG = os.environ.get(
    "TEST_IMAGE_S3_URI",
    "s3://your-bucket/path/to/test/image.dcm",
)
DEFAULT_REGION = "us-east-1"


# ── Predictor (from deploy_realtime.ipynb) ────────────────────────────────────
class MedicalImagingPredictor:
    def __init__(self, endpoint_name, region="us-east-1"):
        self.endpoint_name  = endpoint_name
        self.runtime_client = boto3.client(
            "sagemaker-runtime",
        )
        self.labels = [
            'Surgical_implant', 'Vertebral_collapse', 'Spondylolysthesis',
            'No_finding', 'Foraminal_stenosis', 'Other_lesions',
            'Disc_space_narrowing', 'Osteophytes'
        ]

    def predict(self, s3_image_uri):
        payload  = json.dumps({"file_path": s3_image_uri})
        response = self.runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        return self._parse_response(response)

    def predict_local(self, local_path):
        with open(local_path, "rb") as f:
            image_bytes = f.read()
        response = self.runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/dicom",
            Body=image_bytes,
        )
        return self._parse_response(response)

    def _parse_response(self, response):
        result = json.loads(response["Body"].read().decode("utf-8"))
        return {
            "predicted_class": result["predicted_class"],
            "confidence":      result["confidence"],
            "probabilities":   result["probabilities"],
        }


def print_result(result, title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
    print(f"  Predicted : {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"\n  Probabilities:")
    for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {label:<30} {bar:<30} {prob:.4f}")
    print("=" * 60)


def upload_to_default_bucket(source_s3_uri, region, s3_client, default_bucket):
    """Download from source S3 and re-upload to SageMaker default bucket."""
    parts  = source_s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    fname  = key.split("/")[-1]
    dest_key = f"test-inference/samples/{fname}"

    print(f"Downloading {source_s3_uri} ...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
        tmp.flush()
        tmp_path = tmp.name
    s3_client.download_file(bucket, key, tmp_path)

    print(f"Uploading to s3://{default_bucket}/{dest_key} ...")
    s3_client.upload_file(tmp_path, default_bucket, dest_key)
    os.unlink(tmp_path)
    return f"s3://{default_bucket}/{dest_key}"


# ── Main ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
parser.add_argument("--image",    default=DEFAULT_SOURCE_IMG, help="Source S3 URI (will be re-uploaded to default bucket)")
parser.add_argument("--local",    default=None,               help="Local DICOM file path")
parser.add_argument("--region",   default=DEFAULT_REGION)
args = parser.parse_args()

print(f"Endpoint : {args.endpoint}")
print(f"Region   : {args.region}")

sess           = sagemaker.Session()
default_bucket = sess.default_bucket()
s3_client      = boto3.client("s3", region_name=args.region)
predictor      = MedicalImagingPredictor(args.endpoint, region=args.region)

# Upload test image to SageMaker default bucket (same as notebook)
test_s3_uri = upload_to_default_bucket(args.image, args.region, s3_client, default_bucket)
print(f"\nTest image URI: {test_s3_uri}")

# S3 inference
print("\nInvoking endpoint with S3 URI...")
# result = predictor.predict(test_s3_uri)
# print_result(result, "S3 Inference Result")

# Local file inference
if args.local:
    print(f"\nInvoking endpoint with local file: {args.local}")
    result_local = predictor.predict_local(args.local)
    print_result(result_local, "Local File Inference Result")
