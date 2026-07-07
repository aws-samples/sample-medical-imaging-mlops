#!/bin/bash
# Run SageMaker endpoint inference integration tests
#
# Usage:
#   ./tests/integration/run_inference_tests.sh [local_image_path]
#
# Examples:
#   ./tests/integration/run_inference_tests.sh
#   ./tests/integration/run_inference_tests.sh /path/to/image.dcm
#   ENDPOINT_NAME=my-endpoint ./tests/integration/run_inference_tests.sh

set -e

# ── Config (override via env vars) ────────────────────────────────────────────
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export ENDPOINT_NAME="${ENDPOINT_NAME:-test-endp}"
export TEST_IMAGE_S3_URI="${TEST_IMAGE_S3_URI:-s3://your-bucket/path/to/test/image.dcm}"

# Optional: pass a local image file as first argument
if [ -n "$1" ]; then
    export TEST_LOCAL_IMAGE="$1"
    echo "Using local image: $TEST_LOCAL_IMAGE"
fi

echo "================================================"
echo " SageMaker Endpoint Inference Tests"
echo "================================================"
echo " Region       : $AWS_DEFAULT_REGION"
echo " Endpoint     : $ENDPOINT_NAME"
echo " S3 image     : $TEST_IMAGE_S3_URI"
echo " Local image  : ${TEST_LOCAL_IMAGE:-<synthetic>}"
echo "================================================"
echo ""

# ── Activate venv if present ──────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# ── Run tests ─────────────────────────────────────────────────────────────────
pytest tests/integration/test_local_inference.py -v -s
