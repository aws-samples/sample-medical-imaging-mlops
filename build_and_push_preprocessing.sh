#!/bin/bash
# Build and push the CPU-only preprocessing image for SageMaker Processing Jobs.
#
# Usage: ./build_and_push_preprocessing.sh [--build|--push|--no-cache]
#   ./build_and_push_preprocessing.sh                # build (with cache) and push
#   ./build_and_push_preprocessing.sh --build        # build only
#   ./build_and_push_preprocessing.sh --push         # push only
#   ./build_and_push_preprocessing.sh --no-cache     # build from scratch and push
#
# Reads AWS_ACCOUNT_ID and region from .env.local.
#
# The image runs src/preprocessing/preprocess.py, which is data-driven by a
# spec.json staged into the Processing Job's code channel at runtime. CPU-only:
# no GPU, no Deep Learning Container base.
#
# NOTE: DOCKER_BUILDKIT=0 forces the Docker V2 manifest format SageMaker requires
# (it rejects OCI manifests).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_LOCAL="$SCRIPT_DIR/.env.local"

if [[ -f "$ENV_LOCAL" ]]; then
    eval "$(grep -E '^(AWS_ACCOUNT_ID|AWS_DEFAULT_REGION)=' "$ENV_LOCAL")"
fi

AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ECR_ACCOUNT="${ECR_ACCOUNT:-${AWS_ACCOUNT_ID:-}}"
if [[ -z "${ECR_ACCOUNT:-}" ]]; then
    echo "❌ Error: AWS_ACCOUNT_ID not set. Add it to .env.local or export ECR_ACCOUNT."
    exit 1
fi
echo "Using ECR account: $ECR_ACCOUNT (region $AWS_REGION)"

DO_BUILD=true
DO_PUSH=true
NO_CACHE=false
for arg in "$@"; do
    case "$arg" in
        --build)    DO_BUILD=true; DO_PUSH=false ;;
        --push)     DO_BUILD=false; DO_PUSH=true ;;
        --no-cache) NO_CACHE=true ;;
        *)          echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

ECR_REPO="sm-preprocessing"
IMAGE="${ECR_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"

if [[ "$DO_BUILD" == true ]]; then
    CACHE_FLAG=()
    if [[ "$NO_CACHE" == true ]]; then CACHE_FLAG=(--no-cache); fi
    echo "Building preprocessing image: $IMAGE"
    # Base image (python:3.12-slim) is public — no DLC registry login needed.
    DOCKER_BUILDKIT=0 docker build "${CACHE_FLAG[@]}" --platform linux/amd64 \
      -f Dockerfile.preprocessing -t "$IMAGE" .
fi

if [[ "$DO_PUSH" == true ]]; then
    echo "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" \
      | docker login --username AWS --password-stdin \
        "${ECR_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    echo "Creating ECR repository (if not exists)..."
    aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" 2>/dev/null || true

    echo "Pushing image..."
    docker push "$IMAGE"

    echo "Verifying manifest format..."
    MEDIA_TYPE=$(aws ecr batch-get-image \
      --repository-name "$ECR_REPO" \
      --region "$AWS_REGION" \
      --image-ids imageTag=latest \
      --query 'images[0].imageManifestMediaType' \
      --output text 2>/dev/null || echo "unknown")
    if [[ "$MEDIA_TYPE" == *"oci"* ]]; then
        echo "⚠️  WARNING: Image has OCI manifest ($MEDIA_TYPE). SageMaker may reject it."
        echo "   Run: docker system prune -a --force ; then re-run with --no-cache."
        exit 1
    else
        echo "✅ Manifest format: $MEDIA_TYPE"
    fi
fi

echo "Done. Image URI: $IMAGE"
