#!/bin/bash
# Build and push the SageMaker inference image.
# Usage: ./build_and_push_inference.sh [--build|--push|--no-cache] [classification|segmentation]
# Examples:
#   ./build_and_push_inference.sh classification             # build (with cache) and push
#   ./build_and_push_inference.sh --build classification      # build only (with cache)
#   ./build_and_push_inference.sh --push classification       # push only
#   ./build_and_push_inference.sh --no-cache classification   # build from scratch and push
#
# Reads AWS_ACCOUNT_ID and region from .env.local
#
# NOTE: SageMaker CreateModel rejects OCI image manifests. This script uses
# DOCKER_BUILDKIT=0 and deletes the old ECR image before pushing to ensure
# Docker V2 manifest format. See docs/configuration.md for details.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_LOCAL="$SCRIPT_DIR/.env.local"

if [[ -f "$ENV_LOCAL" ]]; then
    eval "$(grep -E '^(AWS_ACCOUNT_ID|AWS_DEFAULT_REGION)=' "$ENV_LOCAL")"
fi

if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
    echo "❌ Error: AWS_ACCOUNT_ID not set. Add it to .env.local or export it."
    exit 1
fi

# Parse flags
DO_BUILD=true
DO_PUSH=true
NO_CACHE=false
TASK="classification"

for arg in "$@"; do
    case "$arg" in
        --build)    DO_BUILD=true; DO_PUSH=false ;;
        --push)     DO_BUILD=false; DO_PUSH=true ;;
        --no-cache) NO_CACHE=true ;;
        *)          TASK="$arg" ;;
    esac
done

AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ECR_REPO="sm-inference-byoc-${TASK}"
IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"
DLC_REGISTRY="763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com"

if [[ "$DO_BUILD" == true ]]; then
    echo "Logging in to AWS Deep Learning Containers registry..."
    aws ecr get-login-password --region "$AWS_REGION" \
      | docker login --username AWS --password-stdin "$DLC_REGISTRY"

    CACHE_FLAG=""
    if [[ "$NO_CACHE" == true ]]; then
        CACHE_FLAG="--no-cache"
        echo "Building $TASK inference image (no cache): $IMAGE"
    else
        echo "Building $TASK inference image (with cache): $IMAGE"
    fi

    # Disable BuildKit to ensure Docker V2 manifest format (SageMaker rejects OCI manifests)
    DOCKER_BUILDKIT=0 docker build $CACHE_FLAG --platform linux/amd64 -f Dockerfile.inference --build-arg TASK="$TASK" -t "$IMAGE" .
fi

if [[ "$DO_PUSH" == true ]]; then
    echo "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" \
      | docker login --username AWS --password-stdin \
        "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    echo "Creating ECR repository (if not exists)..."
    aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" 2>/dev/null || true

    # Delete existing image to prevent OCI manifest from persisting
    echo "Deleting old image from ECR (if exists)..."
    aws ecr batch-delete-image \
      --repository-name "$ECR_REPO" \
      --region "$AWS_REGION" \
      --image-ids imageTag=latest 2>/dev/null || true

    echo "Pushing image..."
    docker push "$IMAGE"

    # Verify manifest format
    echo "Verifying manifest format..."
    MEDIA_TYPE=$(aws ecr batch-get-image \
      --repository-name "$ECR_REPO" \
      --region "$AWS_REGION" \
      --image-ids imageTag=latest \
      --query 'images[0].imageManifestMediaType' \
      --output text 2>/dev/null || echo "unknown")

    if [[ "$MEDIA_TYPE" == *"oci"* ]]; then
        echo "⚠️  WARNING: Image has OCI manifest ($MEDIA_TYPE). SageMaker CreateModel will reject this."
        echo "   Run: docker system prune -a --force"
        echo "   Then re-run this script."
        exit 1
    else
        echo "✅ Manifest format: $MEDIA_TYPE"
    fi
fi

echo "Done. Image URI: $IMAGE"
