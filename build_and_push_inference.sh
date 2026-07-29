#!/bin/bash
# Build and push the SageMaker inference image.
# Usage: ./build_and_push_inference.sh [--build|--push] [classification|segmentation]
# Examples:
#   ./build_and_push_inference.sh classification          # build and push
#   ./build_and_push_inference.sh --build classification   # build only
#   ./build_and_push_inference.sh --push classification    # push only
#
# Reads AWS_ACCOUNT_ID and region from .env.local

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
TASK="classification"

for arg in "$@"; do
    case "$arg" in
        --build) DO_BUILD=true; DO_PUSH=false ;;
        --push)  DO_BUILD=false; DO_PUSH=true ;;
        *)       TASK="$arg" ;;
    esac
done

AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ECR_REPO="sm-inference-byoc-${TASK}"
IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"

if [[ "$DO_BUILD" == true ]]; then
    echo "Building $TASK inference image: $IMAGE"
    # Disable BuildKit to ensure Docker V2 manifest format (SageMaker rejects OCI manifests)
    DOCKER_BUILDKIT=0 docker build --no-cache --platform linux/amd64 -f Dockerfile.inference --build-arg TASK="$TASK" -t "$IMAGE" .
fi

if [[ "$DO_PUSH" == true ]]; then
    echo "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" \
      | docker login --username AWS --password-stdin \
        "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    echo "Creating ECR repository (if not exists)..."
    aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" 2>/dev/null || true

    echo "Pushing image..."
    docker push "$IMAGE"
fi

echo "Done. Image URI: $IMAGE"
