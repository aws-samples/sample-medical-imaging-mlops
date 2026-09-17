#!/bin/bash
# Build and push the MONAI Bundle INFERENCE image (for SageMaker endpoints).
#
# Usage: ./build_and_push_monai_bundle_inference.sh [--build|--push|--no-cache]
# Derives the ECR account from the active AWS credentials (STS), region from
# .env.local or environment. Override the account with ECR_ACCOUNT if needed.
#
# Separate from the training image: serving needs the PyTorch inference DLC
# (serve/nginx/gunicorn). The bundle model.tar.gz is supplied at runtime via the
# endpoint's ModelDataUrl.
#
# NOTE: DOCKER_BUILDKIT=0 for a Docker V2 manifest (SageMaker rejects OCI). On
# Docker 24+/containerd stores, disable the containerd image store first:
#   echo '{ "features": { "containerd-snapshotter": false } }' | sudo tee /etc/docker/daemon.json
#   sudo systemctl restart docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_LOCAL="$SCRIPT_DIR/.env.local"
if [[ -f "$ENV_LOCAL" ]]; then
    eval "$(grep -E '^(AWS_ACCOUNT_ID|AWS_DEFAULT_REGION)=' "$ENV_LOCAL")"
fi

AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# ECR account comes from .env.local's AWS_ACCOUNT_ID (override with ECR_ACCOUNT).
ECR_ACCOUNT="${ECR_ACCOUNT:-${AWS_ACCOUNT_ID:-}}"
if [[ -z "${ECR_ACCOUNT:-}" ]]; then
    echo "❌ Error: AWS_ACCOUNT_ID not set. Add it to .env.local or export ECR_ACCOUNT."
    exit 1
fi
echo "Using ECR account: $ECR_ACCOUNT (region $AWS_REGION)"

DO_BUILD=true; DO_PUSH=true; NO_CACHE=false
for arg in "$@"; do
    case "$arg" in
        --build)    DO_BUILD=true; DO_PUSH=false ;;
        --push)     DO_BUILD=false; DO_PUSH=true ;;
        --no-cache) NO_CACHE=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

ECR_REPO="sm-monai-bundle-inference"
IMAGE="${ECR_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"
DLC_REGISTRY="763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com"

if [[ "$DO_BUILD" == true ]]; then
    echo "Logging in to AWS Deep Learning Containers registry..."
    aws ecr get-login-password --region "$AWS_REGION" \
      | docker login --username AWS --password-stdin "$DLC_REGISTRY"

    CACHE_FLAG=""
    [[ "$NO_CACHE" == true ]] && CACHE_FLAG="--no-cache"
    echo "Building MONAI bundle inference image: $IMAGE"
    DOCKER_BUILDKIT=0 docker build $CACHE_FLAG --platform linux/amd64 \
      -f Dockerfile.monai-bundle-inference -t "$IMAGE" .
fi

if [[ "$DO_PUSH" == true ]]; then
    echo "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" \
      | docker login --username AWS --password-stdin \
        "${ECR_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    echo "Creating ECR repository (if not exists)..."
    aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" 2>/dev/null || true

    echo "Deleting old image tag (if exists)..."
    aws ecr batch-delete-image --repository-name "$ECR_REPO" --region "$AWS_REGION" \
      --image-ids imageTag=latest 2>/dev/null || true

    echo "Pushing image..."
    docker push "$IMAGE"

    echo "Verifying manifest format..."
    MEDIA_TYPE=$(aws ecr batch-get-image --repository-name "$ECR_REPO" --region "$AWS_REGION" \
      --image-ids imageTag=latest --query 'images[0].imageManifestMediaType' --output text 2>/dev/null || echo "unknown")
    if [[ "$MEDIA_TYPE" == *"oci"* ]]; then
        echo "⚠️  OCI manifest ($MEDIA_TYPE) — SageMaker may reject. Disable containerd image store and rebuild --no-cache."
        exit 1
    else
        echo "✅ Manifest format: $MEDIA_TYPE"
    fi
fi

echo "Done. Image URI: $IMAGE"
