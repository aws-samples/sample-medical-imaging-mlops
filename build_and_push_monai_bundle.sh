#!/bin/bash
# Build and push the MONAI Bundle training/inference image.
#
# Usage: ./build_and_push_monai_bundle.sh [--build|--push|--no-cache]
#   ./build_and_push_monai_bundle.sh                # build (with cache) and push
#   ./build_and_push_monai_bundle.sh --build        # build only (with cache)
#   ./build_and_push_monai_bundle.sh --push         # push only
#   ./build_and_push_monai_bundle.sh --no-cache     # build from scratch and push
#
# Reads AWS_ACCOUNT_ID and region from .env.local.
#
# The image runs ANY MONAI bundle via `python -m monai.bundle run`. The bundle
# (configs/ models/ scripts/) is supplied at RUNTIME via the SageMaker `model`
# channel (model.tar.gz), so this single image serves any supported bundle.
#
# ── Expected SageMaker channel arrangement at RUNTIME ────────────────────────
#   model channel (SM_CHANNEL_MODEL)  -> the bundle, unpacked from model.tar.gz:
#       configs/           # metadata.json, train.json / inference.json, ...
#       models/            # model.pt (pretrained checkpoint)
#       scripts/           # optional bundle python
#   data channel (SM_CHANNEL_DATA)    -> a MONAI Decathlon dataset:
#       dataset.json       # with a "training" list of image/label pairs
#       imagesTr/*.nii.gz
#       labelsTr/*.nii.gz
#   Output (SM_MODEL_DIR)             -> fine-tuned checkpoint + result.json,
#                                        bundled into model.tar.gz by SageMaker.
#
# NOTE: DOCKER_BUILDKIT=0 forces the Docker V2 manifest format SageMaker requires
# (it rejects OCI manifests). Large CUDA base layers can be slow to push over a
# proxied/limited connection; consider building from CloudShell/EC2 if pushes stall.

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

# Parse flags
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

AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ECR_REPO="sm-monai-bundle"
IMAGE="${ECR_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"
DLC_REGISTRY="763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com"

if [[ "$DO_BUILD" == true ]]; then
    echo "Logging in to AWS Deep Learning Containers registry..."
    aws ecr get-login-password --region "$AWS_REGION" \
      | docker login --username AWS --password-stdin "$DLC_REGISTRY"

    CACHE_FLAG=""
    if [[ "$NO_CACHE" == true ]]; then
        CACHE_FLAG="--no-cache"
        echo "Building MONAI bundle image (no cache): $IMAGE"
    else
        echo "Building MONAI bundle image (with cache): $IMAGE"
    fi

    # Disable BuildKit to ensure Docker V2 manifest format (SageMaker rejects OCI manifests)
    DOCKER_BUILDKIT=0 docker build $CACHE_FLAG --platform linux/amd64 \
      -f Dockerfile.monai-bundle -t "$IMAGE" .
fi

if [[ "$DO_PUSH" == true ]]; then
    echo "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" \
      | docker login --username AWS --password-stdin \
        "${ECR_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    echo "Creating ECR repository (if not exists)..."
    aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" 2>/dev/null || true

    echo "Deleting old image from ECR (if exists)..."
    aws ecr batch-delete-image \
      --repository-name "$ECR_REPO" \
      --region "$AWS_REGION" \
      --image-ids imageTag=latest 2>/dev/null || true

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
        echo "⚠️  WARNING: Image has OCI manifest ($MEDIA_TYPE). SageMaker CreateModel may reject this."
        echo "   Run: docker system prune -a --force"
        echo "   Then re-run this script with --no-cache."
        exit 1
    else
        echo "✅ Manifest format: $MEDIA_TYPE"
    fi
fi

echo "Done. Image URI: $IMAGE"
