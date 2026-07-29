#!/bin/bash
# =============================================================================
# MedML Training Dashboard — Full Setup
# =============================================================================
# Creates the SageMaker execution role and builds/pushes all Docker images.
# Reads all configuration from .env.local.
#
# Usage:
#   ./setup.sh           # Run everything (role + all images)
#   ./setup.sh --role    # Only create the IAM role
#   ./setup.sh --images  # Only build and push Docker images
#
# Prerequisites:
#   - .env.local with AWS_ACCOUNT_ID set
#   - AWS CLI configured with IAM and ECR permissions
#   - Docker running (for image builds)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Parse arguments ---------------------------------------------------------
RUN_ROLE=true
RUN_IMAGES=true

if [[ "${1:-}" == "--role" ]]; then
    RUN_IMAGES=false
elif [[ "${1:-}" == "--images" ]]; then
    RUN_ROLE=false
fi

echo "============================================================"
echo "  MedML Training Dashboard — Setup"
echo "============================================================"
echo ""

# --- Step 1: Create IAM Role ------------------------------------------------
if [[ "$RUN_ROLE" == true ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 1: Create SageMaker Execution Role"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    "$SCRIPT_DIR/create-sagemaker-role.sh"
    echo ""
fi

# --- Step 2: Build and push Docker images ------------------------------------
if [[ "$RUN_IMAGES" == true ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 2: Build and Push Docker Images"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Login to AWS Deep Learning Containers registry (base image source)
    echo "📋 Logging in to AWS DLC registry (base images)..."
    aws ecr get-login-password --region us-east-1 \
      | docker login --username AWS --password-stdin \
        763104351884.dkr.ecr.us-east-1.amazonaws.com
    echo ""

    # Training images
    echo "📋 Building classification training image..."
    "$SCRIPT_DIR/build_and_push.sh" classification
    echo ""

    echo "📋 Building segmentation training image..."
    "$SCRIPT_DIR/build_and_push.sh" segmentation
    echo ""

    # Inference images
    echo "📋 Building classification inference image..."
    "$SCRIPT_DIR/build_and_push_inference.sh" classification
    echo ""

    echo "📋 Building segmentation inference image..."
    "$SCRIPT_DIR/build_and_push_inference.sh" segmentation
    echo ""
fi

# --- Done --------------------------------------------------------------------
echo "============================================================"
echo "  ✅ Setup complete!"
echo "============================================================"
echo ""
if [[ "$RUN_ROLE" == true ]]; then
    echo "  • SageMaker execution role created"
fi
if [[ "$RUN_IMAGES" == true ]]; then
    echo "  • Training images pushed (classification + segmentation)"
    echo "  • Inference images pushed (classification + segmentation)"
fi
echo ""
echo "  Next: run the dashboard with ./start.sh"
echo "============================================================"
