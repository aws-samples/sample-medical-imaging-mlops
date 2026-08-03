#!/bin/bash
# Deploy and test the segmentation endpoint
# 1. Repackage model.tar.gz with inference code
# 2. Deploy endpoint
# 3. Test with a sample image

set -e

# Load credentials from .env.local
if [ -f .env.local ]; then
    export $(grep -v '^#' .env.local | xargs)
fi

MODEL_S3=s3://medml-dashboard/model-output/segmentation/segmentation-segresnet-20260729-1559/output/model.tar.gz
ROLE_ARN="${SAGEMAKER_ROLE_ARN:?Error: SAGEMAKER_ROLE_ARN not set. Add it to .env.local}"
TEST_IMAGE=s3://test-imaging-workflows/segmentation/test/10090344/img.nii.gz

echo "Step 1: Repackaging model with inference code..."
python repackage_model.py --model-s3-uri "$MODEL_S3"

echo ""
echo "Step 2: Deploying and testing endpoint..."
python deploy_and_test_segmentation.py \
  --endpoint segmentation \
  --model-data "$MODEL_S3" \
  --role-arn "$ROLE_ARN" \
  --test-image "$TEST_IMAGE"
