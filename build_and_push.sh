#!/bin/bash
# Usage: ./build_and_push.sh <aws_account_id> <region> [classification|segmentation]
# Example: ./build_and_push.sh 575108919340 us-east-1 classification

set -e

AWS_ACCOUNT_ID=${1:?Usage: $0 <aws_account_id> <region> [task]}
AWS_REGION=${2:?Usage: $0 <aws_account_id> <region> [task]}
TASK=${3:-classification}
ECR_REPO=sm-training-byoc-${TASK}
IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

echo "Building $TASK image: $IMAGE"
docker build -f Dockerfile.sagemaker --build-arg TASK=$TASK -t $IMAGE .

echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "Creating ECR repository (if not exists)..."
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION 2>/dev/null || true

echo "Pushing image..."
docker push $IMAGE

echo "Done. Image URI: $IMAGE"
