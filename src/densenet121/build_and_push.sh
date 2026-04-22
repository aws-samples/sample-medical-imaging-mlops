#!/bin/bash
# Usage: ./build_and_push.sh <aws-account-id> <region>
# Example: ./build_and_push.sh 575108919340 us-east-1

AWS_ACCOUNT_ID=${1:?Usage: $0 <aws-account-id> <region>}
AWS_REGION=${2:?Usage: $0 <aws-account-id> <region>}
REPO_NAME=sm-training-byoc
IMAGE_TAG=densenet121

IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG

echo "Building image: $IMAGE"
docker build -t $IMAGE .

echo "Authenticating with ECR..."
aws ecr get-login-password --region $AWS_REGION \
    | docker login --username AWS --password-stdin \
      $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "Creating ECR repository (if not exists)..."
aws ecr create-repository --repository-name $REPO_NAME --region $AWS_REGION 2>/dev/null || true

echo "Pushing image..."
docker push $IMAGE

echo "Done. Image URI: $IMAGE"
