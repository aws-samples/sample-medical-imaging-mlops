# Configuration

This document covers all configuration options for the MedML Training Dashboard — environment files, AWS credentials, Docker setup, IAM permissions, and ECR image management.

---

## 1. Environment Files

The application uses a two-layer environment variable system loaded by `python-dotenv` in `main.py`:

| File | Purpose | Committed to git? |
|---|---|---|
| `.env` | General application config (region, MLflow URI, log settings) | Yes (`.env.example` template) |
| `.env.local` | Account-specific secrets and identifiers (account ID, role ARN, ECR URIs) | **No** (gitignored) |
| `.env.local.example` | Template for `.env.local` with placeholder values | Yes |

**Load order:** `.env` is loaded first, then `.env.local` overrides any overlapping keys:

```python
load_dotenv()                          # loads .env
load_dotenv(".env.local", override=True)  # loads .env.local, overriding duplicates
```

### Initial setup

```bash
cp .env.example .env
cp .env.local.example .env.local
# Edit .env.local with your real AWS account ID, role ARN, and ECR URIs
```

---

## 2. General Config (`.env`)

These variables control application behavior and third-party service endpoints.

| Variable | Description | Default | Example |
|---|---|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key (optional if using `~/.aws/credentials` or IAM roles) | — | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | — | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `AWS_DEFAULT_REGION` | AWS region for all boto3 clients | `us-east-1` | `us-west-2` |
| `MLFLOW_TRACKING_URI` | MLflow server endpoint for experiment tracking | `http://localhost:5000` | `http://mlflow.internal:5000` |
| `MLFLOW_EXPERIMENT_NAME` | Default MLflow experiment name | `medical_imaging_experiments` | `spine-classification-v2` |
| `TENSORBOARD_LOG_DIR` | Directory for TensorBoard event files | `./logs/tensorboard` | `/data/tb-logs` |
| `DEBUG` | Enable debug mode (verbose logging, auto-reload) | `True` | `False` |
| `LOG_LEVEL` | Python logging level | `INFO` | `DEBUG`, `WARNING`, `ERROR` |

### Example `.env`

```bash
# AWS Configuration (optional — can be configured via UI)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=medical_imaging_experiments

# TensorBoard Configuration
TENSORBOARD_LOG_DIR=./logs/tensorboard

# Application Configuration
DEBUG=True
LOG_LEVEL=INFO
```

---

## 3. Account-Specific Config (`.env.local`)

These variables are unique to your AWS account and should never be committed.

| Variable | Description | Example |
|---|---|---|
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID | `123456789012` |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role ARN | `arn:aws:iam::123456789012:role/SageMakerExecutionRole` |
| `TRAINING_IMAGE_CLASSIFICATION` | ECR URI for the classification training container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest` |
| `TRAINING_IMAGE_SEGMENTATION` | ECR URI for the segmentation training container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest` |
| `INFERENCE_IMAGE_CLASSIFICATION` | ECR URI for the classification inference container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest` |
| `INFERENCE_IMAGE_SEGMENTATION` | ECR URI for the segmentation inference container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest` |
| `DEFAULT_DATA_S3_URI` | Default training data S3 URI (pre-populates the UI form) | `s3://medml-data/training/spine-xr/` |
| `TEST_IMAGE_S3_URI` | S3 URI of a test DICOM image for integration tests | `s3://medml-data/test/sample.dcm` |
| `TEST_SEGMENTATION_S3_URI` | S3 URI of a test NIfTI volume for segmentation tests | `s3://medml-data/test/volume.nii.gz` |

### Example `.env.local`

```bash
# Your AWS account ID (used for ECR image URIs)
AWS_ACCOUNT_ID=123456789012

# SageMaker execution role
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerExecutionRole

# ECR image URIs (built via build_and_push.sh)
TRAINING_IMAGE_CLASSIFICATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest
TRAINING_IMAGE_SEGMENTATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest
INFERENCE_IMAGE_CLASSIFICATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest
INFERENCE_IMAGE_SEGMENTATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest

# Default data locations
DEFAULT_DATA_S3_URI=s3://medml-data/training/spine-xr/

# Test data locations
TEST_IMAGE_S3_URI=s3://medml-data/test/sample.dcm
TEST_SEGMENTATION_S3_URI=s3://medml-data/test/volume.nii.gz
```

---

## 4. AWS Credentials

The application supports three methods for providing AWS credentials, checked in this order:

### Method 1: Environment variables (`.env` or `.env.local`)

Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in your `.env` file:

```bash
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
```

Best for: local development when you don't have AWS CLI configured.

### Method 2: AWS credentials file (`~/.aws/credentials`)

Configure the AWS CLI and the app picks up the default profile automatically:

```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region, Output format
```

Or use a named profile by setting `AWS_PROFILE`:

```bash
export AWS_PROFILE=medml-dev
python main.py
```

Best for: developers who use multiple AWS accounts.

### Method 3: Dashboard UI

Navigate to the **AWS** tab in the dashboard and enter credentials interactively. The dashboard calls `AWSService.configure_credentials()` which creates boto3 clients in memory — credentials are not persisted to disk.

Best for: quick testing or demos without touching local config files.

### Credential priority

If environment variables are set at startup, the `AWSService` automatically initializes boto3 clients. If they're absent, the service remains unconfigured until credentials are provided via the UI.

---

## 5. Runtime Config API

The `GET /api/config` endpoint serves non-secret configuration values to the frontend, allowing the UI to pre-populate forms with ECR URIs, role ARNs, and default data paths.

### Response

```json
{
  "aws_account_id": "123456789012",
  "aws_region": "us-east-1",
  "sagemaker_role_arn": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
  "training_image_classification": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest",
  "training_image_segmentation": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest",
  "inference_image_classification": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest",
  "inference_image_segmentation": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest",
  "default_data_s3_uri": "s3://medml-data/training/spine-xr/"
}
```

### How it works

- Values come from environment variables (loaded from `.env.local`)
- If `AWS_ACCOUNT_ID` is set but specific image URIs are not, the endpoint auto-generates ECR URIs using the conventional naming scheme
- No secrets (access keys, secret keys) are ever exposed through this endpoint

### Frontend usage

The dashboard JavaScript fetches `/api/config` on page load to pre-fill the SageMaker training form:

```javascript
const config = await fetch('/api/config').then(r => r.json());
document.getElementById('image-uri').value = config.training_image_classification;
document.getElementById('role-arn').value = config.sagemaker_role_arn;
```

---

## 6. Docker Configuration

### `docker-compose.yml`

The single-container setup runs FastAPI, MLflow, and TensorBoard together:

```yaml
version: '3.8'

services:
  medml-platform:
    build: .
    ports:
      - "8000:8000"  # FastAPI application
      - "5000:5000"  # MLflow
      - "6006:6006"  # TensorBoard
    environment:
      - DEBUG=True
      - LOG_LEVEL=INFO
      - MLFLOW_TRACKING_URI=http://localhost:5000
      - TENSORBOARD_LOG_DIR=./logs/tensorboard
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
```

### Environment variables in Docker

Pass AWS credentials and account config at run time (never bake them into the image):

```bash
docker-compose run -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
                   -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
                   -e AWS_DEFAULT_REGION=us-east-1 \
                   medml-platform
```

Or use an env file:

```bash
# docker-compose.override.yml
services:
  medml-platform:
    env_file:
      - .env
      - .env.local
```

### Volume mounts

| Host path | Container path | Purpose |
|---|---|---|
| `./data` | `/app/data` | Persisted job records, training reports, imported models |
| `./logs` | `/app/logs` | TensorBoard event files |
| `./models` | `/app/models` | Downloaded model artifacts |

### Exposed ports

| Port | Service |
|---|---|
| 8000 | FastAPI dashboard |
| 5000 | MLflow tracking server |
| 6006 | TensorBoard |

---

## 7. SageMaker IAM Role

The `SAGEMAKER_ROLE_ARN` must have permissions to run training jobs, deploy endpoints, access S3 data, pull ECR images, and write logs.

### Required permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3Access",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::your-data-bucket",
        "arn:aws:s3:::your-data-bucket/*",
        "arn:aws:s3:::sagemaker-*",
        "arn:aws:s3:::sagemaker-*/*"
      ]
    },
    {
      "Sid": "SageMakerFullAccess",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:ListTrainingJobs",
        "sagemaker:StopTrainingJob",
        "sagemaker:CreateModel",
        "sagemaker:DeleteModel",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:DeleteEndpointConfig",
        "sagemaker:CreateEndpoint",
        "sagemaker:DeleteEndpoint",
        "sagemaker:DescribeEndpoint",
        "sagemaker:UpdateEndpoint",
        "sagemaker:InvokeEndpoint",
        "sagemaker:CreateProcessingJob",
        "sagemaker:DescribeProcessingJob",
        "sagemaker:ListProcessingJobs",
        "sagemaker:CreateMonitoringSchedule",
        "sagemaker:DeleteMonitoringSchedule",
        "sagemaker:DescribeMonitoringSchedule",
        "sagemaker:ListMonitoringSchedules",
        "sagemaker:ListMonitoringExecutions"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ECRAccess",
      "Effect": "Allow",
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams",
        "logs:GetLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
    },
    {
      "Sid": "PassRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": "sagemaker.amazonaws.com"
        }
      }
    }
  ]
}
```

### Trust policy

The role must trust the SageMaker service:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### Creating the role via CLI

```bash
# Create the role
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document file://trust-policy.json

# Attach the custom policy
aws iam put-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-name MedMLSageMakerPolicy \
  --policy-document file://sagemaker-policy.json

# Or use the AWS managed policy for broader access (development only)
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

---

## 8. ECR Repositories

Four ECR repositories are required — two for training (classification + segmentation) and two for inference.

### Repository names

| Repository | Purpose | Build script |
|---|---|---|
| `sm-training-byoc-classification` | Classification training container | `build_and_push.sh` |
| `sm-training-byoc-segmentation` | Segmentation training container | `build_and_push.sh` |
| `sm-inference-byoc-classification` | Classification inference container | `build_and_push_inference.sh` |
| `sm-inference-byoc-segmentation` | Segmentation inference container | `build_and_push_inference.sh` |

### Prerequisites

1. Docker installed and running
2. AWS CLI configured with ECR push permissions
3. Login to the AWS Deep Learning Containers registry (base image source):

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    763104351884.dkr.ecr.us-east-1.amazonaws.com
```

> This is required on every new machine. Without it, Docker cannot pull the PyTorch base image and the build fails with `403 Forbidden`.

### Build and push commands

```bash
# Classification training image
./build_and_push.sh 123456789012 us-east-1 classification

# Segmentation training image
./build_and_push.sh 123456789012 us-east-1 segmentation

# Classification inference image
./build_and_push_inference.sh 123456789012 us-east-1 classification

# Segmentation inference image
./build_and_push_inference.sh 123456789012 us-east-1 segmentation
```

Each script automatically:
1. Builds the Docker image from the appropriate Dockerfile
2. Logs into your ECR registry
3. Creates the repository if it doesn't exist
4. Pushes the image

### What the training image contains

The `Dockerfile.sagemaker` builds on the AWS PyTorch DLC base image and installs task-specific dependencies:

```dockerfile
ARG TASK=classification
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker
ARG TASK
COPY src/${TASK}/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY src/${TASK}/ /opt/program/
ENV SAGEMAKER_PROGRAM train.py
```

Key packages bundled: `monai`, `torch`, `torchvision`, `nibabel`, `SimpleITK`, `pydicom`, `mlflow`, `tensorboard`.

### After building

Update `.env.local` with the resulting image URIs:

```bash
TRAINING_IMAGE_CLASSIFICATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest
TRAINING_IMAGE_SEGMENTATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest
INFERENCE_IMAGE_CLASSIFICATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest
INFERENCE_IMAGE_SEGMENTATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest
```

### Verifying images

```bash
# List images in a repository
aws ecr describe-images \
  --repository-name sm-training-byoc-classification \
  --region us-east-1 \
  --query 'imageDetails[*].{Tags:imageTags,Pushed:imagePushedAt,Size:imageSizeInBytes}'

# Pull and test locally
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest
docker run --rm -it <image-uri> python -c "import torch; import monai; print(f'PyTorch {torch.__version__}, MONAI {monai.__version__}')"
```

---

## Quick Reference

### Minimal setup (local development)

```bash
cp .env.example .env
cp .env.local.example .env.local
# Edit .env.local:
#   AWS_ACCOUNT_ID=123456789012
#   SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerExecutionRole
python main.py
```

### Full setup (training + deployment)

```bash
# 1. Configure environment
cp .env.example .env
cp .env.local.example .env.local
vim .env.local  # fill in all values

# 2. Build and push Docker images
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
./build_and_push.sh YOUR_ACCOUNT_ID us-east-1 classification
./build_and_push.sh YOUR_ACCOUNT_ID us-east-1 segmentation
./build_and_push_inference.sh YOUR_ACCOUNT_ID us-east-1 classification
./build_and_push_inference.sh YOUR_ACCOUNT_ID us-east-1 segmentation

# 3. Start the dashboard
python main.py

# 4. Open http://localhost:8000 and sync from AWS
```
