# Configuration

> **⚠️ Research Use Only** — Models trained and deployed through this platform are for research and development purposes. Clinical deployment requires regulatory review, validation against clinical standards, and human oversight. AI predictions must not be used as the sole basis for medical decisions.

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
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role ARN | `arn:aws:iam::123456789012:role/AmazonSageMaker-ExecutionRole-sgm` |
| `TRAINING_IMAGE_CLASSIFICATION` | ECR URI for the classification training container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest` |
| `TRAINING_IMAGE_SEGMENTATION` | ECR URI for the segmentation training container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest` |
| `INFERENCE_IMAGE_CLASSIFICATION` | ECR URI for the classification inference container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest` |
| `INFERENCE_IMAGE_SEGMENTATION` | ECR URI for the segmentation inference container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest` |
| `DEFAULT_DATA_S3_URI` | Default training data S3 URI (pre-populates the UI form) | `s3://<your-bucket-name>/training/spine-xr/` |
| `TEST_IMAGE_S3_URI` | S3 URI of a test DICOM image for integration tests | `s3://<your-bucket-name>/test/sample.dcm` |
| `TEST_SEGMENTATION_S3_URI` | S3 URI of a test NIfTI volume for segmentation tests | `s3://<your-bucket-name>/test/volume.nii.gz` |

### Example `.env.local`

```bash
# Your AWS account ID (used for ECR image URIs)
AWS_ACCOUNT_ID=123456789012

# SageMaker execution role (created by ./create-sagemaker-role.sh)
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/AmazonSageMaker-ExecutionRole-sgm

# ECR image URIs (built via build_and_push.sh)
TRAINING_IMAGE_CLASSIFICATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest
TRAINING_IMAGE_SEGMENTATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest
INFERENCE_IMAGE_CLASSIFICATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest
INFERENCE_IMAGE_SEGMENTATION=123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest

# Default data locations
DEFAULT_DATA_S3_URI=s3://<your-bucket-name>/training/spine-xr/

# Test data locations
TEST_IMAGE_S3_URI=s3://<your-bucket-name>/test/sample.dcm
TEST_SEGMENTATION_S3_URI=s3://<your-bucket-name>/test/volume.nii.gz
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
  "sagemaker_role_arn": "arn:aws:iam::123456789012:role/AmazonSageMaker-ExecutionRole-sgm",
  "training_image_classification": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest",
  "training_image_segmentation": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest",
  "inference_image_classification": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest",
  "inference_image_segmentation": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest",
  "default_data_s3_uri": "s3://<your-bucket-name>/training/spine-xr/"
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

The `SAGEMAKER_ROLE_ARN` must have permissions to run training jobs, deploy endpoints, access S3 data, pull ECR images, write logs, and track experiments with MLflow.

### Role creation script

The `create-sagemaker-role.sh` script is the bash equivalent of the shared Python `get_or_create_role()` utility (found in `utils.py`). It creates a role with managed AWS policies and an inline MLflow policy.

```bash
# Create with the default role name (AmazonSageMaker-ExecutionRole-sgm)
./create-sagemaker-role.sh

# Or specify a custom role name
./create-sagemaker-role.sh MyCustomRoleName
```

The script also runs via the setup helper:

```bash
./setup.sh --role
```

If the role already exists, the script updates the trust policy and re-attaches all policies (idempotent).

### Default role name

`AmazonSageMaker-ExecutionRole-sgm`

This matches the Python `get_or_create_role()` default, so both the bash script and Python notebooks produce the same role.

### Trust policy

The role trusts the SageMaker service to assume it:

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

### Managed policies attached

The script attaches the following AWS managed policies:

| Policy | Purpose |
|---|---|
| `AmazonSageMakerFullAccess` | Training jobs, endpoints, processing, monitoring |
| `AmazonS3FullAccess` | Read/write training data, model artifacts, capture logs |
| `AmazonRekognitionFullAccess` | Image analysis capabilities |
| `ComprehendFullAccess` | NLP text analysis |
| `ComprehendMedicalFullAccess` | Medical text extraction |
| `AmazonElasticContainerRegistryPublicFullAccess` | Pull/push container images |
| `AmazonSageMakerCanvasFullAccess` | SageMaker Canvas no-code ML |
| `AmazonSageMakerCanvasAIServicesAccess` | Canvas AI services integration |
| `AmazonSageMakerCanvasSMDataScienceAssistantAccess` | Canvas data science assistant |
| `AmazonSageMakerCanvasDataPrepFullAccess` | Canvas data preparation |

> **Production note:** For production deployments, replace `FullAccess` policies with scoped policies restricted to specific bucket ARNs, repository ARNs, and resource prefixes. The broad policies above are suitable for development and experimentation.

### Inline MLflow policy

An inline policy named `mlflow-policy` is added for SageMaker MLflow experiment tracking:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["sagemaker-mlflow:*"],
      "Resource": "*"
    }
  ]
}
```

### When to use which approach

| Context | Method |
|---|---|
| Running outside SageMaker (EC2, local, CI/CD) | `./create-sagemaker-role.sh` or Python `get_or_create_role()` |
| Running inside SageMaker (Studio, notebook instances) | `sagemaker.get_execution_role()` — retrieves the pre-assigned role from instance metadata |

### Creating the role manually

If you prefer not to use the script:

```bash
# 1. Create the role with trust policy
aws iam create-role \
  --role-name AmazonSageMaker-ExecutionRole-sgm \
  --assume-role-policy-document file://trust-policy.json \
  --description "SageMaker execution role for medical imaging training"

# 2. Attach managed policies
aws iam attach-role-policy --role-name AmazonSageMaker-ExecutionRole-sgm \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-role-policy --role-name AmazonSageMaker-ExecutionRole-sgm \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
# ... (attach remaining policies as listed above)

# 3. Add inline MLflow policy
aws iam put-role-policy \
  --role-name AmazonSageMaker-ExecutionRole-sgm \
  --policy-name mlflow-policy \
  --policy-document file://mlflow-policy.json
```

### After role creation

The script automatically updates `SAGEMAKER_ROLE_ARN` in `.env.local`. If you created the role manually, update it yourself:

```bash
# In .env.local
SAGEMAKER_ROLE_ARN=arn:aws:iam::<your-account-id>:role/AmazonSageMaker-ExecutionRole-sgm
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
# All images at once (recommended)
./setup.sh --images

# Or individually:

# Classification training image
./build_and_push.sh classification

# Segmentation training image
./build_and_push.sh segmentation

# Classification inference image
./build_and_push_inference.sh classification

# Segmentation inference image
./build_and_push_inference.sh segmentation
```

All scripts read `AWS_ACCOUNT_ID` and `AWS_DEFAULT_REGION` from `.env.local`.

Each script automatically:
1. Builds the Docker image from the appropriate Dockerfile
2. Logs into your ECR registry
3. Creates the repository if it doesn't exist
4. Pushes the image

> **Important: Docker manifest format** — SageMaker only supports Docker V2 manifests (`application/vnd.docker.distribution.manifest.v2+json`). Newer versions of Docker with BuildKit enabled produce OCI format images by default, which SageMaker rejects with `Unsupported manifest media type application/vnd.oci.image.index.v1+json`. The build scripts handle this automatically by using `DOCKER_BUILDKIT=0`, which forces Docker V2 output. If you build images manually, always include `DOCKER_BUILDKIT=0` in your build command.
>
> **Note:** SageMaker's `CreateModel` operation (used for inference/deployment) is stricter about manifest formats than `CreateTrainingJob`. You may find that training images built with OCI manifests work fine, but the same format fails at deployment time. Always use Docker V2 manifests for both training and inference images to avoid surprises.

#### Debugging: OCI Manifest Errors

If you encounter `Unsupported manifest media type application/vnd.oci.image.manifest.v1+json` when deploying, follow these steps:

**1. Verify the manifest type in ECR:**

```bash
aws ecr batch-get-image \
  --repository-name sm-inference-byoc-classification \
  --region us-east-1 \
  --image-ids imageTag=latest \
  --query 'images[0].imageManifestMediaType' \
  --output text
```

Expected: `application/vnd.docker.distribution.manifest.v2+json`
Problem: `application/vnd.oci.image.manifest.v1+json`

**2. Why this happens:**

- Docker 23+ uses BuildKit by default, which outputs OCI format images
- SageMaker's `CreateModel` API (inference/deployment) validates the manifest type and rejects OCI
- SageMaker's `CreateTrainingJob` API does **not** perform this check — training images with OCI manifests work fine because the container runtime (containerd) on training instances supports both formats
- This inconsistency means the same build process can produce images that work for training but fail at deployment

**3. Fix — delete from ECR, rebuild, and push:**

The working solution requires three things: clearing the Docker cache, deleting the old image from ECR, and rebuilding with BuildKit disabled.

```bash
# Clear all cached images (removes OCI-format base image layers)
docker system prune -a --force

# Login to DLC registry (for pulling the base image)
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# Delete the existing OCI image from ECR
aws ecr batch-delete-image \
  --repository-name sm-inference-byoc-classification \
  --region us-east-1 \
  --image-ids imageTag=latest

# Rebuild with BuildKit disabled and no cache
DOCKER_BUILDKIT=0 docker build --no-cache --platform linux/amd64 \
  -f Dockerfile.inference --build-arg TASK=classification \
  -t 811260109607.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest .

# Login to your ECR
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 811260109607.dkr.ecr.us-east-1.amazonaws.com

# Push fresh
docker push 811260109607.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest

# Verify — must return application/vnd.docker.distribution.manifest.v2+json
aws ecr batch-get-image \
  --repository-name sm-inference-byoc-classification \
  --region us-east-1 \
  --image-ids imageTag=latest \
  --query 'images[0].imageManifest' --output text \
  | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('mediaType','unknown'))"
```

All three steps are critical:
- **`docker system prune -a`** — Without this, Docker reuses cached OCI-format base image layers even with `--no-cache`
- **Delete from ECR** — Pushing with the same tag doesn't always overwrite the manifest
- **`DOCKER_BUILDKIT=0`** — Forces the legacy builder which outputs Docker V2 format

**4. If the above still produces OCI manifests:**

Configure the Docker daemon to disable OCI output entirely:

```bash
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "features": {
    "containerd-snapshotter": false
  }
}
EOF
sudo systemctl restart docker
```

Then repeat step 3. This ensures Docker never converts back to OCI on push.

**5. References:**

- [AWS re:Post — OCI format unsupported when creating a SageMaker model](https://repost.aws/questions/QUQgFRXiCYTjCtyxH_e2x9yg/oci-format-unsupported-when-creating-a-sagemaker-model)
- [AWS re:Post — Manifest error when staging ML model](https://repost.aws/questions/QUYwYL0SKaTQWvQMyqCSBvdQ/manifest-error-when-staging-ml-model)
- [Stack Overflow — Error when using CreateModel for SageMaker](https://stackoverflow.com/questions/74229726/error-when-using-createmodel-function-for-sagemaker)
- [MLflow GitHub Issue #13834 — Unsupported manifest media type](https://github.com/mlflow/mlflow/issues/13834)
- [AWS re:Post — SageMaker Docker OCI manifests support (feature request)](https://repost.aws/questions/QUkzwkEHI-SgeXyjybgq3vLw/sagemaker-docker-oci-manifests-support)

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
# Then create the role:
./create-sagemaker-role.sh
# Start the dashboard:
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
