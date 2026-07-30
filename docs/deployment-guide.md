# Deployment Guide

This guide covers building the Docker images required for SageMaker training and inference, setting up ECR repositories, and deploying the MedML dashboard itself.

---

## Prerequisites

- Docker installed and running
- AWS CLI v2 configured with credentials that have ECR push/pull permissions
- An AWS account with SageMaker, ECR, and S3 access
- Python 3.11+ (for local dashboard development)

---

## Docker Image Architecture

The project uses three Docker images:

| Image | Purpose | Dockerfile | ECR Repository |
|-------|---------|------------|----------------|
| Training (Classification) | SageMaker BYOC training container | `Dockerfile.sagemaker` | `sm-training-byoc-classification` |
| Training (Segmentation) | SageMaker BYOC training container | `Dockerfile.sagemaker` | `sm-training-byoc-segmentation` |
| Inference (Classification) | SageMaker real-time/async endpoint | `Dockerfile.inference` | `sm-inference-byoc-classification` |
| Inference (Segmentation) | SageMaker real-time/async endpoint | `Dockerfile.inference` | `sm-inference-byoc-segmentation` |
| Dashboard | FastAPI web application | `Dockerfile` | — (local/ECS) |

---

## ECR Repository Setup

Create the ECR repositories before your first build:

```bash
AWS_ACCOUNT_ID=123456789012
AWS_REGION=us-east-1

for repo in sm-training-byoc-classification sm-training-byoc-segmentation \
            sm-inference-byoc-classification sm-inference-byoc-segmentation; do
  aws ecr create-repository \
    --repository-name "$repo" \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true
done
```

The build scripts also create repositories automatically if they don't exist, but pre-creating them allows you to configure scanning and lifecycle policies.

---

## Base Image Authentication

The training and inference Dockerfiles pull from the AWS Deep Learning Containers registry (`763104351884.dkr.ecr.<region>.amazonaws.com`). You must authenticate before building:

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    763104351884.dkr.ecr.us-east-1.amazonaws.com
```

This is required once per machine (or after token expiry). Without it, the build fails with `403 Forbidden`.

---

## Base Image Selection

### Training image

The training container uses the PyTorch Training DLC:

```dockerfile
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker
```

If this tag is unavailable in your region, find alternatives:

```bash
aws ecr describe-images \
  --registry-id 763104351884 \
  --repository-name pytorch-training \
  --region us-east-1 \
  --query 'imageDetails[*].imageTags' \
  --output text | tr '\t' '\n' | grep "py312" | sort
```

A known working alternative:
```
763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker
```

### Inference image

The inference container uses the PyTorch Inference DLC:

```dockerfile
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker
```

This provides the `serve` entrypoint, nginx, gunicorn, and the multi-model server required by SageMaker endpoints.

---

## Building Training Images

### Using setup.sh (recommended)

The easiest way to build and push all images is the unified setup script:

```bash
./setup.sh --images
```

This builds and pushes all four images (training + inference, classification + segmentation), reading `AWS_ACCOUNT_ID` and region from `.env.local`.

### Using build_and_push.sh

To build a single training image:

```bash
# Classification training image
./build_and_push.sh classification

# Segmentation training image
./build_and_push.sh segmentation

# Full rebuild without Docker cache
./build_and_push.sh classification --no-cache
```

The script reads `AWS_ACCOUNT_ID` and `AWS_DEFAULT_REGION` from `.env.local`. No positional arguments for account ID or region are needed.

By default, builds use Docker's layer cache for faster rebuilds. Pass `--no-cache` to force a full rebuild (useful when base image or dependencies have changed).

**What it does:**
1. Logs in to the AWS DLC ECR registry (base image source)
2. Builds the Docker image from `Dockerfile.sagemaker` with the `TASK` build arg
3. Logs in to your ECR registry
4. Creates the repository if it doesn't exist
5. Pushes the image

**Files bundled into the image:**
- `src/<task>/requirements.txt` — pip dependencies
- `src/<task>/` — all training scripts (`train.py`, `model_def.py`, etc.)

The resulting image tag: `<account>.dkr.ecr.<region>.amazonaws.com/sm-training-byoc-<task>:latest`

### Dockerfile.sagemaker structure

```dockerfile
ARG TASK=classification
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker
ARG TASK
COPY src/${TASK}/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY src/${TASK}/ /opt/program/
COPY src/inference/${TASK}/ /opt/program/inference/
ENV SAGEMAKER_PROGRAM train.py
```

Notes:
- SageMaker training containers require root (no `USER` directive)
- No `HEALTHCHECK` — SageMaker manages container lifecycle
- The `SAGEMAKER_PROGRAM` env var tells SageMaker which script to execute
- The `src/inference/${TASK}/` directory is included so training scripts can copy inference artifacts (e.g., `inference.py`) into `model_dir/code/` for SageMaker PyTorch serving

---

## Building Inference Images

### Using build_and_push_inference.sh

```bash
# Classification inference image
./build_and_push_inference.sh classification

# Segmentation inference image
./build_and_push_inference.sh segmentation

# Full rebuild without Docker cache
./build_and_push_inference.sh classification --no-cache
```

The script reads `AWS_ACCOUNT_ID` and `AWS_DEFAULT_REGION` from `.env.local`.

By default, builds use Docker's layer cache. Pass `--no-cache` for a full rebuild.

**What it does:**
1. Logs in to the AWS DLC ECR registry (base image source)
2. Builds the Docker image from `Dockerfile.inference` with the `TASK` build arg
3. Logs in to your ECR registry
4. Creates the repository if it doesn't exist
5. **Deletes the existing `latest` image from ECR** (ensures the new push fully replaces the old manifest)
6. Pushes the new image
7. **Verifies the pushed manifest format** is Docker V2 (not OCI) — SageMaker's `CreateModel` API rejects OCI manifests

### Dockerfile.inference structure

```dockerfile
ARG TASK=classification
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker
ARG TASK
COPY src/inference/${TASK}/requirements.txt /tmp/requirements-inference.txt
RUN pip install --no-cache-dir -r /tmp/requirements-inference.txt
```

Notes:
- `SAGEMAKER_PROGRAM` is set at deploy time via `create_model` Environment — not baked in
- SageMaker inference containers require root for nginx/gunicorn
- The inference script (`inference.py`) is bundled inside `model.tar.gz`, not the container image

### model.tar.gz Structure

SageMaker's PyTorch serving container requires inference code to be located at `code/inference.py` inside the model archive. The training scripts now automatically copy inference artifacts to the correct location (`model_dir/code/`).

**Required structure:**

```
model.tar.gz
├── best_model.pth          # Model weights
├── result.json             # Training metadata
└── code/
    └── inference.py        # Inference handler (model_fn, input_fn, predict_fn, output_fn)
```

> **Important:** Placing `inference.py` at the root of the archive will not work with the SageMaker PyTorch serving container. It must be under `code/`.

#### Repackaging Existing Models

If you have an existing `model.tar.gz` that was created before this change (with `inference.py` at the root), use the repackaging script:

```bash
python repackage_model.py s3://your-bucket/output/job-name/output/model.tar.gz
```

This downloads the archive, restructures it to place inference artifacts under `code/`, and re-uploads it to the same S3 location.

---

## Deploying the Dashboard

### Local Development

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
cp .env.local.example .env.local
# Edit .env.local with your AWS account ID, role ARN, ECR URIs

# Start the app
python main.py
```

The dashboard runs at `http://localhost:8000`.

### Docker (with MLflow + TensorBoard)

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| TensorBoard | http://localhost:6006 |

The `docker-compose.yml` mounts `./data` and `./logs` for persistence.

### Using start.sh

```bash
chmod +x start.sh
./start.sh
```

Starts MLflow, TensorBoard, and the FastAPI app together in a single terminal.

### Production Considerations

#### Environment variables

All account-specific configuration goes in `.env.local` (gitignored):
- `AWS_ACCOUNT_ID`
- `SAGEMAKER_ROLE_ARN`
- ECR image URIs for training and inference
- `DEFAULT_DATA_S3_URI`

See the [Configuration](configuration.md) page for the full list.

#### Persistent state

The dashboard persists state to JSON files in `data/`:
- `data/sagemaker_jobs.json` — training job records
- `data/deployments.json` — endpoint deployment records
- `data/monitoring.json` — Model Monitor schedule records
- `data/training_reports/` — per-job JSON reports

Ensure this directory is on persistent storage (e.g., EBS volume if running on EC2/ECS).

#### AWS credentials

For production, use an IAM instance profile or ECS task role rather than static credentials. The dashboard auto-detects credentials from the environment/instance metadata on startup.

#### Reverse proxy

For production deployments, put the dashboard behind a reverse proxy (nginx, ALB) with:
- HTTPS termination
- Authentication (e.g., Cognito, IAM Identity Center)
- Rate limiting

#### Instance sizing

The dashboard itself is lightweight (FastAPI + Jinja2). A `t3.small` or equivalent is sufficient. The heavy compute (training, inference) happens on SageMaker instances, not the dashboard host.

---

## Transferring to Another Machine

The cleanest approach is git:

```bash
# Source machine
git add -A && git commit -m "your message" && git push origin <branch>

# Target machine
git clone <repo-url> && git checkout <branch>
```

Minimum files needed for the Docker build:
```
Dockerfile.sagemaker
Dockerfile.inference
build_and_push.sh
build_and_push_inference.sh
src/classification/    (for classification images)
src/segmentation/      (for segmentation images)
src/inference/         (for inference images)
```

---

## Troubleshooting

### `403 Forbidden` during Docker build

You haven't authenticated to the AWS DLC ECR registry. Run:
```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    763104351884.dkr.ecr.us-east-1.amazonaws.com
```

### Image tag not found

The base image tag may not be available in your region. List available tags:
```bash
aws ecr describe-images \
  --registry-id 763104351884 \
  --repository-name pytorch-training \
  --region <your-region> \
  --query 'imageDetails[*].imageTags' \
  --output text | tr '\t' '\n' | grep "py312" | sort
```

### SageMaker training job fails immediately

Check:
1. The ECR image URI is correct and accessible by the SageMaker execution role
2. The execution role has permissions to pull from your ECR repository
3. The training data S3 URI is accessible by the role
4. The instance type is available in your region

### Endpoint creation stuck in `Creating`

SageMaker endpoint creation typically takes 5–10 minutes. If it exceeds 15 minutes, check the SageMaker console for error details. Common causes:
- Inference image doesn't have the `serve` entrypoint
- `model.tar.gz` is missing required files or has incorrect structure (must contain `code/inference.py` and model weights)
- Instance type capacity issues in the region
