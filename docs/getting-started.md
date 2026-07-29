# Getting Started

> **Note:** All `http://localhost` URLs in this documentation refer to the local development server and are not exposed to the public internet.

This guide walks you through setting up and running the MedML Training Dashboard — a FastAPI web application for managing medical image classification and segmentation workflows on AWS.

## Prerequisites

Before you begin, ensure you have the following:

| Requirement | Details |
|---|---|
| **Python** | 3.11 or higher |
| **AWS Account** | With permissions for S3, SageMaker, and ECR |
| **AWS CLI** | Installed and configured (`~/.aws/credentials` or environment variables) |
| **Docker** | Optional — required only for the multi-service setup (dashboard + MLflow + TensorBoard) |
| **Git** | For cloning the repository |

### AWS Permissions

Your AWS credentials need access to:
- **S3** — read/write for training data, model artifacts, and capture logs
- **SageMaker** — create/manage training jobs, endpoints, processing jobs, and monitoring schedules
- **ECR** — pull/push Docker images (only needed if building custom training containers)

---

## ⚠️ Medical AI Safety Notice

> **This platform is designed for research and development purposes only.**
>
> Models trained and deployed through this system are **not validated for clinical use**. Production deployment of medical imaging AI requires:
>
> 1. **Regulatory clearance** — FDA 510(k), De Novo, PMA, CE marking, or equivalent regulatory pathway for your jurisdiction
> 2. **Clinical validation studies** — Prospective trials demonstrating safety and efficacy on the target patient population
> 3. **Human-in-the-loop oversight** — Qualified healthcare professionals must review and interpret all AI predictions
> 4. **Bias and fairness evaluation** — Subgroup analysis across demographic groups to ensure equitable performance
> 5. **Continuous monitoring** — Post-deployment surveillance for model drift and performance degradation
>
> **AI predictions must not be used as the sole basis for medical decisions.**
>
> Consult with regulatory affairs, clinical validation teams, and your institution's IRB/ethics board before any patient-facing deployment. For guidance on responsible AI practices in healthcare, see the [AWS Responsible AI documentation](https://aws.amazon.com/ai/responsible-ai/) and [AWS Healthcare AI compliance resources](https://aws.amazon.com/health/solutions/responsible-ai/).

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Kiro-ML-model
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, boto3, PyTorch, MONAI, MLflow, and all other required packages.

### 4. Copy environment files

```bash
cp .env.example .env
cp .env.local.example .env.local
```

Edit both files with your configuration (see [Configuration](#configuration) below).

## Quick Setup

Once `.env.local` is configured with your `AWS_ACCOUNT_ID`, run the setup script to create the SageMaker IAM role and build/push all Docker images in one step:

```bash
./setup.sh
```

Or run individual steps:

```bash
./setup.sh --role    # Only create the SageMaker execution role
./setup.sh --images  # Only build and push Docker images
```

The role creation script (`create-sagemaker-role.sh`) mirrors the Python `get_or_create_role()` utility from `utils.py`. It creates a role named `AmazonSageMaker-ExecutionRole-sgm` with the same managed policies and inline MLflow policy, so the behavior is identical whether you run the bash script or use the Python helper in notebooks.

```bash
# Equivalent standalone usage:
./create-sagemaker-role.sh                          # Uses default name
./create-sagemaker-role.sh MyCustomRoleName         # Custom name
```

The setup script reads all account-specific values from `.env.local` — no positional arguments needed.

## Configuration

The application loads environment variables in two layers:

1. **`.env`** — General application configuration (loaded first)
2. **`.env.local`** — Account-specific overrides (loaded second, takes priority)

`.env.local` is gitignored and should never be committed — it contains your AWS account-specific values.

### `.env` — General config

| Variable | Description | Default |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key (optional if using `~/.aws/credentials`) | — |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key (optional if using `~/.aws/credentials`) | — |
| `AWS_DEFAULT_REGION` | AWS region for all services | `us-east-1` |
| `MLFLOW_TRACKING_URI` | MLflow server URI for experiment tracking | `http://localhost:5000` |
| `MLFLOW_EXPERIMENT_NAME` | Default MLflow experiment name | `medical_imaging_experiments` |
| `TENSORBOARD_LOG_DIR` | Directory for TensorBoard log files | `./logs/tensorboard` |

### `.env.local` — Account-specific config

| Variable | Description | Example |
|---|---|---|
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID | `123456789012` |
| `SAGEMAKER_ROLE_ARN` | IAM role ARN that SageMaker assumes for training and deployment | `arn:aws:iam::123456789012:role/AmazonSageMaker-ExecutionRole-sgm` |
| `TRAINING_IMAGE_CLASSIFICATION` | ECR URI for the classification training container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest` |
| `TRAINING_IMAGE_SEGMENTATION` | ECR URI for the segmentation training container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest` |
| `INFERENCE_IMAGE_CLASSIFICATION` | ECR URI for the classification inference container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest` |
| `INFERENCE_IMAGE_SEGMENTATION` | ECR URI for the segmentation inference container | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest` |
| `DEFAULT_DATA_S3_URI` | Default S3 URI for training data (pre-populates the UI form) | `s3://your-bucket/training-data/` |
| `TEST_IMAGE_S3_URI` | S3 URI of a test DICOM image for integration tests | `s3://your-bucket/test/image.dcm` |
| `TEST_SEGMENTATION_S3_URI` | S3 URI of a test NIfTI volume for segmentation tests | `s3://your-bucket/test/volume.nii.gz` |

> **Tip:** You only need `AWS_ACCOUNT_ID` and `SAGEMAKER_ROLE_ARN` to get started. The ECR image URIs are required when launching SageMaker training jobs or deploying endpoints.

## First Launch

Start the application:

```bash
python main.py
```

The dashboard starts at **http://localhost:8000**.

Open it in your browser — you'll see the main dashboard with tabs for Training, Deployment, Inference, Monitoring, and more.

> On first launch with empty data stores, the local job and deployment lists will be empty. See the Initial Setup section below to populate them from AWS.

## Initial Setup

### Option A: Configure AWS via environment variables

If your `.env` or `~/.aws/credentials` already has valid AWS credentials, the app connects automatically on startup.

### Option B: Configure AWS via the UI

1. Open **http://localhost:8000**
2. Navigate to the **AWS** section
3. Enter your Access Key ID, Secret Access Key, and Region
4. Click **Connect**

### Sync existing resources from AWS

Once connected, pull your existing SageMaker resources into the dashboard:

1. Go to the **Training** tab → click **Sync from AWS** to import existing training jobs
2. Go to the **Deployment** tab → click **Sync from AWS** to import existing endpoints
3. Go to the **Evaluation** tab → click **Sync from AWS** to import existing evaluation jobs

This populates the local data stores (`data/sagemaker_jobs.json`, `data/deployments.json`) so subsequent loads are fast and don't require AWS calls.

## Docker Setup

For a full development environment with MLflow and TensorBoard, use Docker Compose:

```bash
docker-compose up --build
```

This starts three services:

| Service | URL | Purpose |
|---|---|---|
| **Dashboard** | http://localhost:8000 | Main application UI and API |
| **MLflow** | http://localhost:5000 | Experiment tracking and model registry |
| **TensorBoard** | http://localhost:6006 | Training visualization |

Alternatively, use the startup script to run all services without Docker:

```bash
chmod +x start.sh
./start.sh
```

This starts MLflow, TensorBoard, and the FastAPI app together in the background.

## Quick Start Workflow

Here's a complete end-to-end workflow: connect to AWS, prepare data, train a model, deploy it, and run inference.

### Step 1 — Connect to AWS

Ensure your AWS credentials are configured (via `.env`, environment variables, or the UI). Verify the connection by browsing S3 buckets in the AWS tab.

### Step 2 — Prepare training data in S3

Upload your medical imaging dataset to S3 in the expected folder structure:

```
s3://your-bucket/training-data/
├── class_a/
│   ├── image001.dcm
│   ├── image002.dcm
│   └── ...
├── class_b/
│   ├── image003.dcm
│   └── ...
└── class_c/
    └── ...
```

Each subfolder name becomes a class label. The number of classes is determined automatically from the folder structure.

For segmentation tasks, organize data with paired images and labels.

### Step 3 — Train a model

#### Option A: SageMaker training (recommended for GPU workloads)

1. Go to the **Training** tab
2. Select **SageMaker Training**
3. Configure:
   - **Model architecture** — ResNet50, VGG16, EfficientNet, DenseNet121, ViT, or custom
   - **Data S3 URI** — path to your training data (pre-filled from `DEFAULT_DATA_S3_URI`)
   - **Instance type** — e.g. `ml.g5.xlarge` for GPU training
   - **Epochs, learning rate, batch size** — training hyperparameters
   - **MLflow tracking URI** — optional, for experiment tracking
4. Click **Start Training**

The job runs on SageMaker. Monitor progress in the Training tab — active jobs auto-refresh their status.

#### Option B: Local training

1. Select **Local Training** in the Training tab
2. Choose your model and dataset
3. Training runs on your local machine with MLflow and TensorBoard tracking

### Step 4 — Deploy the model

Once training completes:

1. Go to the **Deployment** tab
2. Click **Deploy** on a completed training job, or create a new deployment:
   - **Model artifacts** — S3 URI of the `model.tar.gz` produced by training
   - **Inference image** — ECR URI of the inference container (from `.env.local`)
   - **Instance type** — e.g. `ml.g5.xlarge`
   - **Role ARN** — SageMaker execution role (from `.env.local`)
3. Click **Deploy**

The endpoint progresses from `Creating` → `InService`. This typically takes 5–10 minutes.

### Step 5 — Run inference

Once the endpoint is `InService`:

1. Go to the **Inference** tab
2. Select the endpoint and task type (classification or segmentation)
3. Provide input:
   - **S3 URI** — point to an image in S3
   - **File upload** — upload a DICOM or NIfTI file directly from your browser
4. Click **Invoke**

The response shows predicted classes with confidence scores (classification) or a binary mask with volume statistics (segmentation).

### Step 6 — Monitor (optional)

Set up Model Monitor to track prediction drift:

1. Enable **Data Capture** on the endpoint (Monitoring tab → Data Capture)
2. Run a **Baseline** job against your training data
3. Create a **Monitoring Schedule** to compare live traffic against the baseline

---

## Next Steps

- **Evaluation** — run a SageMaker Processing Job to evaluate endpoint accuracy on a labeled test set
- **Fairness Analysis** — scan DICOM metadata for demographic distributions and bias auditing
- **Fine-tuning** — resume training from a checkpoint using `checkpoint_s3_uri`
- **Report Generation** — generate academic paper sections from completed training and evaluation jobs
- **Custom Containers** — build and push BYOC training/inference images (see README for `build_and_push.sh` instructions)

## Troubleshooting

| Issue | Solution |
|---|---|
| `ValueError: AWS credentials not configured` | Set credentials in `.env`, environment, or `~/.aws/credentials` |
| SageMaker job fails immediately | Check that `SAGEMAKER_ROLE_ARN` has the necessary permissions (run `./create-sagemaker-role.sh` to create/fix it) and the ECR image URI is correct |
| Endpoint stuck in `Creating` | Wait up to 10 minutes; check the SageMaker console for errors |
| MLflow not tracking | Ensure `MLFLOW_TRACKING_URI` points to a running MLflow server (or use Docker Compose) |
| Docker build fails with `403 Forbidden` | Run `aws ecr get-login-password` to authenticate with the AWS Deep Learning Containers ECR (see README) |
| SageMaker rejects image with `Unsupported manifest media type` | Your Docker image was built in OCI format. Rebuild with `docker buildx build --provenance=false` or use the provided `build_and_push.sh` scripts which handle this automatically |
| Empty job list after restart | Data stores auto-rebuild from `data/training_reports/` on first boot; click Sync from AWS to pull fresh data |
