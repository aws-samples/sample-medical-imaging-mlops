# MedML Training Dashboard

A FastAPI web dashboard for managing medical image classification workflows on AWS. Provides a UI for data ingestion, preprocessing, model training (local and SageMaker), and deployment.

## Features

- **AWS Integration** — connect via environment credentials or explicit keys, browse S3 buckets
- **Data Management** — import datasets from S3, manage raw and processed data
- **Preprocessing** — resize, normalize, augment, denoise, and contrast-enhance medical images
- **Local Training** — train ResNet50, VGG16, EfficientNet, DenseNet121, ViT, or custom models with MLflow and TensorBoard tracking
- **SageMaker Training** — launch BYOC training jobs on SageMaker (e.g. `ml.g5.xlarge`) with MLflow integration
- **Deployment** — deploy trained models to SageMaker endpoints

## Stack

- **Backend** — FastAPI, boto3, PyTorch, MONAI, MLflow
- **Frontend** — Jinja2 templates, vanilla JS
- **Infrastructure** — Docker, SageMaker, S3

## Getting Started

### Prerequisites

- Python 3.11+
- AWS credentials configured (`~/.aws/credentials` or environment variables)

### Local Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env

# Start the app
python main.py
```

App runs at `http://localhost:8000`

### With Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| TensorBoard | http://localhost:6006 |

### With startup script

```bash
chmod +x start.sh
./start.sh
```

This starts MLflow, TensorBoard, and the FastAPI app together.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key | — |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | — |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `MLFLOW_TRACKING_URI` | MLflow server URI | `http://localhost:5000` |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment name | `medical_imaging_experiments` |
| `TENSORBOARD_LOG_DIR` | TensorBoard log directory | `./logs/tensorboard` |

## Project Structure

```
├── app/
│   ├── models/         # Pydantic schemas
│   ├── routers/        # API route handlers
│   └── services/       # Business logic (AWS, training, SageMaker)
├── static/             # CSS and JS
├── templates/          # Jinja2 HTML templates
├── data/               # Raw, processed, and model artifacts (gitignored)
├── logs/               # TensorBoard logs (gitignored)
├── main.py             # App entry point
├── Dockerfile
└── docker-compose.yml
```

## API Endpoints

| Prefix | Description |
|---|---|
| `/api/aws` | AWS credentials and S3 operations |
| `/api/data` | Dataset import and management |
| `/api/preprocessing` | Image preprocessing pipelines |
| `/api/training` | Local and SageMaker training jobs |
| `/api/deployment` | Model deployment to SageMaker |
