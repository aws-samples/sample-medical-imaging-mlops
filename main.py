from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
import os
from dotenv import load_dotenv

from app.routers import aws_router, training_router, deployment_router
from app.routers import inference_router
from app.routers import fairness_router
from app.routers import evaluation_router
from app.routers import monitoring_router
from app.services.aws_service import AWSService

# Load .env first, then .env.local overrides (if present)
load_dotenv()
load_dotenv(".env.local", override=True)

app = FastAPI(title="MedML Training Platform", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount evaluation results cache as static
EVAL_RESULTS_DIR = Path("data/evaluation_results")
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/evaluation-results", StaticFiles(directory=str(EVAL_RESULTS_DIR)), name="evaluation-results")

# Templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(aws_router.router, prefix="/api/aws", tags=["AWS"])
app.include_router(training_router.router, prefix="/api/training", tags=["Training"])
app.include_router(deployment_router.router, prefix="/api/deployment", tags=["Deployment"])
app.include_router(inference_router.router, prefix="/api/inference", tags=["Inference"])
app.include_router(fairness_router.router, prefix="/api/fairness", tags=["Fairness"])
app.include_router(evaluation_router.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(monitoring_router.router, prefix="/api/monitoring", tags=["Monitoring"])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/config")
async def get_config():
    """Expose non-secret configuration to the frontend.
    Values come from .env.local (gitignored) so no account-specific data is in source."""
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    account_id = os.environ.get("AWS_ACCOUNT_ID", "")
    return {
        "aws_account_id": account_id,
        "aws_region": region,
        "sagemaker_role_arn": os.environ.get("SAGEMAKER_ROLE_ARN", ""),
        "training_image_classification": os.environ.get(
            "TRAINING_IMAGE_CLASSIFICATION",
            f"{account_id}.dkr.ecr.{region}.amazonaws.com/sm-training-byoc-classification:latest" if account_id else ""
        ),
        "training_image_segmentation": os.environ.get(
            "TRAINING_IMAGE_SEGMENTATION",
            f"{account_id}.dkr.ecr.{region}.amazonaws.com/sm-training-byoc-segmentation:latest" if account_id else ""
        ),
        "inference_image_classification": os.environ.get(
            "INFERENCE_IMAGE_CLASSIFICATION",
            f"{account_id}.dkr.ecr.{region}.amazonaws.com/sm-inference-byoc-classification:latest" if account_id else ""
        ),
        "inference_image_segmentation": os.environ.get(
            "INFERENCE_IMAGE_SEGMENTATION",
            f"{account_id}.dkr.ecr.{region}.amazonaws.com/sm-inference-byoc-segmentation:latest" if account_id else ""
        ),
        "default_data_s3_uri": os.environ.get("DEFAULT_DATA_S3_URI", ""),
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=8000, timeout_keep_alive=5)