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
from app.services.aws_service import AWSService

load_dotenv()

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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=5)