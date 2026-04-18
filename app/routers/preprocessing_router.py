from fastapi import APIRouter, HTTPException
from app.models.schemas import PreprocessingConfig, PreprocessingStep
from typing import List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/steps")
async def get_available_preprocessing_steps():
    """Get list of available preprocessing steps"""
    steps = [
        {
            "name": PreprocessingStep.RESIZE,
            "description": "Resize images to specified dimensions",
            "parameters": {
                "width": {"type": "integer", "default": 224},
                "height": {"type": "integer", "default": 224}
            }
        },
        {
            "name": PreprocessingStep.NORMALIZE,
            "description": "Normalize pixel values",
            "parameters": {
                "mean": {"type": "array", "default": [0.485, 0.456, 0.406]},
                "std": {"type": "array", "default": [0.229, 0.224, 0.225]}
            }
        },
        {
            "name": PreprocessingStep.AUGMENT,
            "description": "Apply data augmentation techniques",
            "parameters": {
                "rotation": {"type": "integer", "default": 15},
                "flip_horizontal": {"type": "boolean", "default": True},
                "brightness": {"type": "float", "default": 0.2}
            }
        },
        {
            "name": PreprocessingStep.DENOISE,
            "description": "Remove noise from medical images",
            "parameters": {
                "method": {"type": "string", "default": "gaussian"},
                "sigma": {"type": "float", "default": 1.0}
            }
        },
        {
            "name": PreprocessingStep.CONTRAST_ENHANCE,
            "description": "Enhance image contrast",
            "parameters": {
                "method": {"type": "string", "default": "clahe"},
                "clip_limit": {"type": "float", "default": 2.0}
            }
        }
    ]
    return {"steps": steps}

@router.post("/configure")
async def configure_preprocessing(config: PreprocessingConfig):
    """Configure preprocessing pipeline"""
    try:
        # Validate preprocessing steps
        for step in config.steps:
            if step not in PreprocessingStep:
                raise ValueError(f"Invalid preprocessing step: {step}")
        
        # Store configuration (in a real app, this would be persisted)
        pipeline_id = f"pipeline_{len(config.steps)}_{hash(str(config.steps))}"
        
        return {
            "pipeline_id": pipeline_id,
            "steps": config.steps,
            "parameters": config.parameters,
            "status": "configured"
        }
    except Exception as e:
        logger.error(f"Preprocessing configuration failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/execute/{pipeline_id}")
async def execute_preprocessing(pipeline_id: str, dataset_path: str):
    """Execute preprocessing pipeline on dataset"""
    try:
        # This would execute the actual preprocessing
        # For now, return a mock job response
        job_id = f"preprocess_job_{pipeline_id}_{hash(dataset_path)}"
        
        return {
            "job_id": job_id,
            "pipeline_id": pipeline_id,
            "dataset_path": dataset_path,
            "status": "started",
            "message": "Preprocessing job started successfully"
        }
    except Exception as e:
        logger.error(f"Preprocessing execution failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start preprocessing job")