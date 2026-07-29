import torch
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")

VALID_EXTENSIONS = {
    "pytorch": [".pt", ".pth"],
    "onnx": [".onnx"],
    "tensorflow": [".pb", ".h5", ".keras"],
}


class TrainingService:
    def __init__(self):
        self.imported_models: Dict[str, dict] = {}
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def validate_model_file(self, local_path: str, model_format: str) -> bool:
        """Validate that the file is a loadable model artifact"""
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {local_path}")

        valid_exts = VALID_EXTENSIONS.get(model_format)
        if valid_exts and path.suffix not in valid_exts:
            raise ValueError(
                f"Invalid extension '{path.suffix}' for format '{model_format}'. "
                f"Expected one of {valid_exts}"
            )

        if model_format == "pytorch":
            try:
                # weights_only=True prevents arbitrary code execution during deserialization.
                # For models requiring custom objects, use torch.jit.load or ONNX format.
                torch.load(local_path, map_location="cpu", weights_only=True)
            except Exception as e:
                raise ValueError(f"Failed to load PyTorch model: {e}")

        return True

    def register_model(self, model_id: str, model_name: str, local_path: str,
                       s3_source: str, model_format: str, num_classes: int) -> dict:
        """Register an imported model so it's available for training/deployment"""
        entry = {
            "model_id": model_id,
            "model_name": model_name,
            "local_path": local_path,
            "s3_source": s3_source,
            "model_format": model_format,
            "num_classes": num_classes,
            "status": "available",
        }
        self.imported_models[model_id] = entry
        logger.info(f"Registered imported model '{model_name}' ({model_id})")
        return entry

    def get_imported_model(self, model_id: str) -> Optional[dict]:
        return self.imported_models.get(model_id)

    def list_imported_models(self) -> list:
        return list(self.imported_models.values())

    def get_local_path_for_import(self, model_id: str, filename: str) -> str:
        """Return the local filesystem path where an imported model should be stored"""
        dest = MODELS_DIR / model_id
        dest.mkdir(parents=True, exist_ok=True)
        return str(dest / filename)
