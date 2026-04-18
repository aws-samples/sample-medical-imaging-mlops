from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class AWSCredentials(BaseModel):
    access_key_id: str
    secret_access_key: str
    region: str = "us-east-1"

class S3Bucket(BaseModel):
    name: str
    region: str
    creation_date: str

class DataImportRequest(BaseModel):
    bucket_name: str
    prefix: str = ""

class PreprocessingStep(str, Enum):
    RESIZE = "resize"
    NORMALIZE = "normalize"
    AUGMENT = "augment"
    DENOISE = "denoise"
    CONTRAST_ENHANCE = "contrast_enhance"

class PreprocessingConfig(BaseModel):
    steps: List[PreprocessingStep]
    parameters: Dict[str, Any] = {}

class ModelType(str, Enum):
    RESNET50 = "resnet50"
    VGG16 = "vgg16"
    EFFICIENTNET = "efficientnet"
    DENSENET121 = "DenseNet121"
    VIT = "ViT"
    CUSTOM = "custom"

class TrainingConfig(BaseModel):
    model_type: ModelType
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"

class SageMakerTrainingConfig(BaseModel):
    model_name: str = "DenseNet121"
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 0.001
    num_classes: int = 8
    val_interval: int = 1
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    image_uri: str
    train_s3_uri: str
    test_s3_uri: str
    role_arn: str
    source_dir: str = "src"
    entry_point: str = "train_mlflow.py"
    framework_version: str = "1.12"
    py_version: str = "py38"
    mlflow_tracking_uri: Optional[str] = None

class ModelImportRequest(BaseModel):
    model_name: str
    s3_bucket: str
    s3_key: str
    model_format: str = "pytorch"
    framework_version: str = "2.1.1"
    num_classes: int = 2

class ImportedModel(BaseModel):
    model_id: str
    model_name: str
    s3_source: str
    local_path: str
    model_format: str
    num_classes: int
    imported_at: str
    status: str = "available"

class DeploymentConfig(BaseModel):
    endpoint_name: str
    instance_type: str = "ml.t2.medium"
    initial_instance_count: int = 1
    auto_scaling: bool = False

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TrainingJob(BaseModel):
    job_id: str
    status: JobStatus
    model_type: ModelType
    created_at: str
    completed_at: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None