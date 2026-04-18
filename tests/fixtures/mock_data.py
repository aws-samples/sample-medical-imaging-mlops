"""Mock data fixtures for testing"""
import numpy as np
from PIL import Image
from io import BytesIO


def create_mock_image(width=224, height=224, channels=3):
    """Create a mock medical image"""
    img_array = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    return Image.fromarray(img_array)


def create_mock_dicom_metadata():
    """Create mock DICOM metadata"""
    return {
        "PatientID": "TEST001",
        "StudyDate": "20260220",
        "Modality": "CT",
        "ImageType": ["ORIGINAL", "PRIMARY"],
        "Rows": 512,
        "Columns": 512
    }


def create_mock_training_config():
    """Create mock training configuration"""
    return {
        "model_type": "resnet50",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss_function": "cross_entropy"
    }


def create_mock_aws_response(service, operation):
    """Create mock AWS API responses"""
    responses = {
        "s3": {
            "list_buckets": {
                "Buckets": [
                    {"Name": "test-bucket-1", "CreationDate": "2026-01-01T00:00:00Z"},
                    {"Name": "test-bucket-2", "CreationDate": "2026-01-02T00:00:00Z"}
                ]
            },
            "list_objects_v2": {
                "Contents": [
                    {"Key": "image1.dcm", "Size": 1024, "LastModified": "2026-01-01T00:00:00Z"},
                    {"Key": "image2.dcm", "Size": 2048, "LastModified": "2026-01-02T00:00:00Z"}
                ]
            }
        },
        "sagemaker": {
            "create_model": {"ModelArn": "arn:aws:sagemaker:us-east-1:123456789012:model/test-model"},
            "create_endpoint": {"EndpointArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint"}
        }
    }
    return responses.get(service, {}).get(operation, {})
