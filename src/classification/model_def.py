"""
Model definitions for medical image classification.

To add a new model:
1. Add an entry to MODELS dict with its metadata
2. Add a case in ModelDef.get_model()
"""
import torch.nn as nn
from monai.networks.nets import DenseNet121, ViT
import torchvision.models as tv_models

# Registry — drives the dashboard model list
MODELS = {
    "densenet121": {
        "description": "DenseNet-121 for 2D medical image classification",
        "parameters": "8M",
        "input_size": "256x256",
        "pretrained": True,
        "task": "classification",
        "dimension": "2D",
    },
    "vit": {
        "description": "Vision Transformer for 2D medical image classification",
        "parameters": "86M",
        "input_size": "256x256",
        "pretrained": True,
        "task": "classification",
        "dimension": "2D",
    },
    "resnet50": {
        "description": "ResNet-50 for 2D medical image classification",
        "parameters": "25.6M",
        "input_size": "224x224",
        "pretrained": True,
        "task": "classification",
        "dimension": "2D",
    },
    "vgg16": {
        "description": "VGG-16 for 2D medical image classification",
        "parameters": "138M",
        "input_size": "224x224",
        "pretrained": True,
        "task": "classification",
        "dimension": "2D",
    },
    "efficientnet": {
        "description": "EfficientNet-B0 for efficient 2D medical image classification",
        "parameters": "5.3M",
        "input_size": "224x224",
        "pretrained": True,
        "task": "classification",
        "dimension": "2D",
    },
}


class ModelDef:
    def __init__(self, num_classes: int, model_name: str):
        self.num_classes = num_classes
        self.model_name  = model_name.lower()

    def get_model(self) -> nn.Module:
        name = self.model_name

        if name == "densenet121":
            return DenseNet121(spatial_dims=2, in_channels=1, out_channels=self.num_classes)

        if name == "vit":
            return ViT(
                in_channels=1,
                img_size=(256, 256, 1),
                patch_size=(16, 16, 1),
                hidden_size=768,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
                classification=True,
                num_classes=self.num_classes,
            )

        if name == "resnet50":
            m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
            m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            m.fc    = nn.Linear(m.fc.in_features, self.num_classes)
            return m

        if name == "vgg16":
            m = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
            m.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            m.classifier[-1] = nn.Linear(4096, self.num_classes)
            return m

        if name == "efficientnet":
            m = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
            m.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, self.num_classes)
            return m

        raise ValueError(f"Unsupported classification model: {self.model_name}")
