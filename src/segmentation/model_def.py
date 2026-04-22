"""
Model definitions for 3D medical image segmentation.

To add a new model:
1. Add an entry to MODELS dict with its metadata
2. Add a case in ModelDef.get_model()
"""
import torch.nn as nn
from monai.networks.nets import SegResNet, SwinUNETR, UNet

# Registry — drives the dashboard model list
MODELS = {
    "segresnet": {
        "description": "SegResNet for 3D medical image segmentation",
        "parameters": "5M",
        "input_size": "128x128x64",
        "pretrained": False,
        "task": "segmentation",
        "dimension": "3D",
    },
    "swinunetr": {
        "description": "SwinUNETR transformer for 3D medical image segmentation",
        "parameters": "62M",
        "input_size": "128x128x64",
        "pretrained": False,
        "task": "segmentation",
        "dimension": "3D",
    },
    "unet": {
        "description": "UNet baseline for 3D medical image segmentation",
        "parameters": "31M",
        "input_size": "128x128x64",
        "pretrained": False,
        "task": "segmentation",
        "dimension": "3D",
    },
}


class ModelDef:
    def __init__(self, num_classes: int = 1, model_name: str = "segresnet"):
        self.num_classes = num_classes
        self.model_name  = model_name.lower()

    def get_model(self) -> nn.Module:
        name = self.model_name

        if name == "segresnet":
            return SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.num_classes,
                init_filters=16,
                dropout_prob=0.2,
            )

        if name == "swinunetr":
            return SwinUNETR(
                img_size=(128, 128, 64),
                in_channels=1,
                out_channels=self.num_classes,
                feature_size=24,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
            )

        if name == "unet":
            return UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.num_classes,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )

        raise ValueError(f"Unsupported segmentation model: {self.model_name}")
