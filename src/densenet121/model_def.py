"""
Model definitions for medical image classification.
Supports DenseNet121 and ViT via MONAI.
"""
from monai.networks.nets import DenseNet121, ViT


class ModelDef:
    def __init__(self, num_classes: int, model_name: str = "DenseNet121"):
        self.num_classes = num_classes
        self.model_name = model_name

    def get_model(self):
        if self.model_name == "DenseNet121":
            return DenseNet121(
                spatial_dims=2,
                in_channels=1,
                out_channels=self.num_classes,
            )
        elif self.model_name == "ViT":
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
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported. Choose DenseNet121 or ViT.")
