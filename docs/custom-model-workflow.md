# Custom Model Workflow

A guide for AI researchers who want to add custom models to the MedML Training Dashboard without modifying the backend code.

## What You Provide (Zip File)

```
my-model-name.zip
├── model_def.py              # REQUIRED — architecture + MODELS registry
├── inference.py              # OPTIONAL — custom serving hooks (defaults to built-in)
├── train.py                  # OPTIONAL — custom training loop (defaults to built-in)
├── requirements.txt          # OPTIONAL — extra pip packages for training
└── sample_config.json        # OPTIONAL — suggested hyperparameters
```

## The Key Contract: `model_def.py`

Your `model_def.py` must have:
1. A `MODELS` dict that registers your model(s) with metadata
2. A `get_model(model_name, num_classes, **kwargs)` function that returns an instantiated PyTorch model

```python
"""
Custom model: Cardiac Attention UNet
"""
import torch.nn as nn

# This dict makes the model appear in the dashboard dropdown under "Custom Models"
MODELS = {
    "cardiac-attention-unet": {
        "description": "Attention UNet for cardiac MRI segmentation",
        "parameters": "4.2M",
        "input_size": "128x128x64",
        "pretrained": False,
        "task": "segmentation",       # "classification" or "segmentation"
        "dimension": "3D",            # "2D" or "3D"
    }
}

def get_model(model_name, num_classes=1, **kwargs):
    """Called by train.py and inference.py to instantiate the model."""
    if model_name == "cardiac-attention-unet":
        return CardiacAttentionUNet(in_channels=1, out_channels=num_classes)
    raise ValueError(f"Unknown model: {model_name}")


class CardiacAttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # ... your architecture here ...

    def forward(self, x):
        # ...
        return x
```

## MODELS Dict Fields

| Field | Required | Description |
|-------|----------|-------------|
| `description` | yes | Human-readable description |
| `parameters` | yes | Approximate parameter count (e.g. `"4.2M"`) |
| `input_size` | yes | Expected input dimensions (e.g. `"128x128x64"`) |
| `pretrained` | yes | Whether pretrained weights are used (`True`/`False`) |
| `task` | yes | `"classification"` or `"segmentation"` |
| `dimension` | yes | `"2D"` or `"3D"` |

## Step-by-Step Workflow

| Step | Action | What happens |
|------|--------|--------------|
| 1 | Upload zip via Training tab → "Upload Custom Model" | Extracted to `data/custom_models/<model_name>/`; model appears in dropdown under "Custom Models" |
| 2 | Select model from dropdown, fill training form, click Train | SageMaker job uses the custom `source_dir`; `train.py` imports `get_model()` from your `model_def.py` |
| 3 | Job completes | `model.tar.gz` bundles: weights + `model_def.py` + `inference.py` + `result.json` |
| 4 | Click Deploy on the completed job | Inference container loads `model_def.py` from artifact dynamically |
| 5 | Run Inference | `model_fn()` imports your `model_def.py` from the model artifact |
| 6 | Run Evaluation | Invokes the endpoint — works with any deployed model |
| 7 | Enable Monitoring | Captures I/O — works unchanged for any endpoint |

## Optional Files

### `inference.py` (custom serving hooks)

Only needed if your model requires:
- Custom input preprocessing (different from DICOM/NIfTI → tensor)
- Custom output format (different from predictions/class/confidence)
- Additional post-processing steps

If omitted, the built-in inference script handles:
- **Classification**: DICOM/PNG/JPG → resize → normalize → softmax → class probabilities
- **Segmentation**: NIfTI → resize to 128x128x64 → sigmoid → binary mask

### `train.py` (custom training loop)

Only needed if your model requires:
- Custom loss function
- Custom data loading/augmentation
- Multi-stage training
- Special optimizer configuration

If omitted, the built-in `train.py` handles:
- Data loading from S3 (class folders for classification, subject folders for segmentation)
- Standard training loop with validation
- Checkpoint saving (best model by validation metric)
- MLflow logging (when `MLFLOW_TRACKING_URI` is set)

### `requirements.txt` (extra dependencies)

List any packages not already in the BYOC training container. Pre-installed packages include:
- `torch`, `torchvision`
- `monai`
- `nibabel`, `SimpleITK`, `pydicom`, `itk`
- `mlflow`, `tensorboard`
- `tqdm`

### `sample_config.json` (suggested hyperparameters)

```json
{
    "epochs": 50,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "instance_type": "ml.g5.xlarge",
    "val_interval": 5
}
```

## What You Don't Need to Worry About

| Concern | Why it's handled |
|---------|-----------------|
| Docker images | Existing BYOC images dynamically import `model_def.py` |
| Deployment pipeline | `model.tar.gz` is self-contained with all serving code |
| Inference routing | Container loads your model architecture from the artifact |
| Monitoring | Endpoint-agnostic — works with any model |
| Experiment tracking | MLflow integration is automatic when URI is configured |
| Result provenance | `result.json` is written automatically with training metadata |

## Data Format Requirements

### Classification
```
s3://bucket/your-data/
├── train/
│   ├── class_a/
│   │   ├── image1.dcm
│   │   └── image2.png
│   └── class_b/
│       └── image3.dcm
├── val/
│   ├── class_a/
│   └── class_b/
└── test/        # optional
    ├── class_a/
    └── class_b/
```

### Segmentation
```
s3://bucket/your-data/
├── train/
│   ├── subject_001/
│   │   ├── img.nii.gz
│   │   └── seg.nii.gz
│   └── subject_002/
│       ├── img.nii.gz
│       └── seg.nii.gz
└── val/
    └── subject_003/
        ├── img.nii.gz
        └── seg.nii.gz
```

## Validation

When you upload a zip, the dashboard validates:
1. `model_def.py` exists in the archive
2. `model_def.py` contains a `MODELS` dict (parsed via AST, no import needed)
3. No path traversal (`..`) or absolute paths in the archive
4. File extension is `.zip`

## Example: Adding a 3D ResNet for Classification

```python
# model_def.py
from monai.networks.nets import ResNet

MODELS = {
    "resnet3d-18": {
        "description": "3D ResNet-18 for volumetric classification",
        "parameters": "33M",
        "input_size": "64x64x64",
        "pretrained": False,
        "task": "classification",
        "dimension": "3D",
    }
}

def get_model(model_name, num_classes=8, **kwargs):
    if model_name == "resnet3d-18":
        return ResNet(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,
        )
    raise ValueError(f"Unknown model: {model_name}")
```

Then zip it:
```bash
zip resnet3d-18.zip model_def.py
```

Upload via the dashboard, select "resnet3d-18" from the model dropdown, and train.
