# Training Guide

This guide covers launching classification and segmentation training jobs — both locally and on SageMaker.

## Available Models

### Classification

| Model | Notes |
|---|---|
| ResNet50 | General-purpose CNN, good baseline |
| VGG16 | Classic architecture, larger memory footprint |
| EfficientNet | Best accuracy/compute trade-off |
| DenseNet121 | Dense connections, good for small datasets |
| ViT | Vision Transformer, requires more data |
| Custom | Upload your own `model_def.py` |

### Segmentation

| Model | Notes |
|---|---|
| SegResNet | MONAI's residual encoder-decoder, fast training |
| SwinUNETR | Transformer-based, state-of-the-art for 3D |
| UNet | Classic encoder-decoder architecture |

## Data Format

Upload your training data to S3 in the following structure:

### Classification

```
s3://your-bucket/dataset-name/
├── train/
│   ├── class_a/
│   │   ├── image001.dcm
│   │   └── image002.dcm
│   └── class_b/
│       ├── image003.dcm
│       └── image004.dcm
├── val/
│   ├── class_a/
│   └── class_b/
└── test/        (optional — used for evaluation)
    ├── class_a/
    └── class_b/
```

- Each subfolder name becomes a class label
- `num_classes` is derived automatically from the folder structure
- Supported file formats: DICOM (`.dcm`), PNG, JPG, NIfTI

### Segmentation

```
s3://your-bucket/dataset-name/
├── train/
│   ├── subject_001/
│   │   ├── img.nii.gz
│   │   └── mask.nii.gz
│   └── subject_002/
│       ├── img.nii.gz
│       └── mask.nii.gz
└── val/
    ├── subject_003/
    └── subject_004/
```

- Each subject folder contains an image volume and a binary mask
- Output is always 1 channel (binary segmentation)

## Launching a SageMaker Training Job

### Step 1 — Open the Training tab

Navigate to `http://localhost:8000` and click the **Training** tab.

### Step 2 — Fill out the training form

| Field | Description | Example |
|---|---|---|
| **Task Type** | `classification` or `segmentation` | `classification` |
| **Data S3 URI** | S3 path to your dataset root | `s3://my-bucket/spine-xray/` |
| **Model** | Architecture to train | `DenseNet121` |
| **Epochs** | Number of training epochs | `50` |
| **Batch Size** | Mini-batch size | `16` |
| **Learning Rate** | Initial learning rate | `0.0001` |
| **Instance Type** | SageMaker instance | `ml.g5.xlarge` |
| **Training Image URI** | ECR URI of the BYOC container | from `.env.local` |
| **Role ARN** | SageMaker execution role | from `.env.local` |
| **MLflow Tracking URI** | (Optional) MLflow server URL | `http://your-mlflow:5000` |

### Step 3 — Submit

Click **Start Training**. The job appears in the Active tab with status `InProgress`.

### Step 4 — Monitor progress

- The job list auto-refreshes status from AWS for active jobs
- Once complete, the job moves to the Completed tab with billable seconds and estimated cost

## MLflow Integration

To enable experiment tracking:

1. Set the `MLFLOW_TRACKING_URI` environment variable (in `.env` or the training form)
2. The training script automatically logs:
   - Hyperparameters (model, epochs, learning rate, batch size)
   - Per-epoch metrics (train loss, train accuracy, val accuracy, val loss)
   - Best model checkpoint as an artifact
   - `result.json` with training provenance

For SageMaker-managed MLflow tracking servers, use the dashboard's presigned URL feature to get a temporary access URL.

## Fine-Tuning from a Previous Job

To continue training from a completed job's checkpoint:

1. Find the completed job's model artifact S3 URI (e.g. `s3://bucket/output/job-name/output/model.tar.gz`)
2. In the training form, set **Checkpoint S3 URI** to that path
3. The training script will:
   - Download and extract the archive
   - Load `best_model.pth` (or `model.pth`) — restoring model weights and optimizer state
   - Set `start_epoch` to the checkpoint's epoch + 1
   - Record the source URI in `result.json` under `fine_tuned_from`

If the checkpoint cannot be loaded, training falls back to starting from scratch with a warning.

## Job Management

### Tabs

| Tab | Shows |
|---|---|
| **Active** | Jobs in `InProgress` or `Stopping` state |
| **Completed** | Successfully finished jobs |
| **Failed** | Jobs that errored out |
| **Archived** | Jobs you've hidden from the default view |

### Actions

| Action | Description |
|---|---|
| **Sync from AWS** | Pull up to 100 jobs from AWS and merge into local store. Use `?name_contains=` to filter. |
| **Archive** | Hide a job from the default list (still accessible in the Archived tab) |
| **Comment** | Attach a free-text note to any job |
| **Delete Artifacts** | Delete `model.tar.gz` from S3 and mark the job as `artifacts_deleted` |
| **Remove** | Delete a job from local tracking (only allowed when not active) |
| **Fetch Result** | Download `result.json` from the model artifact (reconstructs if missing) |

### Search

Use the search bar to filter jobs by name. The `?name_contains=` query parameter also works with the Sync from AWS operation to pull only matching jobs.

## Custom Models

To use a custom architecture:

### Step 1 — Create your model code

Create a zip file containing at minimum a `model_def.py` with a `get_model(num_classes)` function:

```python
# model_def.py
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

### Step 2 — Upload the zip

```bash
curl -X POST "http://localhost:8000/api/training/models/upload-code?model_name=my_custom_model" \
  -F "file=@my_model.zip"
```

Or use the dashboard's Upload Custom Model form.

### Step 3 — Select in training form

Choose your custom model from the model dropdown when launching a job. The code is extracted to `data/custom_models/<model_name>/` and bundled into the training container.

## Local Training

For quick experiments without SageMaker:

1. Select **Local** as the training target
2. Fill in the same form fields (data path can be local or S3)
3. Training runs on the dashboard server with MLflow and TensorBoard tracking
4. Results appear in the local training history

Local training is useful for prototyping but limited by the server's hardware. Use SageMaker for GPU-accelerated training on larger datasets.
