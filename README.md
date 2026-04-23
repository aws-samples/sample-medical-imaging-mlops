# MedML Training Dashboard

A FastAPI web dashboard for managing medical image classification workflows on AWS. Provides a UI for data ingestion, model training (local and SageMaker), and deployment.

## Features

- **AWS Integration** — connect via environment credentials or explicit keys, browse S3 buckets; generate presigned URLs for SageMaker MLflow tracking servers via `create_presigned_mlflow_tracking_server_url` (accepts both full ARNs and plain server names — the server name is extracted from the ARN automatically; presigned URLs expire after 5 minutes so a fresh one is generated on each request)
- **Local Training** — train ResNet50, VGG16, EfficientNet, DenseNet121, ViT, or custom models with MLflow and TensorBoard tracking
- **SageMaker Training** — launch BYOC training jobs on SageMaker (e.g. `ml.g5.xlarge`) with optional MLflow integration; both the classification script (`src/classification/train.py`) and the segmentation script (`src/segmentation/train.py`) enable MLflow automatically when `MLFLOW_TRACKING_URI` is set — they log hyperparameters, per-epoch metrics, the best model checkpoint, and the final model as artifacts; the best checkpoint (`best_model.pth`) saved by the classification script includes full training state: `model` weights, `optimizer` state, `epoch`, `val_acc`, `train_acc`, `train_loss`, `val_loss`, `model_name`, and `num_classes` — making it self-contained for resuming training or auditing; a separate weights-only checkpoint (`model_deploy.pth`) is also written at the end of training — it contains only the model state dict (no optimizer or scheduler state), making it roughly 50% smaller than `model.pth` and suitable for deployment; the segmentation script always uses 1 output channel (binary segmentation) and names the MLflow experiment `segmentation-<model_name>`; `num_classes` for classification is derived dynamically from the training data folder structure rather than passed as a hyperparameter; job list auto-refreshes status, artifacts, and failure reasons from AWS on each request; jobs are persisted to `data/sagemaker_jobs.json` and survive server restarts; completed jobs include billable seconds, an estimated cost in USD, `creation_time`, and `training_end_time`; each job in the list exposes the `mlflow_tracking_uri` it was submitted with (if any); a JSON training report is written to `data/training_reports/<job_name>.json` on completion; at the end of a classification training run, `inference.py` and `requirements.txt` are automatically copied from the training script directory into `model_dir`, and a `result.json` file is written there with training provenance (model name, num classes, final epoch, train/val accuracy, best val accuracy, data S3 URI, task type, dimension, learning rate, batch size) — SageMaker then bundles all of these into `model.tar.gz` alongside the weights, so the deployment service can point directly at the output artifact without any post-processing merge step; `result.json` is also logged as an MLflow artifact when MLflow tracking is enabled; job records support additional lifecycle operations: `remove_job` deletes a job from local tracking (only allowed when the job is not active), `archive_job` marks a job as hidden from the default list (`archived: true`), `set_comment` attaches a free-text note to a job, `delete_model_artifacts` deletes the `model.tar.gz` from S3 and sets `artifacts_deleted: true` on the record, and `store_result` persists the `result.json` contents against the job entry; `fetch_result_json` downloads `model.tar.gz` from S3, extracts `result.json`, stores it on the job record via `store_result`, and returns it — safe to call multiple times (skips the download if `result` is already present on the record); the job list response includes `archived`, `comment`, `artifacts_deleted`, and `result` fields for each job
- **Deployment** — deploy trained models to SageMaker endpoints; creates a SageMaker model, endpoint config, and endpoint in sequence; tracks endpoint status (`Creating` → `InService` / `Failed`) and exposes the invocation URL once in service; deployments are persisted to `data/deployments.json` and survive server restarts; supports status polling, deletion (deletes the endpoint and its associated endpoint config), and listing all active deployments; on deletion the local record is always removed from `data/deployments.json` regardless of whether the AWS API call succeeds — a `try/finally` guarantees cleanup even if the endpoint or config delete fails; if an endpoint is deleted outside the dashboard (e.g. via the AWS console or CLI), the next status poll detects the missing endpoint and marks the local record as `"Deleted"` rather than raising an error; the deployment list endpoint refreshes status from AWS for all entries except those already in `Deleted` state — `InService`, `Failed`, and `Deleting` endpoints are all re-polled on every list request so transient status changes (including deletion progress) are reflected immediately; `Deleted` entries are always filtered out of the list response; the deploy request accepts `model_artifacts`, `image_uri`, and `role_arn` as optional fields — omitting `model_artifacts` lets you import or track an existing endpoint without providing a new artifact; `inference_script` (default `"inference.py"`) specifies which serving entry point to use inside the container; the deployment service points directly at the original `model.tar.gz` — no repacking step is needed because `train.py` already copies `inference.py` and `requirements.txt` into `model_dir` and writes `result.json` at the end of training, so SageMaker bundles them all into the artifact automatically; the PyTorch container loads the entry point via `SAGEMAKER_PROGRAM` — this env var is set on real-time endpoints; set `async_endpoint: true` to create a SageMaker async endpoint — `output_s3_uri` is required in that case and `max_concurrent` (default `10`) controls the `MaxConcurrentInvocationsPerInstance` setting; async endpoints still use the separate sourcedir upload path
- **Fairness Analysis** — `FairnessService` (`app/services/fairness_service.py`) scans DICOM files under an S3 prefix and extracts demographic distributions for bias auditing; reads up to 200 files by default (configurable via `max_files`) using `stop_before_pixels=True` for efficiency; extracts `PatientSex`, `PatientAge`, `EthnicGroup`, `PatientRace`, `Modality`, and `BodyPartExamined` tags; age strings in DICOM format (`045Y`, `6M`, `2W`) are normalised to integer years and bucketed into ranges (`0-17`, `18-29`, `30-44`, `45-59`, `60-74`, `75+`); returns per-attribute value counts, age bucket distribution, per-attribute missing rates (% of files where the tag was absent or empty), a capped sample of up to 50 file-level rows, and summary counts (`total_found`, `total_scanned`, `errors`)

- **Endpoint Invocation** — `InferenceService` (`app/services/inference_service.py`) invokes SageMaker real-time endpoints via the `sagemaker-runtime` client; accepts a JSON payload and an optional `output_s3_uri` — if provided, the raw response is saved to S3 at `<prefix>/<endpoint_name>-result.json` and the S3 path is included in the returned result; the `task_type` field is always appended to the response for downstream routing

## Stack

- **Backend** — FastAPI, boto3, PyTorch, MONAI, MLflow
- **Frontend** — Jinja2 templates, vanilla JS
- **Infrastructure** — Docker, SageMaker, S3

### SageMaker Container Dependencies (`src/classification/requirements.txt`)

Dependencies bundled into the BYOC training container for classification jobs:

| Package | Purpose |
|---|---|
| `monai` | Medical imaging transforms and model architectures |
| `torch` / `torchvision` | Deep learning framework |
| `nibabel` | NIfTI file I/O |
| `SimpleITK` | Medical image I/O and processing |
| `pydicom` | DICOM file I/O |
| `itk` | ITK-based image I/O (used by MONAI's `ITKReader`) |
| `tensorboard` | Training visualization |
| `tqdm` | Progress bars |
| `smdebug` | SageMaker Debugger integration |
| `mlflow` | Experiment tracking (optional — enabled via `MLFLOW_TRACKING_URI` env var) |

### SageMaker Container Dependencies (`src/segmentation/requirements.txt`)

Dependencies bundled into the BYOC training container for segmentation jobs:

| Package | Purpose |
|---|---|
| `monai` | Medical imaging transforms and model architectures (SegResNet, SwinUNETR, UNet) |
| `torch` / `torchvision` | Deep learning framework |
| `nibabel` | NIfTI file I/O |
| `SimpleITK` | Medical image I/O and processing |
| `einops` | Tensor rearrangement (required by SwinUNETR) |
| `itk` | ITK-based image I/O (used by MONAI's `ITKReader`) |
| `tensorboard` | Training visualization |
| `tqdm` | Progress bars |
| `smdebug` | SageMaker Debugger integration |
| `mlflow` | Experiment tracking (optional — enabled via `MLFLOW_TRACKING_URI` env var) |

## SageMaker Inference

Two inference scripts implement the four SageMaker serving hooks — one for classification endpoints and one for segmentation endpoints.

### Classification — `src/inference/classification/inference.py`

| Function | Responsibility |
|---|---|
| `model_fn(model_dir)` | Loads model weights and MONAI transforms from the model artifact directory; populates module-level `_LABEL_DICT` and `_IDX_TO_CLASS` |
| `input_fn(request_body, content_type)` | Parses the request — accepts raw DICOM bytes (`application/dicom`) or a JSON payload with a `file_path` (S3 URI, local path, or base64) |
| `predict_fn(file_path, model_components)` | Runs the forward pass and returns raw logits |
| `output_fn(prediction, accept)` | Converts logits to a JSON response with batched `predictions`, `predicted_class`, and `confidence` arrays |

#### Supported input formats

- **DICOM** — raw bytes with `Content-Type: application/dicom`
- **S3 URI** — `{"file_path": "s3://bucket/key.dcm"}`
- **Local path** — `{"file_path": "/path/to/image.dcm"}`
- **Base64** — `{"file_path": "<base64-encoded bytes>"}`

#### Model loading

Dynamically imports `model_def.py` from the model artifact directory and falls back to DenseNet121 if not present. Weights are loaded from `best_model.pth` or `model.pth` (in that order). `model_deploy.pth` (weights-only, ~50% smaller) is written by `train.py` at the end of training and can be used as a lighter alternative for deployment. The `MODEL_NAME` environment variable controls which architecture is instantiated when using `model_def.py`.

#### Example response

Response format matches `inference_realtime.py` — all fields are batched arrays:

```json
{
  "predictions": [[0.1034, 0.1202, 0.1366, 0.0866, 0.1258, 0.1408, 0.1155, 0.1713]],
  "predicted_class": [7],
  "confidence": [0.1713]
}
```

### Segmentation — `src/inference/segmentation/inference.py`

| Function | Responsibility |
|---|---|
| `model_fn(model_dir)` | Loads segmentation model weights and MONAI transforms; resizes input to `(128, 128, 64)` and clips CT HU range `[-100, 500]` |
| `input_fn(request_body, content_type)` | Returns a local file path to a NIfTI volume — accepts raw NIfTI bytes (`application/octet-stream`), a JSON payload with `s3_uri` or `file_path`, or base64-encoded NIfTI data |
| `predict_fn(file_path, model_components)` | Runs the forward pass with sigmoid activation and 0.5 threshold; returns a binary mask as a `uint8` numpy array |
| `output_fn(mask, accept)` | Saves the mask as a NIfTI file and returns a JSON response with the base64-encoded NIfTI bytes and volume statistics |

#### Supported input formats

- **Raw NIfTI bytes** — binary with `Content-Type: application/octet-stream`
- **S3 URI** — `{"s3_uri": "s3://bucket/subject/img.nii.gz"}`
- **Local path** — `{"file_path": "/path/to/volume.nii.gz"}`
- **Base64** — `{"file_path": "<base64-encoded NIfTI bytes>"}`

#### Model loading

Dynamically imports `model_def.py` from the model artifact directory and falls back to a default `SegResNet` (`spatial_dims=3, in_channels=1, out_channels=1, init_filters=16`) if not present. Weights are loaded from `best_model.pth` or `model.pth`. The `MODEL_NAME` environment variable selects the architecture when using `model_def.py`.

#### Example response

```json
{
  "mask_nifti_b64": "<base64-encoded .nii.gz bytes>",
  "shape": [128, 128, 64],
  "voxels_positive": 14823,
  "voxels_total": 1048576,
  "volume_fraction": 0.0141
}
```

## Getting Started

### Prerequisites

- Python 3.11+
- AWS credentials configured (`~/.aws/credentials` or environment variables)

### Local Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env

# Start the app
python main.py
```

App runs at `http://localhost:8000`

### With Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| TensorBoard | http://localhost:6006 |

### With startup script

```bash
chmod +x start.sh
./start.sh
```

This starts MLflow, TensorBoard, and the FastAPI app together.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key | — |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | — |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `MLFLOW_TRACKING_URI` | MLflow server URI | `http://localhost:5000` |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment name | `medical_imaging_experiments` |
| `TENSORBOARD_LOG_DIR` | TensorBoard log directory | `./logs/tensorboard` |

## Project Structure

```
├── app/
│   ├── models/         # Pydantic schemas
│   ├── routers/        # API route handlers (aws, training, deployment, inference)
│   └── services/       # Business logic (AWS, training, SageMaker, inference)
│       ├── aws_service.py                  # boto3 client management
│       ├── training_service.py             # Local training and model import
│       ├── sagemaker_training_service.py   # SageMaker BYOC job lifecycle
│       ├── deployment_service.py           # SageMaker endpoint deployment
│       ├── inference_service.py            # Real-time endpoint invocation + optional S3 result save
│       └── fairness_service.py             # DICOM demographic scanning and fairness analysis
├── static/             # CSS and JS
├── templates/          # Jinja2 HTML templates
├── data/
│   ├── raw/                    # Raw input data (gitignored)
│   ├── processed/              # Preprocessed datasets (gitignored)
│   ├── models/                 # Imported model artifacts (gitignored)
│   ├── sagemaker_jobs.json     # Persisted SageMaker job records (auto-created)
│   ├── deployments.json        # Persisted endpoint deployment records (auto-created)
│   └── training_reports/       # Per-job JSON reports written on completion (auto-created)
├── logs/               # TensorBoard logs (gitignored)
├── src/
│   ├── classification/ # SageMaker BYOC training scripts (classification)
│   ├── segmentation/   # SageMaker BYOC training scripts (segmentation)
│   └── inference/      # SageMaker serving hooks
│       ├── classification/
│       │   └── inference.py        # Classification endpoint (DICOM → batched predictions array)
│       └── segmentation/
│           └── inference.py        # Segmentation endpoint (NIfTI → binary mask + volume stats)
├── main.py             # App entry point
├── Dockerfile
└── docker-compose.yml
```

## Testing

### Unit / mocked tests

```bash
pytest
# with coverage
coverage run -m pytest && coverage report
```

### Integration tests

Integration tests hit real AWS resources and require valid credentials.

#### Local inference — `tests/integration/test_local_inference.py`

Runs the classification inference pipeline directly (no SageMaker endpoint required). Useful for validating `model_fn`, `input_fn`, `predict_fn`, and `output_fn` locally before deploying.

| Environment variable | Default | Description |
|---|---|---|
| `TEST_LOCAL_IMAGE` | `data/test_data/classification-spine/412cb981a940a9689347c614cb1489dd.dcm` | Path to a local DICOM, PNG, or JPG file |
| `MODEL_DIR` | `data/models` | Directory containing a `.pth` model checkpoint |

```bash
# Run with defaults (requires a model in data/models/)
pytest tests/integration/test_local_inference.py -v -s

# Run with a specific image and model
TEST_LOCAL_IMAGE=/path/to/image.dcm MODEL_DIR=/path/to/model pytest tests/integration/test_local_inference.py -v -s
```

The suite runs three checks:
- `test_local_file_exists` — confirms the test image is present on disk
- `test_input_fn_local_file` — calls `input_fn` with a `file_path` payload and asserts a valid file path is returned
- `test_full_inference_local_file` — runs the full pipeline (`input_fn` → `predict_fn` → `output_fn`) and validates `predicted_class`, `confidence`, and `probabilities` in the response

> The `model_components` fixture skips all model-dependent tests automatically if no `.pth` file is found in `MODEL_DIR`.

#### Endpoint inference — `tests/integration/test_endpoint_inference.py`

Invokes a live SageMaker endpoint with a real S3 image or local file and validates the response.

| Environment variable | Default | Description |
|---|---|---|
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region |
| `ENDPOINT_NAME` | `test-endp` | Name of the `InService` endpoint to test |
| `TEST_IMAGE_S3_URI` | VinDr-SpineXR DICOM sample | S3 URI of the test image |
| `TEST_LOCAL_IMAGE` | _(unset)_ | Path to a local DICOM, PNG, or JPG file; if unset a synthetic 224×224 PNG is generated |

```bash
# Run with defaults
pytest tests/integration/test_endpoint_inference.py -v -s

# Run against a specific endpoint with a local image
ENDPOINT_NAME=my-endpoint TEST_LOCAL_IMAGE=/path/to/image.dcm pytest tests/integration/test_endpoint_inference.py -v -s

# Convenience wrapper (also accepts an optional local image path as first argument)
./tests/integration/run_inference_tests.sh /path/to/image.dcm
```

The suite runs four checks:
- `test_endpoint_is_inservice` — confirms the endpoint status is `InService` before invoking
- `test_invoke_with_s3_uri` — sends a JSON payload with `{"file_path": "<s3-uri>"}` (the key expected by `inference_realtime.py`) and validates the batched response format: `predictions` (list of probability arrays), `predicted_class` (list of class indices), and `confidence` (list of floats); maps the predicted index to a human-readable label using the VinDr-SpineXR label set
- `test_invoke_response_time` — asserts the endpoint responds within 10 seconds
- `test_invoke_with_local_file` — base64-encodes a local file (or a synthetic PNG if `TEST_LOCAL_IMAGE` is unset) and validates the same response fields

> These tests require the endpoint to already be deployed and `InService`. Deploy first via the dashboard or `/api/deployment`.

#### Root-level quick test — `test_endpoint.py`

A minimal script at the project root for a fast sanity-check against a live endpoint. No pytest required — just run it directly.

The script mirrors the notebook workflow: it downloads the source image from S3 and re-uploads it to the SageMaker default bucket before invoking the endpoint.

```bash
# Use defaults (endpoint: test-endp-class-vindr, region: us-east-1)
python test_endpoint.py

# Override endpoint and/or image
python test_endpoint.py --endpoint my-endpoint
python test_endpoint.py --endpoint my-endpoint --image s3://bucket/path/image.dcm --region eu-west-1

# Also test local file inference (raw DICOM bytes)
python test_endpoint.py --endpoint my-endpoint --local /path/to/image.dcm
```

| Argument | Default | Description |
|---|---|---|
| `--endpoint` | `test-endp-class-vindr` | SageMaker endpoint name |
| `--image` | VinDr-SpineXR DICOM sample | Source S3 URI — downloaded and re-uploaded to the SageMaker default bucket before invoking |
| `--local` | _(unset)_ | Path to a local DICOM file; if provided, also invokes the endpoint with raw bytes (`Content-Type: application/dicom`) |
| `--region` | `us-east-1` | AWS region |

Uploads the source image to the SageMaker default bucket but does **not** automatically invoke the endpoint with it (the S3 inference call is currently disabled). If `--local` is provided, the endpoint is invoked with the raw DICOM bytes and the result is printed as a formatted probability bar chart. Requires valid AWS credentials in the environment.

#### Quick endpoint invocation — `tests/invoke_endpoint.py`

A lightweight CLI script for invoking a SageMaker endpoint directly, without running the full pytest suite. Accepts either a local image file or an S3 URI — at least one must be provided.

```bash
# Invoke with a local image file
python tests/invoke_endpoint.py --image /path/to/image.dcm

# Invoke with an S3 URI (no local file needed)
python tests/invoke_endpoint.py --s3-uri s3://my-bucket/images/scan.dcm

# Specify endpoint and region explicitly
python tests/invoke_endpoint.py --image /path/to/image.dcm --endpoint my-endpoint --region us-east-1

# Save the JSON result to a file
python tests/invoke_endpoint.py --image /path/to/image.dcm --output result.json
```

| Argument | Default | Description |
|---|---|---|
| `--image` | _(optional)_ | Path to a local DICOM, PNG, or JPG file (base64-encoded before sending) |
| `--s3-uri` | _(optional)_ | S3 URI of the image (e.g. `s3://bucket/key.dcm`); sent directly as a JSON payload |
| `--endpoint` | `$ENDPOINT_NAME` or `test-endp` | SageMaker endpoint name |
| `--region` | `$AWS_DEFAULT_REGION` or `us-east-1` | AWS region |
| `--output` | _(unset)_ | Optional path to save the JSON response |

One of `--image` or `--s3-uri` is required. When using `--image`, the file is base64-encoded and sent as `{"file_path": "<b64>", "filename": "..."}`. When using `--s3-uri`, the payload is `{"file_path": "s3://..."}` and the endpoint fetches the file directly from S3 — this matches the key expected by `inference_realtime.py`. Requires valid AWS credentials in the environment.

## API Endpoints

| Prefix | Description |
|---|---|
| `/api/aws` | AWS credentials and S3 operations |
| `/api/training` | Local and SageMaker training jobs |
| `/api/deployment` | Model deployment to SageMaker |
| `/api/inference` | Real-time endpoint invocation and listing |

### `/api/inference` routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/inference/invoke` | Invoke a SageMaker real-time endpoint; accepts `endpoint_name`, `task_type` (`classification` \| `segmentation`), `input_type` (`s3_uri` \| `file_path`), `input_value`, and optional `output_s3_uri` |
| `GET` | `/api/inference/endpoints` | List all `InService` SageMaker endpoints |
