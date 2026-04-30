# MedML Training Dashboard

A FastAPI web dashboard for managing medical image classification workflows on AWS. Provides a UI for data ingestion, model training (local and SageMaker), and deployment.

## Features

- **AWS Integration** — connect via environment credentials or explicit keys, browse S3 buckets; generate presigned URLs for SageMaker MLflow tracking servers via `create_presigned_mlflow_tracking_server_url` (accepts both full ARNs and plain server names — the server name is extracted from the ARN automatically; presigned URLs expire after 5 minutes so a fresh one is generated on each request)
- **Local Training** — train ResNet50, VGG16, EfficientNet, DenseNet121, ViT, or custom models with MLflow and TensorBoard tracking
- **SageMaker Training** — launch BYOC training jobs on SageMaker (e.g. `ml.g5.xlarge`) with optional MLflow integration; both the classification script (`src/classification/train.py`) and the segmentation script (`src/segmentation/train.py`) enable MLflow automatically when `MLFLOW_TRACKING_URI` is set — they log hyperparameters, per-epoch metrics, the best model checkpoint, and the final model as artifacts; the best checkpoint (`best_model.pth`) saved by the classification script includes full training state: `model` weights, `optimizer` state, `epoch`, `val_acc`, `train_acc`, `train_loss`, `val_loss`, `model_name`, and `num_classes` — making it self-contained for resuming training or auditing; a separate weights-only checkpoint (`model_deploy.pth`) is also written at the end of training — it contains only the model state dict (no optimizer or scheduler state), making it roughly 50% smaller than `model.pth` and suitable for deployment; the segmentation script always uses 1 output channel (binary segmentation) and names the MLflow experiment `segmentation-<model_name>`; `num_classes` for classification is derived dynamically from the training data folder structure rather than passed as a hyperparameter; job list automatically refreshes status from AWS for any jobs currently in `InProgress` or `Stopping` state (when AWS credentials are configured), so active jobs always reflect their latest status without a separate poll; pass `?local_only=true` to skip the AWS refresh and return only the locally-cached state — useful for fast polling or when AWS credentials are unavailable; completed, failed, and other terminal-state jobs are returned directly from the local store — use the dedicated `/api/training/sagemaker/jobs/{job_name}/status` endpoint or `sync_from_aws` to force a refresh for those; jobs are persisted to `data/sagemaker_jobs.json` and survive server restarts; if `data/sagemaker_jobs.json` is missing on startup, the service automatically bootstraps the jobs store from any JSON files found in `data/training_reports/` — each report is parsed and a minimal job record is reconstructed, then the rebuilt store is saved to `data/sagemaker_jobs.json` so subsequent restarts load normally; completed jobs include billable seconds, an estimated cost in USD, `creation_time`, and `training_end_time`; each job in the list exposes the `mlflow_tracking_uri` it was submitted with (if any); a JSON training report is written to `data/training_reports/<job_name>.json` on completion; at the end of a classification training run, `inference.py` and `requirements.txt` are automatically copied from the training script directory into `model_dir`, and a `result.json` file is written there with training provenance (model name, num classes, final epoch, train/val accuracy, best val accuracy, data S3 URI, task type, dimension, learning rate, batch size); the `data_s3_uri` field is populated from the `--data_s3_uri` argument passed to `train.py` — if omitted it falls back to `SM_CHANNEL_DATA`, so passing the original S3 data path explicitly ensures the artifact records where the training data came from — SageMaker then bundles all of these into `model.tar.gz` alongside the weights, so the deployment service can point directly at the output artifact without any post-processing merge step; `result.json` is also logged as an MLflow artifact when MLflow tracking is enabled; job records support additional lifecycle operations: `remove_job` deletes a job from local tracking (only allowed when the job is not active), `archive_job` marks a job as hidden from the default list (`archived: true`), `set_comment` attaches a free-text note to a job, `delete_model_artifacts` deletes the `model.tar.gz` from S3 and sets `artifacts_deleted: true` on the record, and `store_result` persists the `result.json` contents against the job entry; `fetch_result_json` downloads `model.tar.gz` from S3, extracts `result.json`, stores it on the job record via `store_result`, and returns it — safe to call multiple times (skips the download if `result` is already present on the record); the job list response includes `archived`, `comment`, `artifacts_deleted`, and `result` fields for each job; `sync_from_aws` pulls up to 100 training jobs from AWS (sorted by creation time, newest first) and merges them into the local store — jobs already tracked locally have their mutable fields updated (`status`, `model_artifacts`, `failure_reason`, `billable_seconds`, `creation_time`, `training_end_time`), while jobs not yet tracked are added with a minimal record; an optional `name_contains` filter narrows the AWS query; the method returns `{"added": [...], "updated": [...]}` with the names of affected jobs and persists the merged state to `data/sagemaker_jobs.json`
- **Deployment** — deploy trained models to SageMaker endpoints; creates a SageMaker model, endpoint config, and endpoint in sequence; tracks endpoint status (`Creating` → `InService` / `Failed`) and exposes the invocation URL once in service; deployments are persisted to `data/deployments.json` and survive server restarts; supports status polling, deletion (deletes the endpoint, its associated endpoint config, and the SageMaker model resource), and listing all active deployments; on deletion the local record is always removed from `data/deployments.json` regardless of whether the AWS API call succeeds — a `try/finally` guarantees cleanup even if the endpoint or config delete fails; if an endpoint is deleted outside the dashboard (e.g. via the AWS console or CLI), the next status poll detects the missing endpoint and marks the local record as `"Deleted"` rather than raising an error; the deployment list endpoint refreshes status from AWS for all entries except those already in `Deleted` state — `InService`, `Failed`, and `Deleting` endpoints are all re-polled on every list request so transient status changes (including deletion progress) are reflected immediately; `Deleted` entries are always filtered out of the list response; the deploy request accepts `model_artifacts`, `image_uri`, and `role_arn` as optional fields — omitting `model_artifacts` lets you import or track an existing endpoint without providing a new artifact; `inference_script` (default `"inference.py"`) specifies which serving entry point to use inside the container; the deployment service points directly at the original `model.tar.gz` — no repacking step is needed because `train.py` already copies `inference.py` and `requirements.txt` into `model_dir` and writes `result.json` at the end of training, so SageMaker bundles them all into the artifact automatically; the PyTorch container loads the entry point via `SAGEMAKER_PROGRAM` — this env var is set on real-time endpoints; set `async_endpoint: true` to create a SageMaker async endpoint — `output_s3_uri` is required in that case and `max_concurrent` (default `10`) controls the `MaxConcurrentInvocationsPerInstance` setting; async endpoints still use the separate sourcedir upload path; set `enable_data_capture: true` on the deploy request to automatically enable data capture at endpoint creation time — `capture_s3_uri` specifies the S3 destination for captured traffic and `capture_sampling_pct` (default `100`) controls the sampling percentage (1–100); this is equivalent to calling `/api/monitoring/data-capture` separately but happens atomically as part of the deploy flow
- **Fairness Analysis** — `FairnessService` (`app/services/fairness_service.py`) scans DICOM files under an S3 prefix and extracts demographic distributions for bias auditing; reads up to 200 files by default (configurable via `max_files`) using `stop_before_pixels=True` for efficiency; extracts `PatientSex`, `PatientAge`, `EthnicGroup`, `PatientRace`, `Modality`, and `BodyPartExamined` tags; age strings in DICOM format (`045Y`, `6M`, `2W`) are normalised to integer years and bucketed into ranges (`0-17`, `18-29`, `30-44`, `45-59`, `60-74`, `75+`); returns per-attribute value counts, age bucket distribution, per-attribute missing rates (% of files where the tag was absent or empty), a capped sample of up to 50 file-level rows, and summary counts (`total_found`, `total_scanned`, `errors`)

- **Endpoint Invocation** — `InferenceService` (`app/services/inference_service.py`) invokes SageMaker real-time endpoints via the `sagemaker-runtime` client; accepts a JSON payload and an optional `output_s3_uri` — if provided, the raw response is saved to S3 at `<prefix>/<endpoint_name>-result.json` and the S3 path is included in the returned result; the `task_type` field is always appended to the response for downstream routing; when the input is not raw bytes, the JSON body preserves the original key — S3 URIs are sent as `{"s3_uri": "..."}` and local paths as `{"file_path": "..."}`

- **Model Monitoring** — `MonitoringService` (`app/services/monitoring_service.py`) manages the full SageMaker Model Monitor lifecycle for deployed endpoints; schedules and records are persisted to `data/monitoring.json` and survive server restarts; supports four operations:
  - **Data capture** — enables or updates data capture on an existing endpoint by creating a new endpoint config with `DataCaptureConfig` (captures both `Input` and `Output` at a configurable sampling percentage) and updating the endpoint in-place; the original KMS key is preserved if present
  - **Data quality baseline** — submits a SageMaker Processing Job using the prebuilt Model Monitor container to compute baseline statistics and constraints from a reference dataset in S3; the baseline job record is tracked locally with `type: "baseline"`
  - **Data quality schedule** — creates a `DataQuality` monitoring schedule that compares captured endpoint traffic against the baseline constraints and statistics on a cron or rate expression
  - **Model quality schedule** — creates a `ModelQuality` monitoring schedule that compares endpoint predictions against ground-truth labels uploaded to S3; supports `inference_attribute`, optional `probability_attribute`, and `ground_truth_attribute` configuration
  - **Status refresh** — `refresh_schedule` pulls the current schedule status and the most recent execution summary (status, time, Processing Job ARN, failure reason) from AWS and updates the local record
  - **Deletion** — `delete_schedule` stops and deletes a monitoring schedule from AWS and removes it from the local store
  - **Capture summary** — `get_capture_summary` reads the latest `.jsonl` capture files from S3 under `<capture_s3_uri>/<endpoint_name>/AllTraffic/`, decodes base64-encoded endpoint outputs, and returns aggregated prediction distribution stats; handles both batched-array format (`predictions`, `predicted_class`, `confidence`) and named-probability format (`probabilities`); returns `class_distribution` (per-label counts), a 10-bucket `confidence_histogram`, `avg_confidence`, and up to 50 `recent_records` with per-prediction probabilities; scans up to `max_files` (default `20`) most-recently-modified files; returns a zero-record summary with a descriptive message if no capture files are found or none can be parsed
  - **Listing** — `list_schedules` returns all tracked schedules sorted by creation time (newest first); when a SageMaker client is provided, active schedules are refreshed from AWS before returning
  - The prebuilt Model Monitor container URI is resolved automatically per region using a built-in account map (covers `us-east-1`, `us-east-2`, `us-west-1`, `us-west-2`, `eu-west-1`, `eu-central-1`, `ap-northeast-1`, `ap-southeast-1`, `ap-southeast-2`, `ap-south-1`); defaults to the `us-east-1` account for unlisted regions

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

## SageMaker Endpoint Evaluation

`src/evaluation/evaluate.py` is a SageMaker Processing Job script that evaluates a deployed classification endpoint against a labeled test set in S3.

### How it works

1. Discovers class folders under the provided S3 data URI (each subfolder name is treated as a class label)
2. Invokes the endpoint for every file and collects predictions and probability scores
3. Computes accuracy, per-class precision/recall/F1, and macro AUC (OvR)
4. Writes four output artifacts to `/opt/ml/processing/output` and uploads them to the specified S3 output URI

### Output artifacts

| File | Description |
|---|---|
| `metrics.json` | Accuracy, macro AUC, per-class P/R/F1, and support counts |
| `predictions.csv` | Per-file ground truth vs predicted class + per-class probabilities |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_auc.png` | ROC-AUC curves (one per class, one-vs-rest) |

### Expected S3 input structure

```
s3://bucket/prefix/
    class_a/
        image1.dcm
        image2.dcm
    class_b/
        image3.dcm
```

### CLI arguments

| Argument | Required | Description |
|---|---|---|
| `--endpoint-name` | yes | Name of the `InService` SageMaker endpoint |
| `--data-s3-uri` | yes | S3 URI of the test dataset root (class subfolders) |
| `--output-s3-uri` | yes | S3 URI prefix where result artifacts are uploaded |
| `--region` | no | AWS region (default: `AWS_DEFAULT_REGION` env var or `us-east-1`) |

### Endpoint response formats

The script handles two response formats from the endpoint:

- **Format A** (batched arrays): `{"predictions": [[p0, p1, ...]], "predicted_class": [idx], "confidence": [c]}`
- **Format B** (named probabilities): `{"probabilities": {"class_a": 0.3, ...}, "predicted_class": "class_a"}`

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
│       ├── fairness_service.py             # DICOM demographic scanning and fairness analysis
│       └── monitoring_service.py           # SageMaker Model Monitor — data capture, baselines, and schedules
├── static/             # CSS and JS
├── templates/          # Jinja2 HTML templates
├── data/
│   ├── raw/                    # Raw input data (gitignored)
│   ├── processed/              # Preprocessed datasets (gitignored)
│   ├── models/                 # Imported model artifacts (gitignored)
│   ├── sagemaker_jobs.json     # Persisted SageMaker job records (auto-created)
│   ├── deployments.json        # Persisted endpoint deployment records (auto-created)
│   ├── monitoring.json         # Persisted Model Monitor schedule records (auto-created)
│   └── training_reports/       # Per-job JSON reports written on completion (auto-created)
├── logs/               # TensorBoard logs (gitignored)
├── src/
│   ├── classification/ # SageMaker BYOC training scripts (classification)
│   ├── segmentation/   # SageMaker BYOC training scripts (segmentation)
│   ├── evaluation/
│   │   └── evaluate.py             # SageMaker Processing Job script for endpoint evaluation
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
# Use defaults (endpoint: test-endp, region: us-east-1)
python test_endpoint.py

# Override endpoint and/or image
python test_endpoint.py --endpoint my-endpoint
python test_endpoint.py --endpoint my-endpoint --image s3://bucket/path/image.dcm --region eu-west-1

# Also test local file inference (raw DICOM bytes)
python test_endpoint.py --endpoint my-endpoint --local /path/to/image.dcm
```

| Argument | Default | Description |
|---|---|---|
| `--endpoint` | `test-endp` | SageMaker endpoint name |
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

One of `--image` or `--s3-uri` is required. When using `--image`, the file is base64-encoded and sent as `{"file_path": "<b64>", "filename": "..."}`. When using `--s3-uri`, the payload is `{"s3_uri": "s3://..."}` — the `s3_uri` key is preserved so the endpoint can distinguish it from a local path. Requires valid AWS credentials in the environment.

## API Endpoints

| Prefix | Description |
|---|---|
| `/api/aws` | AWS credentials and S3 operations |
| `/api/training` | Local and SageMaker training jobs |
| `/api/deployment` | Model deployment to SageMaker |
| `/api/inference` | Real-time endpoint invocation and listing |
| `/api/fairness` | DICOM demographic scanning and fairness analysis |
| `/api/evaluation` | SageMaker Processing Job-based endpoint evaluation |
| `/api/monitoring` | SageMaker Model Monitor — data capture, baselines, and schedules |

### `/api/inference` routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/inference/invoke` | Invoke a SageMaker real-time endpoint; accepts `endpoint_name`, `task_type` (`classification` \| `segmentation`), `input_type` (`s3_uri` \| `file_path`), `input_value`, and optional `output_s3_uri` |
| `POST` | `/api/inference/invoke-local` | Invoke a SageMaker real-time endpoint with raw file bytes uploaded directly from the browser; accepts `endpoint_name` and `task_type` as query params, request body is the raw binary file content; `output_s3_uri` is not supported |
| `GET` | `/api/inference/endpoints` | List all `InService` SageMaker endpoints |

### `/api/monitoring` routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/monitoring/data-capture` | Enable or update data capture on an existing SageMaker endpoint; accepts `endpoint_name`, `capture_s3_uri`, and `sampling_percentage` (1–100, default `100`) |
| `POST` | `/api/monitoring/baseline/data-quality` | Submit a SageMaker Processing Job to compute data quality baseline statistics and constraints from a reference dataset in S3; accepts `baseline_data_s3_uri`, `output_s3_uri`, `role_arn`, `instance_type`, and optional `job_name` |
| `POST` | `/api/monitoring/schedules/data-quality` | Create a data quality monitoring schedule that compares captured endpoint traffic against baseline constraints on a cron or rate expression |
| `POST` | `/api/monitoring/schedules/model-quality` | Create a model quality monitoring schedule that compares endpoint predictions against ground-truth labels in S3; supports `BinaryClassification`, `MulticlassClassification`, and `Regression` problem types |
| `GET` | `/api/monitoring/capture-summary` | Read and aggregate the latest captured traffic for an endpoint; accepts `endpoint_name`, `capture_s3_uri`, and optional `max_files` (default `20`); returns `class_distribution`, `confidence_histogram`, `avg_confidence`, and up to 50 `recent_records` |
| `GET` | `/api/monitoring/schedules` | List all tracked monitoring schedules (newest first); pass `?local_only=true` to skip the AWS refresh |
| `GET` | `/api/monitoring/schedules/{schedule_name}` | Get the current status and most recent execution summary for a schedule |
| `DELETE` | `/api/monitoring/schedules/{schedule_name}` | Stop and delete a monitoring schedule from AWS and remove it from the local store |
