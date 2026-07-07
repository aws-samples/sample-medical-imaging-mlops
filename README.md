# MedML Training Dashboard

A FastAPI web dashboard for managing medical image classification workflows on AWS. Provides a UI for data ingestion, model training (local and SageMaker), and deployment.

## Features

- **AWS Integration** — connect via environment credentials or explicit keys, browse S3 buckets; generate presigned URLs for SageMaker MLflow tracking servers via `create_presigned_mlflow_tracking_server_url` (accepts both full ARNs and plain server names — the server name is extracted from the ARN automatically; presigned URLs expire after 5 minutes so a fresh one is generated on each request; `mlflow-app/<name>` ARNs are Studio-managed apps and are handled separately — the service calls `list_domains` and `list_user_profiles` to locate the hosting domain, then calls `create_presigned_domain_url` to generate a Studio presigned URL; if no domain or user profile is found, a descriptive `ValueError` is raised directing the user to open Studio manually; standard `mlflow-tracking-server/<name>` ARNs use `create_presigned_mlflow_tracking_server_url` directly; if the direct name-based lookup returns `ResourceNotFound` or `ValidationException`, the service falls back to listing all tracking servers and matching by name or ARN suffix — raises `ValueError` with a descriptive message if no match is found)
- **Local Training** — train ResNet50, VGG16, EfficientNet, DenseNet121, ViT, or custom models with MLflow and TensorBoard tracking
- **Model Import** — import a pre-trained model artifact from S3 via `POST /api/training/models/import`; accepts `s3_bucket`, `s3_key`, `model_name`, `model_format`, and `num_classes`; downloads the artifact, validates the file format, registers it locally, and returns an `ImportedModel` response with a generated `model_id`, `local_path`, `s3_source`, and `imported_at` timestamp
- **SageMaker Training** — launch BYOC training jobs on SageMaker (e.g. `ml.g5.xlarge`) with optional MLflow integration; supports fine-tuning via `checkpoint_s3_uri` (S3 URI of a `model.tar.gz` from a completed job — the script downloads the archive, extracts `best_model.pth` if present or falls back to `model.pth`, and restores both model weights and optimizer state; `start_epoch` is set to the checkpoint's epoch + 1 so the training loop continues from where the previous run left off); if the checkpoint cannot be loaded for any reason, training falls back to starting from scratch with a warning rather than failing the job; `fine_tuned_from` records the source S3 URI in `result.json` for lineage; both fields are optional and ignored when starting a fresh training run; both the classification script (`src/classification/train.py`) and the segmentation script (`src/segmentation/train.py`) enable MLflow automatically when `MLFLOW_TRACKING_URI` is set — they log hyperparameters, per-epoch metrics, the best model checkpoint, and the final model as artifacts; the best checkpoint (`best_model.pth`) saved by the classification script includes full training state: `model` weights, `optimizer` state, `epoch`, `val_acc`, `train_acc`, `train_loss`, `val_loss`, `model_name`, and `num_classes` — making it self-contained for resuming training or auditing; a separate weights-only checkpoint (`model_deploy.pth`) is also written at the end of training — it contains only the model state dict (no optimizer or scheduler state), making it roughly 50% smaller than `model.pth` and suitable for deployment; the segmentation script always uses 1 output channel (binary segmentation) and names the MLflow experiment `segmentation-<model_name>`; at the end of a segmentation training run, `inference.py`, `model_def.py`, and `requirements.txt` are automatically copied from `src/inference/segmentation/` (falling back to the training script directory) into `model_dir`, so SageMaker bundles them into `model.tar.gz` alongside the weights — the deployment service can point directly at the output artifact without any post-processing merge step; a `result.json` file is also written to `model_dir` with training provenance: `model_name`, `task_type`, `dimension`, `final_epoch`, `best_val_dice` (rounded to 4 decimal places), `learning_rate`, `batch_size`, `data_s3_uri` (from `SM_CHANNEL_DATA` or the `--data_s3_uri` argument), and `fine_tuned_from` (source checkpoint S3 URI, or `null` for fresh runs); `result.json` is also logged as an MLflow artifact when MLflow tracking is enabled; `num_classes` for classification is derived dynamically from the training data folder structure rather than passed as a hyperparameter; job list automatically refreshes status from AWS for any jobs currently in `InProgress` or `Stopping` state (when AWS credentials are configured), so active jobs always reflect their latest status without a separate poll; pass `?local_only=true` to skip the AWS refresh and return only the locally-cached state — useful for fast polling or when AWS credentials are unavailable; completed, failed, and other terminal-state jobs are returned directly from the local store — use the dedicated `/api/training/sagemaker/jobs/{job_name}/status` endpoint or `sync_from_aws` to force a refresh for those; jobs are persisted to `data/sagemaker_jobs.json` and survive server restarts; if `data/sagemaker_jobs.json` is missing on startup, the service automatically bootstraps the jobs store from any JSON files found in `data/training_reports/` — each report is parsed and a minimal job record is reconstructed, then the rebuilt store is saved to `data/sagemaker_jobs.json` so subsequent restarts load normally; completed jobs include billable seconds, an estimated cost in USD, `creation_time`, and `training_end_time`; each job in the list exposes the `mlflow_tracking_uri` it was submitted with (if any); a JSON training report is written to `data/training_reports/<job_name>.json` on completion; at the end of a classification training run, `inference.py` and `requirements.txt` are automatically copied from the training script directory into `model_dir`, and a `result.json` file is written there with training provenance (model name, num classes, final epoch, train/val accuracy, best val accuracy, data S3 URI, task type, dimension, learning rate, batch size); the `data_s3_uri` field is populated from the `--data_s3_uri` argument passed to `train.py` — if omitted it falls back to `SM_CHANNEL_DATA`, so passing the original S3 data path explicitly ensures the artifact records where the training data came from — SageMaker then bundles all of these into `model.tar.gz` alongside the weights, so the deployment service can point directly at the output artifact without any post-processing merge step; `result.json` is also logged as an MLflow artifact when MLflow tracking is enabled; job records support additional lifecycle operations: `remove_job` deletes a job from local tracking (only allowed when the job is not active), `archive_job` marks a job as hidden from the default list (`archived: true`), `set_comment` attaches a free-text note to a job, `delete_model_artifacts` deletes the `model.tar.gz` from S3 and sets `artifacts_deleted: true` on the record, and `store_result` persists the `result.json` contents against the job entry; `fetch_result_json` downloads `model.tar.gz` from S3 and attempts to extract `result.json`; if `result.json` is absent (e.g. jobs trained before it was added to `train.py`), it falls back to reconstructing a result dict from the stored job config (`model_name`, `epochs`, `data_s3_uri`, `task_type`, `dimension`, `learning_rate`, `batch_size`) with accuracy fields set to `null` and a `reconstructed: true` flag so the UI can surface a note; the result is stored on the job record via `store_result` and returned — safe to call multiple times (skips the download if `result` is already present on the record); the job list response includes `archived`, `comment`, `artifacts_deleted`, and `result` fields for each job; `sync_from_aws` pulls up to 100 training jobs from AWS (sorted by creation time, newest first) and merges them into the local store — jobs already tracked locally have their mutable fields updated (`status`, `model_artifacts`, `failure_reason`, `billable_seconds`, `creation_time`, `training_end_time`), while jobs not yet tracked are added with a minimal record; an optional `name_contains` filter narrows the AWS query; the method returns `{"added": [...], "updated": [...]}` with the names of affected jobs and persists the merged state to `data/sagemaker_jobs.json`
- **Deployment** — deploy trained models to SageMaker endpoints; creates a SageMaker model, endpoint config, and endpoint in sequence; tracks endpoint status (`Creating` → `InService` / `Failed`) and exposes the invocation URL once in service; deployments are persisted to `data/deployments.json` and survive server restarts; supports status polling, deletion (deletes the endpoint, its associated endpoint config, and the SageMaker model resource), and listing all active deployments; on deletion the local record is always removed from `data/deployments.json` regardless of whether the AWS API call succeeds — a `try/finally` guarantees cleanup even if the endpoint or config delete fails; if an endpoint is deleted outside the dashboard (e.g. via the AWS console or CLI), the next status poll detects the missing endpoint and marks the local record as `"Deleted"` rather than raising an error; the deployment list endpoint refreshes status from AWS for all entries except those already in `Deleted` state — `InService`, `Failed`, and `Deleting` endpoints are all re-polled on every list request so transient status changes (including deletion progress) are reflected immediately; `Deleted` entries are always filtered out of the list response; the deploy request accepts `model_artifacts`, `image_uri`, and `role_arn` as optional fields — omitting `model_artifacts` lets you import or track an existing endpoint without providing a new artifact; `inference_script` (default `"inference.py"`) specifies which serving entry point to use inside the container; the deployment service points directly at the original `model.tar.gz` — no repacking step is needed because both `train.py` scripts copy inference artifacts (`inference.py`, `model_def.py`, `requirements.txt`) into `model_dir` and write `result.json` at the end of training, so SageMaker bundles them all into the artifact automatically; the PyTorch container loads the entry point via `SAGEMAKER_PROGRAM` — this env var is set on real-time endpoints; set `async_endpoint: true` to create a SageMaker async endpoint — `output_s3_uri` is required in that case and `max_concurrent` (default `10`) controls the `MaxConcurrentInvocationsPerInstance` setting; async endpoints still use the separate sourcedir upload path; set `enable_data_capture: true` on the deploy request to automatically enable data capture at endpoint creation time — `capture_s3_uri` specifies the S3 destination for captured traffic and `capture_sampling_pct` (default `100`) controls the sampling percentage (1–100); this is equivalent to calling `/api/monitoring/data-capture` separately but happens atomically as part of the deploy flow
- **Paper / Report Generation** — `PaperReportService` (`app/services/paper_report_service.py`) generates structured academic paper sections from completed training and evaluation jobs; accepts a list of SageMaker training job names and evaluation job names along with a study title and a set of sections to include (`abstract`, `methods`, `results`, `discussion`); optionally incorporates fairness analysis results when `include_fairness: true`; the `/api/report/jobs` endpoint returns only completed, non-archived training jobs and completed evaluation jobs so the UI can present a clean selection list

- **Fairness Analysis** — `FairnessService` (`app/services/fairness_service.py`) scans DICOM files under an S3 prefix and extracts demographic distributions for bias auditing; reads up to 200 files by default (configurable via `max_files`) using `stop_before_pixels=True` for efficiency; extracts `PatientSex`, `PatientAge`, `EthnicGroup`, `PatientRace`, `Modality`, and `BodyPartExamined` tags; age strings in DICOM format (`045Y`, `6M`, `2W`) are normalised to integer years and bucketed into ranges (`0-17`, `18-29`, `30-44`, `45-59`, `60-74`, `75+`); returns per-attribute value counts, age bucket distribution, per-attribute missing rates (% of files where the tag was absent or empty), a capped sample of up to 50 file-level rows, and summary counts (`total_found`, `total_scanned`, `errors`)

- **Endpoint Invocation** — `InferenceService` (`app/services/inference_service.py`) invokes SageMaker real-time endpoints via the `sagemaker-runtime` client; accepts a JSON payload and an optional `output_s3_uri` — if provided, the raw response is saved to S3 at `<prefix>/<endpoint_name>-result.json` and the S3 path is included in the returned result; the `task_type` field is always appended to the response for downstream routing; when the input is not raw bytes, the JSON body preserves the original key — S3 URIs are sent as `{"s3_uri": "..."}` and local paths as `{"file_path": "..."}` ; the `sagemaker-runtime` client is configured with a **300 s read timeout** and a 10 s connect timeout (1 retry max) to accommodate slow 3D segmentation inference — the timeout is applied globally via a module-level `botocore.config.Config` instance (`_INFERENCE_CONFIG`) shared across all invocations

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

Dynamically imports `model_def.py` from the model artifact directory (`model_dir/model_def.py`). If the file is not present, falls back to a default `SegResNet` (`spatial_dims=3, in_channels=1, out_channels=1, init_filters=16`). Weights are loaded from `best_model.pth` or `model.pth`. The `MODEL_NAME` environment variable selects the architecture when using `model_def.py`.

> **Note:** `model_def.py` must be bundled inside `model.tar.gz` for a custom architecture to be used. The segmentation `train.py` copies it into `model_dir` automatically at the end of training, so artifacts produced by this dashboard are self-contained.

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

## Model Monitoring Workflow

SageMaker Model Monitor tracks data quality and prediction drift on deployed endpoints. The dashboard covers the full lifecycle.

### Step 1 — Enable data capture

Data capture records every inference request and response to S3 as `.jsonl` files. You have two options:

**At deploy time (recommended)** — check "Enable Data Capture" in the Deployment form before clicking Deploy. Fill in the Capture S3 URI. Capture is active from the first inference call.

**After deployment** — go to Monitoring → Data Capture, select the endpoint, enter the Capture S3 URI, set the sampling percentage (default 100%), and click Enable. SageMaker updates the endpoint in-place (status shows `Updating` for a few minutes).

The S3 path SageMaker writes to follows this structure:
```
s3://your-bucket/capture-prefix/<endpoint-name>/AllTraffic/<year>/<month>/<day>/<hour>/
```
Enter the root prefix (e.g. `s3://your-bucket/capture-prefix/`) — not the full dated path.

### Step 2 — View captured traffic (Capture Viewer)

After making inference calls, go to Monitoring → Capture Viewer:

1. Select the endpoint from the dropdown — the Capture S3 URI auto-fills if capture was enabled via the dashboard
2. Set Max Files (default 20)
3. Click Analyse

The viewer shows:
- Summary cards: total records, files scanned, average confidence, classes seen
- Prediction class distribution bar chart
- Confidence score histogram
- Recent captures table (up to 50 rows)

### Step 3 — Run a data quality baseline (optional)

A baseline computes statistics and constraints from a reference dataset (typically your training data). These are used by the monitoring schedule to detect drift.

Go to Monitoring → Data Quality Baseline:
- **Baseline Data S3 URI** — S3 path to your reference dataset
- **Output S3 URI** — where constraints and statistics JSON files are written
- **IAM Role ARN** — SageMaker execution role
- Click Run Baseline Job

Once complete, the baseline card shows the exact S3 paths for `constraints.json` and `statistics.json`. Click "Use in Schedule" to auto-fill the Data Quality Schedule form.

### Step 4 — Create a monitoring schedule (optional)

Go to Monitoring → Data Quality Schedule:
- **Schedule Name** — unique name for the schedule
- **Endpoint Name** — the endpoint to monitor
- **Constraints / Statistics S3 URI** — from the baseline output (auto-filled if you used "Use in Schedule")
- **Output S3 URI** — where violation reports are written
- **Schedule Expression** — cron or rate expression (e.g. `cron(0 * ? * * *)` for hourly)
- Click Create Schedule

SageMaker runs the monitor on the schedule and writes violation reports to the output S3 URI. The schedule card shows the last execution status and time.

### Inference container requirements

The inference container must support data capture. The BYOC inference images (`sm-inference-byoc-classification` and `sm-inference-byoc-segmentation`) are pre-configured for this. The generic AWS PyTorch inference image (`pytorch-inference:2.6.0-gpu-py312-cu124`) also supports capture but does not include `inference.py` — use the BYOC images for all deployments from this dashboard.

---

## Building Docker Images

The dashboard uses two BYOC (Bring Your Own Container) Docker images — one for training and one for inference. Both are built from `Dockerfile.sagemaker` and pushed to Amazon ECR.

### Prerequisites on the build machine

- Docker installed and running
- AWS CLI configured with credentials that have ECR push and pull permissions
- Login to AWS Deep Learning Containers ECR (required before the first build — the base image is hosted in AWS's account):

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    763104351884.dkr.ecr.us-east-1.amazonaws.com
```

> This step is required on every new machine. Without it the build fails with `403 Forbidden` when pulling the base image.

### Build and push

```bash
# Classification training image
./build_and_push.sh <YOUR_AWS_ACCOUNT_ID> us-east-1 classification

# Segmentation training image
./build_and_push.sh <YOUR_AWS_ACCOUNT_ID> us-east-1 segmentation

# Inference image (classification)
./build_and_push_inference.sh <YOUR_AWS_ACCOUNT_ID> us-east-1
```

### Base image

The `Dockerfile.sagemaker` uses:

```
763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker
```

If this tag is unavailable in your region, find the closest available tag:

```bash
aws ecr describe-images \
  --registry-id 763104351884 \
  --repository-name pytorch-training \
  --region us-east-1 \
  --query 'imageDetails[*].imageTags' \
  --output text | tr '\t' '\n' | grep "py312" | sort
```

Then update the `FROM` line in `Dockerfile.sagemaker` accordingly. A known working alternative is:

```
763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker
```

### Transferring to another machine

The cleanest approach is git — push your branch and pull on the other machine:

```bash
# This machine
git add -A && git commit -m "your message" && git push origin <branch>

# Other machine
git clone <repo-url> && git checkout <branch>
```

If git is not available, the minimum files needed for the Docker build are:

```
Dockerfile.sagemaker
build_and_push.sh
src/classification/   (for classification image)
src/segmentation/     (for segmentation image)
```



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

# Configure account-specific values (ECR URIs, IAM role, etc.)
cp .env.local.example .env.local
# Edit .env.local with your AWS account ID, role ARN, and S3 paths

# Start the app
python main.py
```

App runs at `http://localhost:8000`

> On first launch with empty data stores, click **Sync from AWS** on the Training or Deployment tabs to pull existing jobs and endpoints from your AWS account.

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

### `.env` — General application config

| Variable | Description | Default |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key | — |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | — |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `MLFLOW_TRACKING_URI` | MLflow server URI | `http://localhost:5000` |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment name | `medical_imaging_experiments` |
| `TENSORBOARD_LOG_DIR` | TensorBoard log directory | `./logs/tensorboard` |

### `.env.local` — Account-specific config (gitignored)

| Variable | Description | Example |
|---|---|---|
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID | `123456789012` |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role ARN | `arn:aws:iam::123456789012:role/SageMakerRole` |
| `TRAINING_IMAGE_CLASSIFICATION` | ECR URI for classification training image | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest` |
| `TRAINING_IMAGE_SEGMENTATION` | ECR URI for segmentation training image | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest` |
| `INFERENCE_IMAGE_CLASSIFICATION` | ECR URI for classification inference image | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest` |
| `INFERENCE_IMAGE_SEGMENTATION` | ECR URI for segmentation inference image | `123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest` |
| `DEFAULT_DATA_S3_URI` | Default training data S3 URI (pre-populates form) | `s3://your-bucket/training-data/` |
| `TEST_IMAGE_S3_URI` | Test image for integration tests | `s3://your-bucket/test/image.dcm` |
| `TEST_SEGMENTATION_S3_URI` | Test volume for segmentation tests | `s3://your-bucket/test/volume.nii.gz` |

> `.env.local` is loaded after `.env` with override priority. Copy `.env.local.example` and fill in your values. This file is gitignored and never committed.

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
│       ├── monitoring_service.py           # SageMaker Model Monitor — data capture, baselines, and schedules
│       └── paper_report_service.py         # Academic paper/report section generation from job data
├── static/             # CSS and JS
├── templates/          # Jinja2 HTML templates
├── data/
│   ├── raw/                    # Raw input data (gitignored)
│   ├── processed/              # Preprocessed datasets (gitignored)
│   ├── models/                 # Imported model artifacts (gitignored)
│   ├── custom_models/          # Extracted custom training code uploads (auto-created, gitignored)
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
├── deploy_segmentation.py  # Standalone script: deploy segmentation model.tar.gz to SageMaker + optional inference test + cleanup
├── test_endpoint_segmentation.py  # Standalone script: test a deployed segmentation endpoint (S3 URI / local / base64)
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

#### Segmentation deployment script — `deploy_segmentation.py`

A standalone script for deploying a segmentation `model.tar.gz` to a SageMaker real-time endpoint and optionally running an inference test against it. Mirrors the notebook deployment pattern adapted for NIfTI/segmentation workflows.

```bash
# Deploy a model artifact
python deploy_segmentation.py \
    --model-artifacts s3://bucket/path/model.tar.gz

# Deploy with a custom endpoint name
python deploy_segmentation.py \
    --model-artifacts s3://bucket/path/model.tar.gz \
    --endpoint-name my-aorta-seg-endpoint

# Skip deploy and test an already-deployed endpoint via S3 URI
python deploy_segmentation.py \
    --endpoint-name aorta-segmentation \
    --skip-deploy \
    --s3-uri s3://bucket/val/subject_001/img.nii.gz

# Skip deploy and test with a local NIfTI file, saving the returned mask
python deploy_segmentation.py \
    --endpoint-name aorta-segmentation \
    --skip-deploy \
    --local /path/to/volume.nii.gz \
    --save-mask /tmp/mask.nii.gz

# Delete the endpoint, endpoint config, and model
python deploy_segmentation.py \
    --endpoint-name aorta-segmentation \
    --cleanup
```

| Argument | Default | Description |
|---|---|---|
| `--endpoint-name` | `aorta-segmentation` | SageMaker endpoint name |
| `--region` | `us-east-1` | AWS region |
| `--model-artifacts` | _(required unless `--skip-deploy` or `--cleanup`)_ | S3 URI of the `model.tar.gz` to deploy |
| `--role-arn` | Default execution role ARN | IAM role ARN for SageMaker |
| `--instance-type` | `ml.g5.xlarge` | SageMaker instance type for the endpoint |
| `--skip-deploy` | `false` | Skip deployment and go straight to inference testing |
| `--cleanup` | `false` | Delete the endpoint, endpoint config, and model, then exit |
| `--s3-uri` | _(unset)_ | S3 URI of a NIfTI volume to test with |
| `--local` | _(unset)_ | Local `.nii.gz` file to test with (sent as raw bytes) |
| `--b64` | _(unset)_ | Local `.nii.gz` file to test with (sent as base64-encoded JSON) |
| `--save-mask` | _(unset)_ | Output path for the returned binary mask (`.nii.gz`) |
| `--timeout` | `300` | Invoke read timeout in seconds (3D segmentation can be slow) |

The script checks whether the endpoint already exists before deploying — if it does, it prints a message and skips the deploy step. After deployment (or with `--skip-deploy`), it verifies the endpoint is `InService` before invoking. The inference output includes latency, mask shape, positive voxel count, total voxels, and a visual volume-fraction bar. An empty-mask warning is shown when no positive voxels are returned. The `--cleanup` flag deletes the endpoint, its endpoint config, and the associated SageMaker model in sequence. Requires valid AWS credentials in the environment.

#### Segmentation endpoint quick test — `test_endpoint_segmentation.py`

A standalone script at the project root for testing a deployed SageMaker **segmentation** endpoint. No pytest required — run it directly. Supports three input modes and optionally saves the returned binary mask to disk.

```bash
# S3 URI input (default — recommended)
python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
    --s3-uri s3://your-bucket/dataset/val/subject_001/img.nii.gz

# Local NIfTI file — raw bytes (application/octet-stream)
python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
    --local /path/to/volume.nii.gz

# Local NIfTI file — base64-encoded JSON payload
python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
    --b64 /path/to/volume.nii.gz

# Save the returned binary mask to disk
python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
    --s3-uri s3://bucket/img.nii.gz --save-mask /tmp/mask.nii.gz
```

| Argument | Default | Description |
|---|---|---|
| `--endpoint` | `my-seg-endpoint` | SageMaker endpoint name |
| `--region` | `us-east-1` | AWS region |
| `--s3-uri` | Public aorta NIfTI sample | S3 URI of a NIfTI volume; used when neither `--local` nor `--b64` is provided |
| `--local` | _(unset)_ | Path to a local `.nii.gz` file; sent as raw bytes with `Content-Type: application/octet-stream` |
| `--b64` | _(unset)_ | Path to a local `.nii.gz` file; base64-encoded and sent as a JSON payload |
| `--save-mask` | _(unset)_ | Output path for the returned binary mask (`.nii.gz`); requires `nibabel` if installed, otherwise writes raw bytes |

The script first checks that the endpoint is `InService` before invoking. After inference it prints a summary including latency, mask shape, positive voxel count, total voxels, and a visual volume-fraction bar. An empty-mask warning is shown when no positive voxels are returned, with a hint to check the model and HU range. Requires valid AWS credentials in the environment.

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
| `/api/config` | Runtime configuration (account ID, role ARN, ECR URIs) |
| `/api/aws` | AWS credentials and S3 operations |
| `/api/training` | Local and SageMaker training jobs |
| `/api/deployment` | Model deployment to SageMaker |
| `/api/inference` | Real-time endpoint invocation and listing |
| `/api/fairness` | DICOM demographic scanning and fairness analysis |
| `/api/evaluation` | SageMaker Processing Job-based endpoint evaluation |
| `/api/monitoring` | SageMaker Model Monitor — data capture, baselines, and schedules |
| `/api/report` | Paper/report section generation from training and evaluation jobs |

### `/api/training` routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/training/models/upload-code` | Upload a `.zip` file of custom model code; extracts to `data/custom_models/<model_name>/` and returns `{"model_name", "source_dir", "files"}`; query params: `model_name` (string); form field: `file` (`.zip` only); rejects absolute paths and `..` traversal in the archive |
| `POST` | `/api/training/models/import` | Import a pre-trained model artifact from S3 |
| `POST` | `/api/training/sagemaker/sync` | Pull up to 100 training jobs from AWS and merge into local store; optional `?name_contains=` filter |

### `/api/deployment` routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/deployment/sync` | Pull endpoints from AWS and merge into local store; optional `?name_contains=` filter |

### `/api/evaluation` routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/evaluation/sync` | Pull evaluation processing jobs from AWS and merge into local store; optional `?name_contains=` filter |

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

### `/api/report` routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/report/jobs` | List completed training and evaluation jobs available for report generation; returns only non-archived, `Completed` training jobs and `Completed` evaluation jobs |
| `POST` | `/api/report/generate` | Generate paper sections from selected jobs; accepts `title`, `training_job_names`, `evaluation_job_names`, `sections` (`abstract`, `methods`, `results`, `discussion`), and `include_fairness` (bool) |
