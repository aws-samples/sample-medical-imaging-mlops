# API Reference

Base URL: `http://localhost:8000`

All endpoints return JSON. Errors follow the format:
```json
{"detail": "Error message describing what went wrong"}
```

---

## GET /api/config

Runtime configuration (non-secret values from `.env.local`).

**Response:**
```json
{
  "aws_account_id": "123456789012",
  "aws_region": "us-east-1",
  "sagemaker_role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
  "training_image_classification": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest",
  "training_image_segmentation": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-segmentation:latest",
  "inference_image_classification": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest",
  "inference_image_segmentation": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-segmentation:latest",
  "default_data_s3_uri": "s3://your-bucket/training-data/"
}
```

---

## GET /health

Health check endpoint.

**Response:**
```json
{"status": "healthy"}
```

---

## /api/aws — AWS Credentials & S3 Browsing

### GET /api/aws/status

Check whether AWS credentials are configured.

**Response:**
```json
{"configured": true, "source": "environment"}
```

`source` is `"environment"` if auto-configured from env vars/profiles, or `"manual"` if set via the `/configure` endpoint.

---

### POST /api/aws/configure

Manually set AWS credentials.

**Request body:**
```json
{
  "access_key_id": "AKIA...",
  "secret_access_key": "...",
  "region": "us-east-1"
}
```

**Response:**
```json
{"status": "success", "message": "AWS credentials configured successfully"}
```

**Errors:** 400 (invalid credentials), 500 (internal error)

---

### GET /api/aws/buckets

List all S3 buckets in the account.

**Response:**
```json
[
  {"name": "my-bucket", "region": "us-east-1", "creation_date": "2024-01-15T10:30:00Z"}
]
```

**Errors:** 400 (credentials not configured), 500 (AWS API error)

---

### GET /api/aws/buckets/{bucket_name}/objects

List objects in an S3 bucket.

**Query parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | string | `""` | S3 key prefix to filter objects |

**Response:**
```json
{
  "bucket": "my-bucket",
  "objects": [
    {"key": "data/train/", "size": 0, "last_modified": "2024-01-15T10:30:00Z"},
    {"key": "data/train/image1.dcm", "size": 524288, "last_modified": "2024-01-15T10:31:00Z"}
  ]
}
```

---

## /api/training — Models, Training Jobs, SageMaker

### GET /api/training/models

List available model architectures from all task registries (`src/classification/model_def.py` and `src/segmentation/model_def.py`).

**Response:**
```json
{
  "models": [
    {
      "name": "densenet121",
      "description": "DenseNet-121 pretrained on ImageNet",
      "parameters": "7.0M",
      "input_size": "224x224",
      "pretrained": true,
      "task": "classification",
      "dimension": "2D",
      "source_dir": "src/classification"
    }
  ]
}
```

---

### POST /api/training/models/upload-code

Upload a `.zip` file containing custom model code.

**Query parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | yes | Name for the custom model |

**Request:** `multipart/form-data` with field `file` (`.zip` only)

**Response:**
```json
{
  "model_name": "my_custom_model",
  "source_dir": "data/custom_models/my_custom_model",
  "files": ["model_def.py", "train.py", "requirements.txt"]
}
```

**Notes:** Rejects absolute paths and `..` traversal in the archive.

---

### POST /api/training/models/import

Import a pre-trained model artifact from S3.

**Request body:**
```json
{
  "model_name": "spine-classifier-v2",
  "s3_bucket": "my-bucket",
  "s3_key": "models/spine_model.pth",
  "model_format": "pytorch",
  "framework_version": "2.1.1",
  "num_classes": 8
}
```

**Response:**
```json
{
  "model_id": "uuid-string",
  "model_name": "spine-classifier-v2",
  "s3_source": "s3://my-bucket/models/spine_model.pth",
  "local_path": "data/models/uuid-string/spine_model.pth",
  "model_format": "pytorch",
  "num_classes": 8,
  "imported_at": "2024-06-15T10:30:00",
  "status": "available"
}
```

---

### GET /api/training/models/imported

List all imported models.

**Response:**
```text
{"models": [...]}
```

---

### GET /api/training/models/imported/{model_id}

Get details of a specific imported model.

**Response:** Single model object or 404.

---

### POST /api/training/start

Start a local training job.

**Query parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dataset_path` | string | yes | Path to the training dataset |

**Request body:**
```json
{
  "model_type": "densenet121",
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 0.001,
  "optimizer": "adam",
  "loss_function": "cross_entropy"
}
```

`model_type` enum: `resnet50`, `vgg16`, `efficientnet`, `DenseNet121`, `ViT`, `SegResNet`, `custom`

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "model_type": "densenet121",
  "created_at": "2024-06-15T10:30:00",
  "completed_at": null,
  "metrics": null
}
```

---

### GET /api/training/jobs

List all local training jobs.

**Response:** Array of `TrainingJob` objects.

---

### GET /api/training/jobs/{job_id}

Get a specific local training job.

**Response:** `TrainingJob` object or 404.

---

### GET /api/training/jobs/{job_id}/metrics

Get training metrics (MLflow/TensorBoard integration).

**Response:**
```json
{
  "job_id": "uuid-string",
  "metrics": {
    "epoch": 5,
    "train_loss": 0.234,
    "val_loss": 0.267,
    "train_accuracy": 0.892,
    "val_accuracy": 0.876,
    "learning_rate": 0.001,
    "tensorboard_url": "http://localhost:6006/#scalars&run=uuid-string",
    "mlflow_url": "http://localhost:5000/#/experiments/1/runs/uuid-string"
  }
}
```

---

### POST /api/training/jobs/{job_id}/stop

Stop a running local training job.

**Response:**
```json
{"job_id": "uuid-string", "status": "stopped"}
```

---

### POST /api/training/sagemaker/validate

Validate data, credentials, IAM role, and image URI before launching a training job.

**Request body:** `SageMakerTrainingConfig` (same schema as `/sagemaker/start`)

**Response:** Validation result object with checks for S3 access, role validity, ECR image accessibility, and dataset structure.

---

### POST /api/training/sagemaker/start

Start a SageMaker BYOC training job.

**Request body:**
```json
{
  "job_name": "my-training-2024-06-15",
  "model_name": "densenet121",
  "task_type": "classification",
  "dimension": "2D",
  "epochs": 20,
  "batch_size": 16,
  "learning_rate": 0.001,
  "num_classes": 1,
  "val_interval": 1,
  "instance_type": "ml.g5.xlarge",
  "instance_count": 1,
  "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-training-byoc-classification:latest",
  "data_s3_uri": "s3://bucket/training-data/",
  "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
  "source_dir": "src/classification",
  "entry_point": "train.py",
  "mlflow_tracking_uri": "arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/my-server",
  "output_s3_uri": "s3://bucket/output/"
}
```

**Response:**
```json
{
  "job_name": "my-training-2024-06-15",
  "status": "InProgress",
  "message": "SageMaker training job started"
}
```

---

### GET /api/training/sagemaker/jobs

List all SageMaker training jobs. Active jobs (`InProgress`, `Stopping`) are refreshed from AWS automatically.

**Query parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `local_only` | bool | `false` | Skip AWS refresh; return cached state only |

**Response:**
```json
{
  "jobs": [
    {
      "job_name": "my-training-2024-06-15",
      "status": "Completed",
      "model_name": "densenet121",
      "task_type": "classification",
      "instance_type": "ml.g5.xlarge",
      "model_artifacts": "s3://bucket/output/my-training-2024-06-15/output/model.tar.gz",
      "creation_time": "2024-06-15T10:30:00Z",
      "training_end_time": "2024-06-15T11:15:00Z",
      "billable_seconds": 2700,
      "estimated_cost_usd": 3.04,
      "mlflow_tracking_uri": "arn:aws:sagemaker:...",
      "archived": false,
      "comment": "",
      "artifacts_deleted": false,
      "result": null
    }
  ]
}
```

---

### GET /api/training/sagemaker/jobs/{job_name}/status

Get current status of a SageMaker training job from AWS.

**Response:**
```json
{
  "job_name": "my-training-2024-06-15",
  "status": "Completed",
  "model_artifacts": "s3://...",
  "billable_seconds": 2700,
  "estimated_cost_usd": 3.04,
  "creation_time": "2024-06-15T10:30:00Z",
  "training_end_time": "2024-06-15T11:15:00Z"
}
```

---

### POST /api/training/sagemaker/jobs/{job_name}/stop

Stop a running SageMaker training job.

**Response:**
```json
{"job_name": "my-training-2024-06-15", "status": "Stopping"}
```

---

### GET /api/training/sagemaker/jobs/{job_name}/result

Fetch `result.json` from the job's `model.tar.gz`. Downloads from S3 on first call; cached thereafter.

**Response:**
```json
{
  "job_name": "my-training-2024-06-15",
  "result": {
    "model_name": "densenet121",
    "num_classes": 8,
    "final_epoch": 20,
    "train_accuracy": 0.95,
    "val_accuracy": 0.91,
    "best_val_accuracy": 0.92,
    "data_s3_uri": "s3://bucket/data/",
    "task_type": "classification",
    "dimension": "2D",
    "learning_rate": 0.001,
    "batch_size": 16
  }
}
```

If `result.json` is absent from the archive, a reconstructed result is returned with `"reconstructed": true` and accuracy fields set to `null`.

---

### DELETE /api/training/sagemaker/jobs/{job_name}

Remove a terminal-state job from local tracking. Cannot remove active jobs.

**Response:**
```json
{"job_name": "my-training-2024-06-15", "removed": true}
```

**Errors:** 404 (not found), 400 (job still active)

---

### POST /api/training/sagemaker/jobs/{job_name}/archive

Archive a job (hidden from default list).

**Response:**
```json
{"job_name": "my-training-2024-06-15", "archived": true}
```

---

### POST /api/training/sagemaker/jobs/{job_name}/comment

Add or update a free-text comment on a job.

**Request body:**
```json
{"comment": "Best run so far — 92% val accuracy"}
```

**Response:**
```json
{"job_name": "my-training-2024-06-15", "comment": "Best run so far — 92% val accuracy"}
```

---

### DELETE /api/training/sagemaker/jobs/{job_name}/artifacts

Delete `model.tar.gz` from S3 for a completed job.

**Response:**
```json
{"job_name": "my-training-2024-06-15", "deleted_uri": "s3://bucket/output/model.tar.gz"}
```

---

### POST /api/training/sagemaker/finetune

Start a fine-tuning job from a completed job's checkpoint.

**Request body:** Same as `/sagemaker/start` with these additional fields:
```json
{
  "checkpoint_s3_uri": "s3://bucket/output/previous-job/output/model.tar.gz",
  "fine_tuned_from": "previous-job-name"
}
```

`checkpoint_s3_uri` is required for fine-tuning.

**Response:**
```json
{
  "job_name": "my-finetune-2024-06-16",
  "status": "InProgress",
  "fine_tuned_from": "previous-job-name",
  "message": "Fine-tuning job started"
}
```

---

### GET /api/training/sagemaker/mlflow-url

Resolve a SageMaker MLflow tracking server ARN to a presigned HTTP URL.

**Query parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `arn` | string | yes | Full ARN or server name |

**Response:**
```json
{"url": "https://mlflow-12345.studio.us-east-1.sagemaker.aws/..."}
```

Presigned URLs expire after 5 minutes. Handles both `mlflow-tracking-server/<name>` and `mlflow-app/<name>` (Studio-managed) ARN formats.

---

### POST /api/training/sagemaker/sync

Pull up to 100 training jobs from AWS and merge into local store.

**Query parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_contains` | string | none | Filter AWS query by job name substring |

**Response:**
```json
{
  "added": ["new-job-1", "new-job-2"],
  "updated": ["existing-job-1"]
}
```

---

## /api/deployment — SageMaker Endpoint Deployment

### POST /api/deployment/deploy

Create a SageMaker endpoint or import/track an existing one.

**Request body:**
```json
{
  "endpoint_name": "my-classifier-endpoint",
  "model_artifacts": "s3://bucket/output/job/output/model.tar.gz",
  "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest",
  "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
  "instance_type": "ml.m5.xlarge",
  "instance_count": 1,
  "inference_script": "inference.py",
  "environment": {},
  "async_endpoint": false,
  "output_s3_uri": null,
  "max_concurrent": 10,
  "enable_data_capture": false,
  "capture_s3_uri": null,
  "capture_sampling_pct": 100
}
```

| Field | Default | Notes |
|-------|---------|-------|
| `model_artifacts` | — | Omit to import/track an existing endpoint by name only |
| `image_uri` | — | Required when deploying a new endpoint |
| `role_arn` | — | Required when deploying a new endpoint |
| `async_endpoint` | `false` | Set `true` for async inference; `output_s3_uri` required |
| `enable_data_capture` | `false` | Enable capture at creation time |
| `capture_sampling_pct` | `100` | 1–100 |

**Response (new endpoint):**
```json
{
  "endpoint_name": "my-classifier-endpoint",
  "status": "Creating",
  "model_name": "my-classifier-endpoint-model",
  "endpoint_config_name": "my-classifier-endpoint-config"
}
```

**Response (import existing):**
```json
{
  "endpoint_name": "my-classifier-endpoint",
  "status": "InService",
  "imported": true
}
```

---

### GET /api/deployment/deployments

List all deployments with refreshed status from AWS. `Deleted` entries are excluded.

**Response:**
```json
{
  "deployments": [
    {
      "endpoint_name": "my-classifier-endpoint",
      "status": "InService",
      "instance_type": "ml.m5.xlarge",
      "model_artifacts": "s3://...",
      "creation_time": "2024-06-15T12:00:00Z",
      "invocation_url": "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-classifier-endpoint/invocations"
    }
  ]
}
```

---

### GET /api/deployment/deployments/{endpoint_name}/status

Get current endpoint status from AWS.

**Response:**
```json
{
  "endpoint_name": "my-classifier-endpoint",
  "status": "InService",
  "creation_time": "2024-06-15T12:00:00Z"
}
```

---

### DELETE /api/deployment/deployments/{endpoint_name}

Delete an endpoint, its endpoint config, and SageMaker model. The local record is removed regardless of AWS API success.

**Response:**
```json
{"endpoint_name": "my-classifier-endpoint", "status": "Deleting"}
```

---

### POST /api/deployment/invoke-async

Submit an async inference request to a SageMaker async endpoint.

**Request body:**
```json
{
  "endpoint_name": "my-async-endpoint",
  "input_s3_uri": "s3://bucket/inputs/request.json",
  "output_s3_uri": "s3://bucket/outputs/"
}
```

**Response:**
```json
{
  "output_location": "s3://bucket/outputs/uuid.out",
  "inference_id": "uuid-string"
}
```

---

### POST /api/deployment/sync

Pull endpoints from AWS and merge into local store.

**Query parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_contains` | string | none | Filter by endpoint name substring |

**Response:**
```json
{
  "added": ["new-endpoint-1"],
  "updated": ["existing-endpoint-1"]
}
```

---

## /api/inference — Real-time Endpoint Invocation

### POST /api/inference/invoke

Invoke a SageMaker real-time endpoint with a JSON payload.

**Request body:**
```json
{
  "endpoint_name": "my-classifier-endpoint",
  "task_type": "classification",
  "input_type": "s3_uri",
  "input_value": "s3://bucket/images/scan.dcm",
  "output_s3_uri": "s3://bucket/results/"
}
```

| Field | Values | Description |
|-------|--------|-------------|
| `task_type` | `classification`, `segmentation` | Determines response format |
| `input_type` | `s3_uri`, `file_path` | How to send the input to the endpoint |
| `output_s3_uri` | — | Optional; saves raw response to S3 |

**Response (classification):**
```json
{
  "predictions": [[0.1034, 0.1202, 0.1366, 0.0866, 0.1258, 0.1408, 0.1155, 0.1713]],
  "predicted_class": [7],
  "confidence": [0.1713],
  "task_type": "classification",
  "s3_result_path": "s3://bucket/results/my-classifier-endpoint-result.json"
}
```

**Response (segmentation):**
```json
{
  "mask_nifti_b64": "<base64-encoded .nii.gz bytes>",
  "shape": [128, 128, 64],
  "voxels_positive": 14823,
  "voxels_total": 1048576,
  "volume_fraction": 0.0141,
  "task_type": "segmentation"
}
```

---

### POST /api/inference/invoke-local

Invoke an endpoint with raw file bytes uploaded from the browser.

**Query parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `endpoint_name` | string | yes | SageMaker endpoint name |
| `task_type` | string | no | `classification` (default) or `segmentation` |

**Request body:** Raw binary file content (the entire request body is the file).

**Response:** Same format as `/invoke` based on `task_type`.

---

### GET /api/inference/endpoints

List all `InService` SageMaker endpoints.

**Response:**
```json
{
  "endpoints": [
    {"name": "my-classifier-endpoint", "status": "InService", "created": "2024-06-15T12:00:00Z"}
  ]
}
```

---

## /api/evaluation — Endpoint Evaluation via Processing Jobs

### POST /api/evaluation/submit

Submit a SageMaker Processing Job to evaluate a deployed endpoint against a labeled test set.

**Request body:**
```json
{
  "endpoint_name": "my-classifier-endpoint",
  "data_s3_uri": "s3://bucket/test-data/",
  "output_s3_uri": "s3://bucket/eval-results/my-eval",
  "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sm-inference-byoc-classification:latest",
  "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
  "instance_type": "ml.m5.xlarge",
  "task_type": "classification"
}
```

`task_type`: `classification` or `segmentation`

**Response:**
```json
{
  "job_name": "eval-my-classifier-20240615-103000",
  "status": "InProgress",
  "endpoint_name": "my-classifier-endpoint",
  "data_s3_uri": "s3://bucket/test-data/",
  "output_s3_uri": "s3://bucket/eval-results/my-eval",
  "task_type": "classification",
  "created_at": "2024-06-15T10:30:00Z"
}
```

---

### GET /api/evaluation/jobs

List all evaluation jobs. In-progress jobs are refreshed from AWS.

**Response:**
```text
{"jobs": [...]}
```

---

### GET /api/evaluation/jobs/{job_name}

Get status and metrics for a specific evaluation job.

**Response:** Job record with status, metrics (once completed), and cached result paths.

---

### DELETE /api/evaluation/jobs/{job_name}

Remove a job record and its local cache. Does not delete S3 outputs.

**Response:**
```json
{"deleted": "eval-my-classifier-20240615-103000"}
```

---

### POST /api/evaluation/jobs/{job_name}/download-results

Download output files from S3 into the local cache.

**Query parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plots_only` | bool | `false` | Download only PNG plots (faster) |

**Response:**
```json
{
  "cached": {
    "metrics.json": "/evaluation-results/job-name/metrics.json",
    "confusion_matrix.png": "/evaluation-results/job-name/confusion_matrix.png",
    "roc_auc.png": "/evaluation-results/job-name/roc_auc.png"
  },
  "job_name": "eval-my-classifier-20240615-103000"
}
```

---

### GET /api/evaluation/jobs/{job_name}/plots

Return local cache URLs for plot files.

**Response:**
```json
{
  "plots": {
    "confusion_matrix.png": "/evaluation-results/job-name/confusion_matrix.png",
    "roc_auc.png": "/evaluation-results/job-name/roc_auc.png"
  },
  "task_type": "classification",
  "output_s3_uri": "s3://bucket/eval-results/my-eval",
  "source": "cache"
}
```

Classification plots: `confusion_matrix.png`, `roc_auc.png`, `subgroup_accuracy.png`
Segmentation plots: `dice_distribution.png`

---

### GET /api/evaluation/jobs/{job_name}/subgroup-metrics

Fetch `subgroup_metrics.json` from the job's S3 output. Available only if the test data contained DICOM files with demographic tags.

**Response:** JSON object with per-subgroup accuracy/metrics breakdowns.

**Errors:** 404 if `subgroup_metrics.json` not found.

---

### POST /api/evaluation/sync

Pull evaluation processing jobs from AWS and merge into local store.

**Query parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_contains` | string | none | Filter by job name substring |

**Response:**
```json
{
  "added": ["eval-job-1"],
  "updated": ["eval-job-2"]
}
```

---

## /api/monitoring — SageMaker Model Monitor

### GET /api/monitoring/capture-summary

Read and aggregate the latest captured traffic for an endpoint.

**Query parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint_name` | string | — | Required. Endpoint to analyze |
| `capture_s3_uri` | string | — | Required. Root S3 prefix for capture data |
| `max_files` | int | `20` | Maximum `.jsonl` files to scan |

**Response:**
```json
{
  "class_distribution": {"class_a": 45, "class_b": 32, "class_c": 23},
  "confidence_histogram": [0, 2, 5, 8, 12, 18, 22, 15, 10, 8],
  "avg_confidence": 0.847,
  "recent_records": [
    {
      "time": "2024-06-15T14:30:00Z",
      "predicted_class": "class_a",
      "confidence": 0.92,
      "probabilities": {"class_a": 0.92, "class_b": 0.05, "class_c": 0.03}
    }
  ],
  "total_records": 100,
  "files_scanned": 5
}
```

Returns a zero-record summary with a descriptive message if no capture files are found.

---

### POST /api/monitoring/data-capture

Enable or update data capture on an existing SageMaker endpoint.

**Request body:**
```json
{
  "endpoint_name": "my-classifier-endpoint",
  "capture_s3_uri": "s3://bucket/capture-data/",
  "sampling_percentage": 100
}
```

`sampling_percentage`: 1–100 (default `100`).

**Response:**
```json
{
  "endpoint_name": "my-classifier-endpoint",
  "status": "Updating",
  "capture_s3_uri": "s3://bucket/capture-data/"
}
```

SageMaker updates the endpoint in-place (status shows `Updating` for a few minutes).

---

### POST /api/monitoring/baseline/data-quality

Submit a SageMaker Processing Job to compute data quality baseline statistics and constraints.

**Request body:**
```json
{
  "baseline_data_s3_uri": "s3://bucket/baseline-data/",
  "output_s3_uri": "s3://bucket/baseline-output/",
  "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
  "instance_type": "ml.m5.xlarge",
  "job_name": "dq-baseline-20240615"
}
```

`job_name` is optional (auto-generated if omitted).

**Response:**
```json
{
  "job_name": "dq-baseline-20240615",
  "status": "InProgress",
  "type": "baseline",
  "output_s3_uri": "s3://bucket/baseline-output/",
  "created_at": "2024-06-15T10:30:00Z"
}
```

---

### POST /api/monitoring/schedules/data-quality

Create a data quality monitoring schedule.

**Request body:**
```json
{
  "schedule_name": "my-dq-schedule",
  "endpoint_name": "my-classifier-endpoint",
  "baseline_constraints_s3_uri": "s3://bucket/baseline-output/constraints.json",
  "baseline_statistics_s3_uri": "s3://bucket/baseline-output/statistics.json",
  "output_s3_uri": "s3://bucket/monitor-output/",
  "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
  "instance_type": "ml.m5.xlarge",
  "schedule_expression": "cron(0 * ? * * *)"
}
```

**Response:**
```json
{
  "schedule_name": "my-dq-schedule",
  "endpoint_name": "my-classifier-endpoint",
  "status": "Scheduled",
  "type": "data_quality",
  "schedule_expression": "cron(0 * ? * * *)",
  "created_at": "2024-06-15T10:30:00Z"
}
```

---

### POST /api/monitoring/schedules/model-quality

Create a model quality monitoring schedule.

**Request body:**
```json
{
  "schedule_name": "my-mq-schedule",
  "endpoint_name": "my-classifier-endpoint",
  "ground_truth_s3_uri": "s3://bucket/ground-truth/",
  "output_s3_uri": "s3://bucket/mq-output/",
  "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
  "instance_type": "ml.m5.xlarge",
  "schedule_expression": "cron(0 * ? * * *)",
  "problem_type": "MulticlassClassification",
  "inference_attribute": "predicted_label",
  "probability_attribute": null,
  "ground_truth_attribute": "label"
}
```

`problem_type`: `BinaryClassification`, `MulticlassClassification`, or `Regression`.

**Response:**
```json
{
  "schedule_name": "my-mq-schedule",
  "endpoint_name": "my-classifier-endpoint",
  "status": "Scheduled",
  "type": "model_quality",
  "created_at": "2024-06-15T10:30:00Z"
}
```

---

### GET /api/monitoring/schedules

List all monitoring schedules (newest first).

**Query parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `local_only` | bool | `false` | Skip AWS refresh |

**Response:**
```text
{"schedules": [...]}
```

---

### GET /api/monitoring/baselines

List all baseline jobs.

**Query parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `local_only` | bool | `false` | Skip AWS refresh for in-progress baselines |

**Response:**
```text
{"baselines": [...]}
```

---

### GET /api/monitoring/schedules/{schedule_name}

Get status and latest execution summary for a monitoring schedule.

**Response:**
```json
{
  "schedule_name": "my-dq-schedule",
  "status": "Scheduled",
  "last_execution_status": "Completed",
  "last_execution_time": "2024-06-15T14:00:00Z",
  "processing_job_arn": "arn:aws:sagemaker:...",
  "failure_reason": null
}
```

---

### DELETE /api/monitoring/schedules/{schedule_name}

Stop and delete a monitoring schedule from AWS and remove from local store.

**Response:**
```json
{"deleted": "my-dq-schedule"}
```

---

### DELETE /api/monitoring/baselines/{job_name}

Remove a baseline job record from local tracking.

**Response:**
```json
{"deleted": "dq-baseline-20240615"}
```

---

## /api/fairness — Demographic Analysis

### POST /api/fairness/scan

Scan DICOM files under an S3 prefix and extract demographic distributions for bias auditing.

**Request body:**
```json
{
  "s3_uri": "s3://bucket/dicom-data/",
  "max_files": 200
}
```

**Response:**
```json
{
  "total_found": 200,
  "total_scanned": 195,
  "errors": 5,
  "distributions": {
    "PatientSex": {"M": 98, "F": 92, "": 5},
    "Modality": {"CR": 150, "DX": 45},
    "BodyPartExamined": {"SPINE": 120, "CHEST": 75}
  },
  "age_buckets": {
    "0-17": 12,
    "18-29": 35,
    "30-44": 52,
    "45-59": 48,
    "60-74": 33,
    "75+": 15
  },
  "missing_rates": {
    "PatientSex": 2.6,
    "PatientAge": 5.1,
    "EthnicGroup": 85.0,
    "PatientRace": 90.0
  },
  "sample": [
    {
      "file": "patient001/image.dcm",
      "PatientSex": "M",
      "PatientAge": "045Y",
      "Modality": "CR"
    }
  ]
}
```

Extracts: `PatientSex`, `PatientAge`, `EthnicGroup`, `PatientRace`, `Modality`, `BodyPartExamined`. Age strings (`045Y`, `6M`, `2W`) are normalized to integer years and bucketed. Sample is capped at 50 rows.

---

## /api/report — Paper / Report Generation

### GET /api/report/jobs

List completed training and evaluation jobs available for report generation.

**Response:**
```json
{
  "training_jobs": [
    {"job_name": "my-training-2024-06-15", "status": "Completed", "model_name": "densenet121"}
  ],
  "evaluation_jobs": [
    {"job_name": "eval-my-classifier-20240615", "status": "Completed", "endpoint_name": "my-endpoint"}
  ]
}
```

Returns only non-archived, `Completed` training jobs and `Completed` evaluation jobs.

---

### POST /api/report/generate

Generate structured academic paper sections from selected jobs.

**Request body:**
```json
{
  "title": "Spine X-ray Classification Using DenseNet-121",
  "training_job_names": ["my-training-2024-06-15"],
  "evaluation_job_names": ["eval-my-classifier-20240615"],
  "sections": ["abstract", "methods", "results", "discussion"],
  "include_fairness": true
}
```

**Response:**
```json
{
  "title": "Spine X-ray Classification Using DenseNet-121",
  "sections": {
    "abstract": "This study presents a deep learning approach...",
    "methods": "We trained a DenseNet-121 model...",
    "results": "The model achieved 92% validation accuracy...",
    "discussion": "Our results demonstrate..."
  }
}
```

