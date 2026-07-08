# Monitoring Guide

This guide covers the full SageMaker Model Monitor lifecycle: data capture, traffic analysis, baselines, and monitoring schedules.

## Overview

Model monitoring detects data quality drift and prediction drift on deployed endpoints. The workflow is:

1. **Enable data capture** — record inference requests and responses to S3
2. **View captured traffic** — analyze prediction distributions and confidence scores
3. **Run a data quality baseline** — compute reference statistics from training data
4. **Create a monitoring schedule** — automatically compare live traffic against the baseline

## Step 1 — Enable Data Capture

Data capture records every inference request and response as `.jsonl` files in S3.

### Option A: At deploy time (recommended)

When deploying an endpoint, check **Enable Data Capture** in the form:
- **Capture S3 URI** — e.g. `s3://your-bucket/capture/`
- **Capture Sampling %** — percentage of requests to capture (1-100, default `100`)

Capture is active from the first inference call.

### Option B: After deployment

1. Go to **Monitoring → Data Capture**
2. Select the endpoint from the dropdown
3. Enter the **Capture S3 URI** (root prefix only, e.g. `s3://your-bucket/capture/`)
4. Set the **Sampling Percentage** (default `100`)
5. Click **Enable**

SageMaker updates the endpoint in-place (status shows `Updating` for a few minutes).

### API

```bash
curl -X POST http://localhost:8000/api/monitoring/data-capture \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint_name": "spine-classifier-v2",
    "capture_s3_uri": "s3://my-bucket/capture/",
    "sampling_percentage": 100
  }'
```

### S3 path structure

SageMaker writes captured data to:
```
s3://your-bucket/capture/<endpoint-name>/AllTraffic/<year>/<month>/<day>/<hour>/
```

Enter the root prefix only — not the full dated path.

## Step 2 — View Captured Traffic (Capture Viewer)

After making inference calls, analyze the captured traffic:

1. Go to **Monitoring → Capture Viewer**
2. Select the endpoint from the dropdown — the Capture S3 URI auto-fills if capture was enabled via the dashboard
3. Set **Max Files** (default `20`)
4. Click **Analyse**

### What the viewer shows

- **Summary cards** — total records, files scanned, average confidence, classes seen
- **Prediction class distribution** — bar chart of predicted classes
- **Confidence score histogram** — 10-bucket histogram of confidence values
- **Recent captures table** — up to 50 rows with per-prediction probabilities

### API

```bash
curl "http://localhost:8000/api/monitoring/capture-summary?endpoint_name=spine-classifier-v2&capture_s3_uri=s3://bucket/capture/&max_files=20"
```

Returns:
```json
{
  "class_distribution": {"0": 15, "1": 8, "2": 12},
  "confidence_histogram": [0, 2, 1, 0, 3, 5, 8, 10, 6, 0],
  "avg_confidence": 0.72,
  "recent_records": []
}
```

## Step 3 — Run a Data Quality Baseline

A baseline computes statistics and constraints from a reference dataset (typically your training data). These are used by monitoring schedules to detect drift.

1. Go to **Monitoring → Data Quality Baseline**
2. Fill in the form:

| Field | Description | Example |
|---|---|---|
| **Baseline Data S3 URI** | S3 path to your reference dataset | `s3://bucket/training-data/` |
| **Output S3 URI** | Where constraints and statistics are written | `s3://bucket/baseline-output/` |
| **IAM Role ARN** | SageMaker execution role | from `.env.local` |
| **Instance Type** | Processing instance | `ml.m5.xlarge` |
| **Job Name** | (Optional) Custom name | `spine-baseline-v1` |

3. Click **Run Baseline Job**

The job submits a SageMaker Processing Job using the prebuilt Model Monitor container. Once complete, the baseline card shows S3 paths for:
- `constraints.json`
- `statistics.json`

Click **"Use in Schedule"** to auto-fill the monitoring schedule form with these paths.

### API

```bash
curl -X POST http://localhost:8000/api/monitoring/baseline/data-quality \
  -H "Content-Type: application/json" \
  -d '{
    "baseline_data_s3_uri": "s3://bucket/training-data/",
    "output_s3_uri": "s3://bucket/baseline-output/",
    "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole"
  }'
```

## Step 4 — Create a Monitoring Schedule

### Data Quality Schedule

Monitors for data drift by comparing captured traffic against baseline constraints.

1. Go to **Monitoring → Data Quality Schedule**
2. Fill in the form:

| Field | Description | Example |
|---|---|---|
| **Schedule Name** | Unique name | `spine-data-quality-hourly` |
| **Endpoint Name** | Endpoint to monitor | `spine-classifier-v2` |
| **Constraints S3 URI** | From baseline output | auto-filled from baseline |
| **Statistics S3 URI** | From baseline output | auto-filled from baseline |
| **Output S3 URI** | Where violation reports go | `s3://bucket/monitoring-output/` |
| **Schedule Expression** | Cron or rate expression | `cron(0 * ? * * *)` (hourly) |

3. Click **Create Schedule**

### Model Quality Schedule

Monitors prediction quality by comparing endpoint predictions against ground-truth labels in S3.

1. Go to **Monitoring → Model Quality Schedule**
2. Fill in the form:

| Field | Description | Example |
|---|---|---|
| **Schedule Name** | Unique name | `spine-model-quality-daily` |
| **Endpoint Name** | Endpoint to monitor | `spine-classifier-v2` |
| **Problem Type** | Classification type | `MulticlassClassification` |
| **Ground Truth S3 URI** | S3 path to labels | `s3://bucket/ground-truth/` |
| **Inference Attribute** | Response field for prediction | `predicted_class` |
| **Probability Attribute** | (Optional) Probability field | `confidence` |
| **Output S3 URI** | Where reports go | `s3://bucket/model-quality-output/` |
| **Schedule Expression** | Cron or rate | `rate(1 day)` |

3. Click **Create Schedule**

Supported problem types: `BinaryClassification`, `MulticlassClassification`, `Regression`.

## Schedule Management

### List all schedules

Go to **Monitoring → Schedules** or call:

```bash
curl http://localhost:8000/api/monitoring/schedules
```

Schedules are sorted by creation time (newest first). Active schedules are refreshed from AWS.

Pass `?local_only=true` to skip the AWS refresh.

### Refresh a schedule

Click **Refresh** on a schedule card to pull the latest status and most recent execution summary (status, time, Processing Job ARN, failure reason).

```bash
curl http://localhost:8000/api/monitoring/schedules/spine-data-quality-hourly
```

### Delete a schedule

Click **Delete** to stop and delete the monitoring schedule from AWS and remove it from the local store.

```bash
curl -X DELETE http://localhost:8000/api/monitoring/schedules/spine-data-quality-hourly
```

## Schedule Expressions

| Expression | Meaning |
|---|---|
| `cron(0 * ? * * *)` | Every hour at minute 0 |
| `cron(0 0 ? * * *)` | Daily at midnight |
| `cron(0 0 ? * MON *)` | Every Monday at midnight |
| `rate(1 hour)` | Every hour |
| `rate(1 day)` | Every day |

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| No capture files found | Capture not yet enabled or no invocations made | Enable capture and invoke the endpoint |
| Baseline job fails | Insufficient permissions or bad data path | Check the Processing Job logs in CloudWatch |
| Schedule shows no executions | Not enough time elapsed or no captured data | Wait for the next scheduled run; ensure data capture is working |
