# Deployment Guide

This guide covers deploying trained models to SageMaker endpoints for real-time or asynchronous inference.

## Prerequisites

- A completed training job with a `model.tar.gz` artifact in S3
- An inference container image pushed to ECR (classification or segmentation)
- A SageMaker execution role with endpoint creation permissions

## Deploy a Model

### Step 1 — Open the Deployment tab

Navigate to `http://localhost:8000` and click the **Deployment** tab.

### Step 2 — Fill out the deploy form

| Field | Description | Example |
|---|---|---|
| **Endpoint Name** | Unique name for the endpoint | `spine-classifier-v2` |
| **Model Artifacts** | S3 URI of `model.tar.gz` | `s3://bucket/output/job/output/model.tar.gz` |
| **Inference Image URI** | ECR URI of the inference container | from `.env.local` |
| **Role ARN** | SageMaker execution role | from `.env.local` |
| **Instance Type** | Endpoint instance type | `ml.g5.xlarge` |

### Step 3 — Submit

Click **Deploy**. The dashboard creates three AWS resources in sequence:
1. SageMaker Model
2. Endpoint Configuration
3. Endpoint

Status transitions: `Creating` → `InService` (or `Failed`).

### Step 4 — Wait for InService

The deployment list auto-refreshes status from AWS. Once the endpoint shows `InService`, it is ready to receive inference requests.

## Real-Time vs Async Endpoints

### Real-time (default)

- Synchronous request/response
- Best for single-image inference with low latency requirements
- Timeout: 300 seconds (sufficient for 3D segmentation)

### Async endpoint

Set `async_endpoint: true` in the deploy request to create an asynchronous endpoint:

| Field | Required | Description |
|---|---|---|
| `async_endpoint` | yes | Set to `true` |
| `output_s3_uri` | yes | S3 prefix where results are written |
| `max_concurrent` | no | Max concurrent invocations per instance (default `10`) |

Async endpoints are useful for:
- Batch inference workloads
- Large 3D volumes that exceed real-time timeout limits
- Cost optimization (scale to zero when idle)

## Enable Data Capture at Deploy Time

Check **Enable Data Capture** before deploying to record all inference traffic from the first request:

| Field | Description |
|---|---|
| **Enable Data Capture** | Check to activate |
| **Capture S3 URI** | S3 prefix for captured traffic (e.g. `s3://bucket/capture/`) |
| **Capture Sampling %** | Percentage of requests to capture (1-100, default `100`) |

This is equivalent to calling the monitoring data-capture API separately, but happens atomically during deployment.

## Import an Existing Endpoint

To track an endpoint that was created outside the dashboard (e.g. via the AWS console or CLI):

1. Click **Import Endpoint** (or call `POST /api/deployment` with only the endpoint name)
2. Omit `model_artifacts` — the dashboard registers the endpoint without creating a new artifact
3. The endpoint appears in the deployment list and its status is polled from AWS

## Sync from AWS

Click **Sync from AWS** to pull existing endpoints from your AWS account and merge them into the local store:

- Optional `?name_contains=` filter to narrow results
- Already-tracked endpoints have their status updated
- New endpoints are added with minimal records

## Delete an Endpoint

### From the dashboard

Click the **Delete** button on any endpoint in the list. This:
1. Deletes the SageMaker endpoint
2. Deletes the endpoint configuration
3. Deletes the SageMaker model resource
4. Removes the local record from `data/deployments.json`

The local record is always cleaned up via `try/finally`, even if an AWS API call fails.

### External deletion handling

If an endpoint is deleted outside the dashboard (AWS console, CLI, etc.):
- The next status poll detects the missing endpoint
- The local record is marked as `"Deleted"`
- `Deleted` entries are filtered out of the list response

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| Endpoint stuck in `Creating` | Container startup failure | Check CloudWatch logs for the endpoint |
| `Failed` status | Missing inference script or bad model | Verify `model.tar.gz` contains `inference.py` and `best_model.pth` |
| Permission denied | Role ARN lacks permissions | Ensure the role has `sagemaker:CreateEndpoint`, `ecr:GetAuthorizationToken`, `s3:GetObject` |
| Timeout on invocation | Model too large or slow | Increase instance type or use async endpoint |
