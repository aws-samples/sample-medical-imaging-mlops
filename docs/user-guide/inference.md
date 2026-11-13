# Inference Guide

This guide covers invoking deployed SageMaker endpoints for classification and segmentation predictions.

## Prerequisites

- A deployed endpoint in `InService` status (see [Deployment Guide](deployment.md))
- The endpoint's task type: `classification` or `segmentation`

## Invoking an Endpoint

### Via the Dashboard UI

1. Go to the **Inference** tab
2. Select the endpoint from the dropdown
3. Choose the input method:
   - **S3 URI** — enter the S3 path to your file
   - **File Upload** — upload a file directly from your browser
4. Click **Invoke**
5. View the response in the results panel

### Via the API — S3 URI input

```bash
curl -X POST http://localhost:8000/api/inference/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint_name": "spine-classifier-v2",
    "task_type": "classification",
    "input_type": "s3_uri",
    "input_value": "s3://<your-bucket-name>/test/image.dcm"
  }'
```

### Via the API — File upload (raw bytes)

```bash
curl -X POST "http://localhost:8000/api/inference/invoke-local?endpoint_name=spine-classifier-v2&task_type=classification" \
  --data-binary @/path/to/image.dcm
```

### Optional: save result to S3

Add `output_s3_uri` to save the response:

```bash
curl -X POST http://localhost:8000/api/inference/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint_name": "spine-classifier-v2",
    "task_type": "classification",
    "input_type": "s3_uri",
    "input_value": "s3://<your-bucket-name>/test/image.dcm",
    "output_s3_uri": "s3://<your-bucket-name>/results/"
  }'
```

The result is saved to `s3://<prefix>/<endpoint_name>-result.json`.

## Supported Input Formats

### Classification endpoints

| Format | How to send |
|---|---|
| **DICOM** (`.dcm`) | Raw bytes with `Content-Type: application/dicom`, or S3 URI, or base64 |
| **PNG / JPG** | Same as DICOM — raw bytes, S3 URI, or base64 |
| **S3 URI** | JSON: `{"file_path": "s3://bucket/key.dcm"}` |
| **Local path** | JSON: `{"file_path": "/opt/ml/input/data/image.dcm"}` (must be under `LOCAL_INPUT_ROOTS`, see below) |
| **Base64** | JSON: `{"file_path": "<base64-encoded bytes>"}` |

### Segmentation endpoints

| Format | How to send |
|---|---|
| **NIfTI** (`.nii.gz`) | Raw bytes with `Content-Type: application/octet-stream` |
| **S3 URI** | JSON: `{"file_path": "s3://bucket/subject/img.nii.gz"}` |
| **Local path** | JSON: `{"file_path": "/opt/ml/input/data/volume.nii.gz"}` (must be under `LOCAL_INPUT_ROOTS`, see below) |
| **Base64** | JSON: `{"file_path": "<base64-encoded NIfTI bytes>"}` |

> **Note:** The segmentation inference script uses the `file_path` key (same as classification). The older `s3_uri` key is no longer used — always use `file_path` for both task types.

### Input restrictions

The inference handlers treat every request as untrusted:

- **Local paths** are accepted only if they resolve inside `LOCAL_INPUT_ROOTS` (default `/opt/ml/input/data,/opt/ml/processing/input`). Anything else, including the model directory, is rejected. Send an S3 URI or the file bytes instead.
- **S3 buckets** can be restricted by setting `ALLOWED_INPUT_BUCKETS` (comma-separated) on the endpoint. When it is unset, any bucket the endpoint role can read is allowed.
- **Output filename patterns** (`output_pattern`) may use only `{patient_id}`, `{filename}`, `{endpoint_name}`, `{timestamp}` and `{model_name}`, with no `..` or leading `/`. Other patterns fall back to the default.
- **MONAI bundle requests:** `label_prompt`, `points` and `point_labels` must be numbers. `config_overrides` accepts plain literal values on non-reserved keys only; MONAI `$` expressions, `@` references and `%` macros are rejected.
- **NV-Segment-CT `checkpoint`** must name a file in the bundle's `models/` directory.

Set these variables in the endpoint's environment (see [Configuration](../configuration.md#inference-endpoint-security)).

## Response Formats

### Classification

```json
{
  "predictions": [[0.1034, 0.1202, 0.1366, 0.0866, 0.1258, 0.1408, 0.1155, 0.1713]],
  "predicted_class": [7],
  "confidence": [0.1713]
}
```

| Field | Type | Description |
|---|---|---|
| `predictions` | `list[list[float]]` | Probability array per input (batched) |
| `predicted_class` | `list[int]` | Index of the highest-probability class |
| `confidence` | `list[float]` | Probability of the predicted class |

### Segmentation

```json
{
  "mask_nifti_b64": "<base64-encoded .nii.gz bytes>",
  "shape": [128, 128, 64],
  "voxels_positive": 14823,
  "voxels_total": 1048576,
  "volume_fraction": 0.0141
}
```

| Field | Type | Description |
|---|---|---|
| `mask_nifti_b64` | `string` | Base64-encoded binary mask as a NIfTI file |
| `shape` | `list[int]` | Dimensions of the output mask |
| `voxels_positive` | `int` | Number of positive (foreground) voxels |
| `voxels_total` | `int` | Total voxels in the mask |
| `volume_fraction` | `float` | Ratio of positive voxels to total |

The segmentation inference pipeline uses `SlidingWindowInferer` for memory-efficient processing of large 3D volumes, with `Orientation` and `Spacing` transforms applied to normalize inputs before prediction. This ensures consistent results regardless of the input volume's original orientation or voxel spacing.

To save the segmentation mask locally:

```python
import base64

mask_bytes = base64.b64decode(response["mask_nifti_b64"])
with open("mask.nii.gz", "wb") as f:
    f.write(mask_bytes)
```

## Async Endpoint Inference

When invoking an endpoint, the dashboard automatically detects whether the target is a real-time or async endpoint and adjusts the invocation method accordingly:

- **Real-time endpoints** — sends a synchronous `InvokeEndpoint` request and returns the response immediately
- **Async endpoints** — sends an `InvokeEndpointAsync` request, which returns an output location in S3 where the result will be written

No additional configuration is needed in the inference UI — select the endpoint and invoke as normal. The dashboard handles the async/real-time distinction transparently.

For async endpoints, the result panel shows the S3 output location and polls for completion automatically.

## Viewing Results in the Prediction Viewer

After a run completes, results can be inspected visually in the
[Prediction Viewer](viewer.md) (experimental):

- **Segmentation** — click **View in viewer** on the result card to open the
  source image with the prediction mask overlaid
- **Classification** — click **View source image in viewer** to open the
  input image with the predicted class, confidence, and probability bars in
  the sidebar

The **Open Prediction Viewer** button on the Inference tab opens the viewer
prefilled with your most recent run.

## CLI Tools

For quick testing without the dashboard:

### Classification

```bash
python tests/invoke_endpoint.py --s3-uri s3://bucket/image.dcm --endpoint my-endpoint
python tests/invoke_endpoint.py --image /path/to/image.dcm --endpoint my-endpoint
```

### Segmentation

```bash
python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
    --s3-uri s3://bucket/subject/img.nii.gz

python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
    --local /path/to/volume.nii.gz

python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
    --s3-uri s3://bucket/img.nii.gz --save-mask /tmp/mask.nii.gz
```

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| Empty mask (0 positive voxels) | Input outside expected HU range | Check that input is CT data in [-100, 500] HU range |
| `ModelError` | Inference script crash | Check CloudWatch logs for the endpoint |
| Timeout | Volume too large | Use async endpoint or a larger instance type |
| `ValidationException` | Endpoint not InService | Wait for deployment to complete or check for failures |
