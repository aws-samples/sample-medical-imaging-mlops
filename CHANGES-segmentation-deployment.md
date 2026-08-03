# Segmentation Endpoint Deployment — Changes Summary

## Problem

The segmentation endpoint was deployed but never became healthy. Every `/ping` health check returned HTTP 500, and async inference requests timed out without producing results.

**Root cause:** The SageMaker PyTorch serving container was not loading the custom `inference.py`. Instead, it used the default handler (`default_pytorch_inference_handler.py`), which requires exactly one `.pth` file. Our `model.tar.gz` contained two (`best_model.pth` and `model.pth`), so the default handler raised:

```
ValueError: Exactly one .pth or .pt file is required for PyTorch models: ['best_model.pth', 'model.pth']
```

## Why the Custom Handler Wasn't Loaded

The SageMaker PyTorch serving container discovers custom inference code by looking for `code/inference.py` inside the extracted `model.tar.gz` at `/opt/ml/model/code/inference.py`. Our original archive only contained model weights and metadata — no `code/` directory.

Setting `SAGEMAKER_PROGRAM` as an environment variable in `create_model()` is not sufficient for the standard PyTorch container. It only works when the SageMaker Python SDK packages the code (via `PyTorchModel(source_dir=..., entry_point=...)`).

---

## Changes Made

### 1. `src/inference/segmentation/inference.py` — Updated to Match Reference

**Before:** Used different response keys (`mask_nifti_b64`, `voxels_positive`, `volume_fraction`) and a `model_def.py`-based dynamic loading approach.

**After:** Aligned with the reference implementation from the working lab notebook:

| Aspect | Before | After |
|--------|--------|-------|
| Input key | `s3_uri` or `file_path` | `file_path` only |
| Model loading | Dynamic `model_def.py` import | Direct SegResNet/SwinUNETR from config.json |
| Weight loading | Single file only | Tries `best_model.pth`, `final_model.pth`, `model.pth`, `model.pt` in order |
| Preprocessing | Basic resize + scale | Orientation (RAS) + Spacing + Scale + Resize + EnsureType |
| Inference | Direct `model(tensor)` | `SlidingWindowInferer` with overlap |
| Response keys | `mask_nifti_b64`, `voxels_positive`, `voxels_total`, `volume_fraction` | `segmentation_mask_base64`, `segmented_voxels`, `total_voxels`, `segmentation_percentage`, `model_name`, `original_shape`, `segmentation_shape` |
| Error handling | None (crashes propagate) | Returns `{"error": "...", "model_name": "..."}` |

### 2. `deploy_and_test_segmentation.py` — Client Fixes

| Change | Reason |
|--------|--------|
| Payload key `s3_uri` → `file_path` | Match `input_fn` expectation |
| Result keys updated in `print_result()` | Match `predict_fn`/`output_fn` response |
| Added error response handling | `predict_fn` can return `{"error": ...}` |
| Added `SAGEMAKER_SUBMIT_DIRECTORY` env var | Tells container where to find inference code |
| Added `SAGEMAKER_MODEL_SERVER_TIMEOUT: "300"` | Prevents timeouts on large 3D volumes |
| Added `SAGEMAKER_MODEL_SERVER_WORKERS: "1"` | One worker per GPU for memory-intensive segmentation |
| Added `MaxConcurrentInvocationsPerInstance: 2` | Limits async concurrency for GPU memory |
| Input S3 key uses timestamp | Avoid overwriting previous test inputs |
| `DEPLOY_TIMEOUT` increased to 900s | GPU instances take 8-12 min to provision |
| `INFERENCE_TIMEOUT` increased to 900s | Accounts for cold starts |
| Added failure path polling | Detects inference failures instead of just timing out |

### 3. `repackage_model.py` — New Script

Downloads the existing `model.tar.gz` from S3, adds a `code/` directory containing:
- `code/inference.py` — from `src/inference/segmentation/inference.py`
- `code/requirements.txt` — from `src/inference/segmentation/requirements.txt`

Then re-uploads to the same S3 path. This is what makes the container discover and use our custom handler.

### 4. Helper Scripts — New

| Script | Purpose |
|--------|---------|
| `warm_endpoint.py` | Scale endpoint to 1 instance and wait for InService |
| `delete_endpoint.py` | Clean up endpoint, config, and model resources |
| `check_endpoint_logs.py` | Pull CloudWatch logs for debugging container failures |

---

## Deployment Workflow

```bash
# 1. Repackage model archive with inference code
python repackage_model.py

# 2. Delete old (broken) endpoint if it exists
python delete_endpoint.py

# 3. Deploy and test
python deploy_and_test_segmentation.py \
    --model-data s3://medml-dashboard/model-output/segmentation/segmentation-segresnet-20260729-1559/output/model.tar.gz \
    --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole \
    --test-image s3://test-imaging-workflows/segmentation/test/10090344/img.nii.gz
```

---

## Architecture

```
Client (deploy_and_test_segmentation.py)
    │
    ├── deploy: create_model → create_endpoint_config → create_endpoint
    └── test: PUT {"file_path": "s3://..."} → invoke_endpoint_async
                     │
                     ▼
            SageMaker Async Endpoint (ml.g5.xlarge)
            Container: PyTorch 2.1.0, Python 3.12
            model.tar.gz extracted to /opt/ml/model/
                ├── best_model.pth, model.pth (weights)
                ├── model_def.py, config.json (metadata)
                └── code/
                    ├── inference.py  ← custom handler
                    └── requirements.txt (monai, nibabel, etc.)
                         │
                         ├── model_fn: Load SegResNet/SwinUNETR from config
                         ├── input_fn: Parse S3 URI, download NIfTI
                         ├── predict_fn: SlidingWindowInferer → binary mask
                         └── output_fn: JSON + base64 segmentation
                         │
                         ▼
                    S3 Output (JSON result with mask)
```

---

## Key Lesson

When using raw `boto3.create_model()` with a standard SageMaker PyTorch serving container (not a truly custom BYOC image), the custom inference script **must** be inside `model.tar.gz` at `code/inference.py`. The `SAGEMAKER_PROGRAM` environment variable alone is not enough — it's only used when the SageMaker Python SDK packages the deployment.
