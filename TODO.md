# TODO — MedML Training Dashboard

## Training

- [ ] **Custom train.py upload** — Allow users to upload a custom `train.py` (and optionally `utils.py`, `model_def.py`) via the UI. Save to a temp directory on the server and use it as `source_dir` when submitting the SageMaker job.

- [ ] **Expose source_dir field** — Add an optional visible `source_dir` input in the SageMaker form so power users can point to any local folder without editing JS.

- [ ] **Segmentation num_classes** — `src/segmentation/train.py` still accepts `--num_classes` as an argument. Derive it from the data structure (number of unique label values in NIfTI masks) similar to how classification does it from folder names.

- [ ] **Multi-GPU support** — Add DDP/FSDP training strategy selector in the SageMaker form (single GPU, DDP, FSDP) and wire it to the correct entry point script.

## Model Registry

- [ ] **Register model after training** — On SageMaker job completion, register the model artifact in SageMaker Model Registry under a named model package group. Attach training metrics (accuracy, AUC), training job ARN, data S3 URI, and container image URI as metadata.

- [ ] **Approval workflow** — Expose approve/reject actions in the dashboard for models in `PendingManualApproval` state. Only approved versions should be deployable from the UI.

- [ ] **Deploy from registry** — Add a "Deploy from Registry" option in the deployment form that lists approved model versions and deploys directly from the registry version ARN rather than a raw `model.tar.gz` S3 URI.

- [ ] **Model lineage** — Surface lineage info (training job → model version → endpoint) in the dashboard so users can trace exactly what data and code produced a live endpoint.

## Models

- [ ] **Add UNet to segmentation model_def** — UNet is listed in `MODELS` registry but not yet tested end-to-end on SageMaker.

- [ ] **Custom model upload** — Allow importing a custom model architecture file and registering it in the model list.

## Infrastructure

- [ ] **Auto-rebuild container** — Trigger a container rebuild and push to ECR from the dashboard when source files change.

- [ ] **Spot instance support** — Add a checkbox in the SageMaker form for `use_spot_instances` + `max_wait` to reduce training costs.

- [ ] **Warm pool support** — Add `keep_alive_period_in_seconds` option to the SageMaker form.

## Explainability

- [ ] **Grad-CAM visualization** — Add a Grad-CAM service that loads a model checkpoint, runs a forward/backward pass on a DICOM input, and returns a heatmap overlay. Expose via `/api/inference/gradcam` and render in the UI alongside the prediction result. Support classification models (ResNet50, DenseNet121, EfficientNet, ViT) with target layer auto-detection.

- [ ] **Grad-CAM in SageMaker Processing Job** — Run Grad-CAM at scale on a test set via a SageMaker Processing Job using the inference container; save heatmap overlays to S3.

## Model Evaluation

- [ ] **SageMaker Processing Job for endpoint evaluation** — Submit a Processing Job using the inference container image to evaluate a deployed endpoint against a labeled test set in S3. Compute accuracy, AUC, per-class precision/recall/F1, and confusion matrix. Save a JSON report to S3 and persist results in the dashboard.

- [ ] **Evaluation UI** — Add an Evaluation tab in the dashboard to configure and launch evaluation jobs, track their status, and display metric results with a confusion matrix visualization.

- [ ] **Batch Transform evaluation** — Support offline evaluation via SageMaker Batch Transform (no live endpoint required) for large test sets.

## Model Monitoring

- [ ] **SageMaker Model Monitor** — Enable data capture on deployed endpoints and schedule a `DefaultModelMonitor` baseline + monitoring job. Alert on data quality drift (feature distribution shift against the training baseline).

- [ ] **Model quality monitoring** — Schedule a `ModelQualityMonitor` job to track prediction quality metrics (accuracy, AUC) over time against a ground-truth label dataset in S3.

- [ ] **Monitoring dashboard** — Surface Model Monitor violation reports and drift statistics in the dashboard; show per-feature drift scores and flag endpoints with active violations.

## UI / UX

- [ ] **S3 browser for data URI** — Add a browse button next to the Data S3 URI field that opens the S3 bucket browser and lets users select a path directly.

- [ ] **Job logs viewer** — Show CloudWatch logs for a running/completed SageMaker job inline in the dashboard.

- [ ] **Model artifacts download** — Add a button on completed jobs to download or copy the `model.tar.gz` S3 URI.
