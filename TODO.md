# TODO — MedML Training Dashboard

## Training

- [ ] **Custom train.py upload** — Allow users to upload a custom `train.py` (and optionally `utils.py`, `model_def.py`) via the UI. Save to a temp directory on the server and use it as `source_dir` when submitting the SageMaker job.

- [ ] **Expose source_dir field** — Add an optional visible `source_dir` input in the SageMaker form so power users can point to any local folder without editing JS.

- [ ] **Segmentation num_classes** — `src/segmentation/train.py` still accepts `--num_classes` as an argument. Derive it from the data structure (number of unique label values in NIfTI masks) similar to how classification does it from folder names.

- [ ] **Multi-GPU support** — Add DDP/FSDP training strategy selector in the SageMaker form (single GPU, DDP, FSDP) and wire it to the correct entry point script.

## Models

- [ ] **Add UNet to segmentation model_def** — UNet is listed in `MODELS` registry but not yet tested end-to-end on SageMaker.

- [ ] **Custom model upload** — Allow importing a custom model architecture file and registering it in the model list.

## Infrastructure

- [ ] **Auto-rebuild container** — Trigger a container rebuild and push to ECR from the dashboard when source files change.

- [ ] **Spot instance support** — Add a checkbox in the SageMaker form for `use_spot_instances` + `max_wait` to reduce training costs.

- [ ] **Warm pool support** — Add `keep_alive_period_in_seconds` option to the SageMaker form.

## UI / UX

- [ ] **S3 browser for data URI** — Add a browse button next to the Data S3 URI field that opens the S3 bucket browser and lets users select a path directly.

- [ ] **Job logs viewer** — Show CloudWatch logs for a running/completed SageMaker job inline in the dashboard.

- [ ] **Model artifacts download** — Add a button on completed jobs to download or copy the `model.tar.gz` S3 URI.
