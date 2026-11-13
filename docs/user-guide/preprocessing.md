# Preprocessing

> **⚠ Research use only** — preprocessing outputs feed models that are not cleared for clinical use.

The **Preprocessing** tab builds a MONAI transform pipeline, previews it on a single image, and then runs the same pipeline over a whole S3 dataset as a CPU SageMaker Processing Job. The preview and the batch job use the same builder (`src/preprocessing/pipeline_builder.py`), so what you see in the preview is what the job writes.

## 1. Build the pipeline

Under **Steps**, add transforms with **Add Step**, start from **Load Default**, or paste a saved spec with **Import JSON** and then **Validate & Load spec**. The catalog holds 24 deterministic MONAI transforms:

| Group | Transforms |
|---|---|
| Load and layout | `load`, `ensure_channel_first`, `orientation`, `resample` |
| Intensity | `intensity_window`, `scale_intensity`, `scale_intensity_percentiles`, `clip_percentiles`, `threshold_intensity`, `histogram_normalize`, `normalize` |
| Smoothing | `gaussian_smooth`, `median_smooth` |
| Crop, pad and resize | `crop_foreground`, `center_crop`, `spatial_pad`, `divisible_pad`, `resize`, `zoom` |
| Orientation and type | `flip`, `rotate90`, `cast`, `ensure_type` |
| Output | `save` |

Spatial transforms are applied to the image and the label together; intensity transforms apply to the image only. Random augmentations are deliberately left out: they belong in training, not in a saved dataset.

## 2. Preview one image

Enter a local path or an S3 URI to one image under **Input** and click **Run Preview**. The panel shows before/after statistics (shape, spacing, intensity range), per-step timing, and a **Preview Slice**. S3 downloads use the dashboard's connected AWS session.

## 3. Describe the dataset

The batch job reads an [MSD/Decathlon](http://medicaldecathlon.com/) `dataset.json` at the root of the input prefix:

```json
{
  "name": "MyDataset",
  "labels": {"0": "background", "1": "tumour"},
  "training": [{"image": "imagesTr/case001.nii.gz", "label": "labelsTr/case001.nii.gz"}],
  "test": ["imagesTs/case101.nii.gz"]
}
```

- Entries with a `label` are treated as **segmentation**; image-only entries as **classification**.
- For classification without a `dataset.json`, organise images as `<root>/<class-name>/<images>` (ImageFolder) and the classes are discovered from the folder names.
- If neither is found, **Resolve dataset (dry-run)** shows the expected location and an example `dataset.json` you can copy.

Click **Resolve dataset (dry-run)** to check the task type, split counts and the first image/label pairs before you launch a job.

## 4. Run the batch job

| Field | Description |
|---|---|
| **Input S3 URI** | Dataset prefix (the folder containing `dataset.json`) |
| **Output S3 URI** | Where processed images, the new `dataset.json` and `result.json` are written |
| **Processing Image URI** | ECR URI of the `sm-preprocessing` image (auto-filled from config when available) |
| **SageMaker Role ARN** | Execution role (auto-filled from `SAGEMAKER_ROLE_ARN`) |
| **Instance Type** | CPU instance, e.g. `ml.m5.xlarge` or `ml.m5.2xlarge` |

The dashboard uploads the preprocessing scripts and your pipeline spec to `preprocessing-scripts/<job-name>/` in the output bucket and mounts them as the job's code. When `AWS_ACCOUNT_ID` is set, it first checks that the output bucket belongs to your account (see [Configuration](../configuration.md#dashboard-security-settings)). Missing files are skipped and reported rather than failing the job.

The jobs table polls automatically and shows Completed/Failed status (with the failure reason on hover). Use **Sync** to pull jobs from SageMaker and the delete button to remove a job record; S3 outputs are kept.

## Building the image

The job runs in a CPU-only image with CPU PyTorch, so it is much smaller than the GPU training images:

```bash
./build_and_push_preprocessing.sh
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/preprocessing/catalog` | Transform catalog with default parameters |
| `POST` | `/api/preprocessing/validate` | Validate a pipeline spec |
| `GET` | `/api/preprocessing/resolve-dataset?input_s3_uri=` | Dry-run dataset resolution |
| `POST` | `/api/preprocessing/preview` | Run the pipeline on one image |
| `POST` | `/api/preprocessing/submit` | Launch the Processing Job |
| `GET` | `/api/preprocessing/jobs`, `/api/preprocessing/jobs/{job_name}` | List or get jobs |
| `DELETE` | `/api/preprocessing/jobs/{job_name}` | Delete a job record |
| `POST` | `/api/preprocessing/sync` | Sync jobs from SageMaker |
