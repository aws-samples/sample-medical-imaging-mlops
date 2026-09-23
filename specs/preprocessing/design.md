# Design — MONAI Preprocessing Pipeline

## Guiding principle: one spec, two runners

A single **pipeline spec** (JSON) is the source of truth. A shared
`pipeline_builder` turns the spec into a `monai.transforms.Compose`. Two thin
runners execute that Compose:

- **Local runner** — in-process on the app server, one image, returns stats + a
  preview slice. No S3 writes, no cost.
- **Batch runner** — a script (`src/preprocessing/preprocess.py`) shipped to a
  SageMaker Processing Job that applies the same Compose to every image under an
  S3 input prefix.

Because both runners consume the identical spec through the identical builder,
"what you previewed is what the batch job runs."

## Architecture (fits existing MedAnvil patterns)

```
templates/index.html  ──(Preprocessing tab: form/JSON editor + Preview + Submit)
static/js/preprocessing.js
        │  REST
        ▼
app/routers/preprocessing_router.py     ← mirrors evaluation_router.py
        │
        ▼
app/services/preprocessing_service.py   ← mirrors EvaluationService (JSON store + SM job)
        │
        ├── local preview ─► src/preprocessing/pipeline_builder.py  (MONAI Compose)
        │                     src/preprocessing/local_preview.py
        │
        └── submit batch ─► SageMaker Processing Job
                              runs src/preprocessing/preprocess.py
                              (uses the SAME pipeline_builder.py)
                              reads  s3://.../input/**
                              writes s3://.../output/** (+ dataset.json)

data/preprocessing_jobs.json            ← job state store (like evaluation_jobs.json)
```

## Pipeline spec schema (Pydantic)

```jsonc
{
  "task_type": "segmentation",          // or "classification"
  "keys": ["image", "label"],           // "label" optional (segmentation only)
  "steps": [
    { "type": "load",              "params": { "reader": "auto" } },
    { "type": "ensure_channel_first" },
    { "type": "orientation",       "params": { "axcodes": "RAS" } },
    { "type": "resample",          "params": { "pixdim": [1.5, 1.5, 2.0], "mode": ["bilinear","nearest"] } },
    { "type": "intensity_window",  "params": { "a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1.0, "clip": true } },
    { "type": "crop_foreground",   "params": { "source_key": "image" } },
    { "type": "resize",            "params": { "spatial_size": [128,128,128], "mode": ["area","nearest"] } },
    { "type": "save",              "params": { "output_postfix": "proc", "output_ext": ".nii.gz" } }
  ],
  "dataset_prep": {                      // optional; batch-only
    "enabled": true,
    "split": { "train": 0.7, "val": 0.15, "test": 0.15 },
    "seed": 42
  }
}
```

`Step.params` is validated per-`type` (a discriminated union of Pydantic models),
satisfying Requirement 1.1–1.3. For classification, `keys=["image"]` and label
handling is skipped.

## Component design

### `src/preprocessing/pipeline_builder.py` (shared — the heart of the feature)

- `build_transforms(spec: PipelineSpec) -> monai.transforms.Compose`
- Maps each step `type` → MONAI transform, wiring `keys` and per-key
  interpolation `mode` (bilinear/area for image, nearest for label).
- `STEP_CATALOG`: dict of `type → (transform, param-schema)` — also used to serve
  the catalog to the UI (Requirement 2.4).
- Pure and dependency-light so it imports cleanly both in the app process and
  inside the processing container.

### `src/preprocessing/local_preview.py` (local runner)

- `preview(spec, input_ref) -> PreviewResult`
- Resolves `input_ref` (local path or S3 URI); if S3, downloads to a confined
  temp dir (`validate_path`, cleanup in `finally`).
- Runs the Compose on one sample; collects input/output shape, spacing, intensity
  min/max, and per-step timing.
- Renders a mid-axial slice to PNG (matplotlib, already used by evaluation plots)
  returned as a static URL or base64.
- Never writes to S3, never creates a job (Requirement 3.5).

### `src/preprocessing/preprocess.py` (batch runner — runs in SageMaker)

- CLI: `--input-s3-uri --output-s3-uri --spec-path --task-type`.
- SageMaker Processing mounts input at `/opt/ml/processing/input`, writes to
  `/opt/ml/processing/output`; script enumerates images, builds the Compose from
  the spec, applies per sample, writes processed `.nii.gz`.
- If `dataset_prep.enabled`, writes train/val/test split + Decathlon `dataset.json`
  to the output dir (Requirement 4.4).
- Reuses `pipeline_builder.build_transforms` verbatim.

### `app/services/preprocessing_service.py` (mirrors `EvaluationService`)

- JSON store at `data/preprocessing_jobs.json` with `_load`/`_save`.
- `preview_local(spec, input_ref)` → calls `local_preview.preview`.
- `submit_batch(spec, input_s3_uri, output_s3_uri, image_uri, role_arn, instance_type, sagemaker_client, region)`:
  - Upload `preprocess.py`, `pipeline_builder.py`, `requirements.txt`, and the
    serialized `spec.json` to `s3://<out-bucket>/preprocessing-scripts/<job>/`.
  - `create_processing_job` with entrypoint that `pip install`s reqs then runs
    `preprocess.py` with the spec — same shape as the evaluation job config
    (ProcessingInputs code channel + ProcessingOutputConfig).
  - Job name: `preprocess-<short>-<UTC timestamp>`.
  - Persist record (spec included) and return it.
- `get_status`, `list_jobs`, `delete_job`, `sync_from_aws(NameContains="preprocess-")`
  — copied structurally from `EvaluationService`.

### `app/routers/preprocessing_router.py` (mirrors `evaluation_router.py`)

- `GET  /api/preprocessing/catalog`               → supported steps + param schemas
- `POST /api/preprocessing/preview`               → `{ input_ref, spec }` → stats + preview PNG
- `POST /api/preprocessing/submit`                → `{ input_s3_uri, output_s3_uri, image_uri, role_arn, instance_type, spec }`
- `GET  /api/preprocessing/jobs`                  → list (refresh in-progress)
- `GET  /api/preprocessing/jobs/{job_name}`       → one job
- `DELETE /api/preprocessing/jobs/{job_name}`     → delete record (keep S3)
- `POST /api/preprocessing/sync`                  → sync from AWS
- Uses the shared `aws_service` (sagemaker_client / s3_client), like other routers.
- Registered in `main.py` with prefix `/api/preprocessing`.

### Pydantic models: `app/models/preprocessing.py`

- `Step`, per-type param models, `DatasetPrep`, `PipelineSpec`,
  `PreviewRequest`, `SubmitRequest` — server-side validation (Requirement 1, 7).

### UI: `static/js/preprocessing.js` + Preprocessing tab in `index.html`

- Form/JSON editor for the spec (seeded from `/catalog`).
- **Preview** button → shows before/after stats table + preview slice image.
- **Approve & Run on Dataset** button (enabled after a successful preview) →
  posts to `/submit` with input/output S3 URIs.
- Jobs table with Active/Completed/Failed status, delete, and sync — reusing the
  evaluation tab's JS patterns. All dynamic values HTML-escaped (Requirement 7.5).

## Execution image (Requirement 6)

Prefer reusing the existing **MONAI bundle image** (`Dockerfile.monai-bundle`),
which already ships MONAI + nibabel; add `pydicom` via the job's
`pip install -r requirements.txt` step if not present. Only introduce a dedicated
`Dockerfile.preprocessing` if that reuse proves insufficient.

## Data flow summary

1. User composes spec → **Preview** on one image (local, free) → inspects
   stats + slice → iterates.
2. On approval → **Submit** → Processing Job runs identical pipeline over the S3
   dataset → writes processed volumes (+ `dataset.json`) to output prefix.
3. Output prefix is directly usable as a training job's **Data S3 URI**.

## Security notes (maps to Requirement 7)

- `validate_path()` + confined temp dirs for local preview input; `finally`
  cleanup of downloaded S3 temp files.
- `s3://bucket/key` regex validation before any boto3 call.
- Image loading via MONAI/nibabel/pydicom only — no untrusted deserialization.
- Client-facing errors are sanitized (step name + message, no absolute paths).
- Bucket/role/image come from request + runtime config, never hardcoded.
