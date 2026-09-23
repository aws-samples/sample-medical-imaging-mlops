# Requirements — MONAI Preprocessing Pipeline

## Overview

Add a **preprocessing** feature to MedAnvil that lets a user build a MONAI-based
preprocessing pipeline, **test it locally on a single image** (local file or S3
URI) to visually confirm the result, and — once approved — **run the same
pipeline over an entire dataset in S3 as a SageMaker Processing Job**.

The core principle is **one pipeline definition, two execution modes**:

- **Local preview** — run the pipeline on ONE image on the dashboard server,
  return before/after stats (and a preview slice) so the user can iterate on
  parameters quickly and cheaply.
- **Batch (SageMaker Processing Job)** — run the *identical* pipeline over a full
  dataset read from S3, writing processed outputs back to S3, ready to feed a
  training job's Data S3 URI.

The pipeline covers four capability groups:

1. **Format conversion** — DICOM series → NIfTI, and NIfTI passthrough/normalize.
2. **Intensity** — HU windowing/clipping (CT), intensity scaling/normalization.
3. **Spatial** — resampling to target spacing, reorientation (RAS), resize/crop
   to target shape, crop-to-foreground.
4. **Dataset prep** — train/val/test split and Decathlon-style `dataset.json`
   generation over the processed outputs.

> ⚠️ **RESEARCH USE ONLY.** Preprocessing transforms imaging data for research
> and development. It performs no diagnosis. Downstream model outputs are AI
> predictions for research only, not clinical diagnoses. Clinical use requires
> regulatory clearance, clinical validation, and qualified professional review.

## Definitions

- **Pipeline spec** — a JSON description of an ordered list of preprocessing
  steps and their parameters. The single source of truth shared by both
  execution modes.
- **Step** — one named transform with parameters (e.g. `resample` with
  `pixdim=[1.5,1.5,2.0]`), mapped internally to a MONAI transform.
- **Local preview** — executing the pipeline on exactly one image on the app
  server, returning summary stats and an optional preview image.
- **Batch job** — a SageMaker Processing Job that applies the pipeline to every
  image under an S3 input prefix.

## Requirements

### Requirement 1 — Define a preprocessing pipeline

**User story:** As a user, I want to compose an ordered list of preprocessing
steps with parameters, so that I can describe exactly how my images should be
transformed.

#### Acceptance Criteria

1. WHEN the user defines a pipeline THEN the system SHALL accept an ordered list of
   steps, each with a `type` (from the supported catalog) and a `params` object,
   validated server-side against a Pydantic schema.
2. WHEN a step `type` or a parameter is invalid THEN the system SHALL reject the
   pipeline with a descriptive message and SHALL NOT execute anything.
3. WHERE steps have ordering constraints (e.g. load must be first) THEN the system
   SHALL validate ordering and report a clear error if violated.
4. WHEN a pipeline is defined THEN the SAME pipeline spec SHALL be usable for both
   local preview and batch execution without modification.

### Requirement 2 — Supported transform catalog (MONAI-based)

**User story:** As a user, I want a documented set of preprocessing steps backed
by MONAI transforms, so my preview and batch results are consistent with
training-time transforms.

#### Acceptance Criteria

1. WHEN building the pipeline THEN the system SHALL support at minimum these steps,
   each mapped to a MONAI transform:
   - `load` → `LoadImaged` (DICOM series folder or NIfTI file)
   - `ensure_channel_first` → `EnsureChannelFirstd`
   - `orientation` → `Orientationd` (e.g. `axcodes="RAS"`)
   - `resample` → `Spacingd` (target `pixdim`, `mode`)
   - `intensity_window` → `ScaleIntensityRanged` (CT HU `a_min/a_max/b_min/b_max/clip`)
   - `normalize` → `NormalizeIntensityd`
   - `crop_foreground` → `CropForegroundd`
   - `resize` → `Resized` (target `spatial_size`, `mode`)
   - `save` → `SaveImaged` (write `.nii.gz`)
2. WHEN a `load` step targets a DICOM directory THEN the system SHALL stack the
   series into a single volume; WHEN it targets a NIfTI file THEN it SHALL load it
   directly.
3. WHERE a transform is image+label paired (segmentation) THEN the pipeline SHALL
   support applying spatial steps to both `image` and `label` keys with correct
   interpolation modes (nearest for labels).
4. WHEN the catalog is queried THEN the system SHALL expose the list of supported
   step types and their parameter schemas to the UI.

### Requirement 3 — Local single-image preview

**User story:** As a user, I want to run my pipeline on one image locally and see
the result, so that I can iterate on parameters before spending money on a batch
job.

#### Acceptance Criteria

1. WHEN a preview is requested with a local path or S3 URI to a single image
   (DICOM dir or NIfTI) THEN the system SHALL run the pipeline on the app server
   and return: input shape/spacing/intensity range, output shape/spacing/intensity
   range, and per-step timing.
2. WHEN the S3 URI is provided THEN the system SHALL download the object(s) to a
   confined temp directory before processing and SHALL clean up afterward.
3. WHEN a preview completes THEN the system SHALL optionally return a preview slice
   (e.g. mid-axial PNG) of the processed volume for visual inspection.
4. IF preview processing fails THEN the system SHALL return the failing step and a
   descriptive error without leaking internal absolute paths to the client.
5. WHEN a preview runs THEN it SHALL NOT write to S3 and SHALL NOT create a
   SageMaker job (local only, no cost).

### Requirement 4 — Approve and submit a batch job

**User story:** As a user, once the preview looks right, I want to run the exact
same pipeline over my whole dataset in S3 on SageMaker.

#### Acceptance Criteria

1. WHEN the user submits a batch job THEN the system SHALL create a SageMaker
   Processing Job that reads all images under the input S3 prefix, applies the
   pipeline, and writes processed outputs to the output S3 prefix.
2. WHEN submitting THEN the system SHALL upload the pipeline spec and the
   processing script to S3 and pass the spec to the job (via the code channel or
   an environment variable), so the batch job runs the IDENTICAL pipeline that was
   previewed.
3. WHEN a batch job is submitted THEN it SHALL be recorded in a JSON job store
   (`data/preprocessing_jobs.json`) with input/output URIs, the pipeline spec,
   status, timestamps, and instance type — following the existing evaluation-jobs
   pattern.
4. WHEN dataset-prep steps are enabled THEN the batch job SHALL, after processing,
   write a train/val/test split and a Decathlon-style `dataset.json` under the
   output prefix.
5. IF submission fails THEN the system SHALL clean up any partially uploaded
   artifacts and report the failing step.

### Requirement 5 — Track, list, and inspect batch jobs

**User story:** As a user, I want to see my preprocessing jobs and their status,
consistent with how training/evaluation jobs are tracked.

#### Acceptance Criteria

1. WHEN jobs are listed THEN the system SHALL return all preprocessing jobs, and
   SHALL refresh status from AWS for jobs still `InProgress`/`Stopping`.
2. WHEN a job completes THEN the record SHALL capture `completed_at` and the output
   S3 URI (and the `dataset.json` location if dataset-prep ran).
3. WHEN a job fails THEN the record SHALL capture the AWS `FailureReason`.
4. WHEN a user deletes a job record THEN the system SHALL remove it from the local
   store WITHOUT deleting the S3 outputs (matching evaluation behavior).
5. WHERE the user requests it THEN the system SHALL support syncing preprocessing
   jobs from AWS (`NameContains="preprocess-"`) into the local store.

### Requirement 6 — Processing image

**User story:** As a platform owner, I want a container image that can run the
preprocessing script at scale.

#### Acceptance Criteria

1. WHEN the batch script runs THEN it SHALL run in an image that includes MONAI,
   nibabel, pydicom, and the app's supported transform dependencies.
2. WHERE an existing MedAnvil image already provides MONAI + I/O deps (e.g. the
   MONAI bundle image) THEN the preprocessing job SHOULD reuse it rather than build
   a new image, to minimize maintenance.
3. WHEN built for SageMaker THEN any new image SHALL follow the existing build
   script conventions (Docker V2 manifest / BuildKit disabled).

### Requirement 7 — Security & safety

#### Acceptance Criteria

1. WHEN handling any user-supplied path (local preview input, temp dirs) THEN the
   system SHALL resolve and confine paths to allowed directories and guard against
   `../` traversal.
2. WHEN handling S3 URIs THEN the system SHALL validate them against the
   `s3://bucket/key` pattern before passing to boto3.
3. WHEN loading images THEN the system SHALL use MONAI/nibabel/pydicom readers
   (no `pickle`/`torch.load` of untrusted objects); IF any checkpoint-like load is
   introduced THEN it SHALL use `weights_only=True` or a documented trust boundary.
4. WHEN returning errors to the client THEN messages SHALL NOT leak internal
   absolute paths, stack traces, or credentials.
5. WHEN rendering job data or step params in the UI THEN dynamic values SHALL be
   HTML-escaped before insertion into the DOM.
6. WHEN configuring the job THEN bucket names/account IDs SHALL come from
   runtime config/env, never hardcoded.

## Out of Scope (initial release)

- Registration / atlas-based spatial normalization and N4 bias correction (may be
  added to the catalog later).
- De-identification / PHI stripping (handled by the separate dicom-anon work).
- A visual drag-and-drop pipeline builder (initial UI is a form/JSON editor +
  preview; drag-and-drop can come later).
- Automatic quality/correctness validation of the processed medical images.
- Distributed multi-node processing (single-node processing instance initially;
  parallelism via `S3DataDistributionType=ShardedByS3Key` across instances is a
  possible later enhancement).
