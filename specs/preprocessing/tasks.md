# Implementation Plan — MONAI Preprocessing Pipeline

Incremental, testable tasks. Each references the requirements it satisfies.
Order is chosen so the shared builder and local preview (cheap, no AWS) land and
are verified before any SageMaker/cost-incurring work.

- [ ] 1. Pydantic pipeline models
  - `app/models/preprocessing.py`: `Step` (+ per-type param models as a
    discriminated union), `DatasetPrep`, `PipelineSpec`, `PreviewRequest`,
    `SubmitRequest`. Server-side validation + ordering rules (load first, save last).
  - _Requirements: 1.1, 1.2, 1.3, 7.6_

- [ ] 2. Shared pipeline builder (the heart)
  - `src/preprocessing/pipeline_builder.py`: `STEP_CATALOG` + `build_transforms(spec)`
    → `monai.transforms.Compose`; correct per-key interpolation (nearest for label);
    `get_catalog()` for the UI. Dependency-light so it imports in both app + container.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 1.4_

- [ ] 3. Unit tests for models + builder
  - Valid spec builds a Compose with expected transforms/order; invalid type and
    bad params rejected; ordering violation rejected; label keys get nearest mode;
    catalog lists all step types with param schemas. (No AWS, no GPU.)
  - _Requirements: 1 (all), 2 (all)_

- [ ] 4. Local single-image preview
  - `src/preprocessing/local_preview.py`: `preview(spec, input_ref)`; resolve
    local path or S3 URI (download to confined temp, `validate_path`, `finally`
    cleanup); run Compose on one sample; collect in/out shape, spacing, intensity
    range, per-step timing; render mid-axial PNG.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 7.1, 7.2, 7.3, 7.4_
  - _Unit test: preview over a synthetic NIfTI + a tiny DICOM series; error path
    returns step name without absolute paths; asserts no S3 write / no job._

- [ ] 5. Batch processing script
  - `src/preprocessing/preprocess.py`: CLI (`--input-s3-uri --output-s3-uri
    --spec-path --task-type`); enumerate inputs from `/opt/ml/processing/input`,
    build Compose from spec (reuse `pipeline_builder`), write processed `.nii.gz`
    to `/opt/ml/processing/output`; if `dataset_prep.enabled` write split +
    Decathlon `dataset.json`. `src/preprocessing/requirements.txt` (monai, nibabel,
    pydicom).
  - _Requirements: 4.1, 4.2, 4.4, 2.2, 2.3_
  - _Unit test: run main() against a local temp "input" dir with mocked S3; assert
    outputs + dataset.json split ratios/seed determinism._

- [ ] 6. Preprocessing service (JSON store + SM job)
  - `app/services/preprocessing_service.py` mirroring `EvaluationService`:
    `_load/_save` on `data/preprocessing_jobs.json`; `preview_local`; `submit_batch`
    (upload script + builder + spec.json to S3, `create_processing_job` with code
    channel + output config, persist record incl. spec); `get_status`, `list_jobs`,
    `delete_job`, `sync_from_aws(NameContains="preprocess-")`.
  - _Requirements: 4.2, 4.3, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 7.6_
  - _Unit test: submit_batch with a mocked sagemaker/s3 client asserts job config
    shape, spec upload, and persisted record; delete_job keeps S3._

- [ ] 7. Router + wiring
  - `app/routers/preprocessing_router.py` mirroring `evaluation_router.py`:
    `/catalog`, `/preview`, `/submit`, `/jobs`, `/jobs/{name}` (GET/DELETE),
    `/sync`; use shared `aws_service`; sanitized `HTTPException`s. Register in
    `main.py` under `/api/preprocessing`.
  - _Requirements: 3.*, 4.*, 5.*, 7.4_
  - _Unit test: FastAPI TestClient — catalog returns steps; preview validation
    error → 400; submit without AWS creds → 400; bad S3 URI → 400._

- [ ] 8. UI — Preprocessing tab
  - Preprocessing tab in `templates/index.html` + `static/js/preprocessing.js`:
    spec editor seeded from `/catalog`; Preview (stats table + slice image);
    "Approve & Run on Dataset" (enabled post-preview) → `/submit`; jobs table
    (Active/Completed/Failed) with delete + sync. All dynamic values HTML-escaped
    via the shared `escapeHtml` helper.
  - _Requirements: 3.1, 3.3, 4.1, 5.1, 7.5_

- [ ] 9. Execution image
  - Reuse the existing MONAI bundle image; ensure `pydicom` available (add to the
    job's `requirements.txt` install step). Add a dedicated `Dockerfile.preprocessing`
    ONLY if reuse is insufficient (revise this task then).
  - _Requirements: 6.1, 6.2, 6.3_

- [ ] 10. End-to-end batch validation (cost-gated, separate)
  - Preview a spec on one image; on approval submit a SMALL real Processing Job
    over a few images in S3; confirm processed outputs + `dataset.json` land at the
    output prefix and are usable as a training Data S3 URI.
  - _Requirements: validates 4 + 5 end-to-end. Requires explicit approval for the
    billable create-processing-job call._

- [ ] 11. Verification pass
  - Run `pytest` + linters; clean temp files; confirm no secrets/bucket
    names/account IDs committed; confirm research-use-only disclaimer present in
    the batch script output and the UI tab.
  - _Requirements: 7 (all), verification standard_

## Suggested TODO.md entry

Add under a new `## Preprocessing` section in `TODO.md`:

> - [ ] **MONAI preprocessing pipeline** — Compose a MONAI-based preprocessing
>   pipeline (format conversion, intensity, spatial, dataset prep), preview it on
>   one image locally (local path / S3 URI), then run the identical pipeline over a
>   full S3 dataset as a SageMaker Processing Job writing processed NIfTI +
>   `dataset.json` to S3. See `specs/preprocessing/`.
