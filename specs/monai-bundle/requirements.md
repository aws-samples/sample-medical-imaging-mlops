# Requirements — MONAI Bundle Support

## Overview

Add first-class support to MedAnvil for models packaged in the **MONAI Bundle**
format, so users can run **inference** and **fine-tuning** on any MONAI bundle
(e.g. NVIDIA NV-Segment-CT / NV-Segment-CTMR / VISTA3D, and other MONAI bundles)
on SageMaker, reusing MedAnvil's existing training and deployment infrastructure.

This feature is **model-agnostic**: it drives a bundle through the bundle's own
configs via `python -m monai.bundle run`, rather than hardcoding any single
model. VISTA3D is treated as a validation case for the generic path, not a
special code path (a thin specialization is added only if the generic path
proves insufficient).

> ⚠️ **RESEARCH USE ONLY.** Bundle outputs are AI predictions for research and
> development only, not clinical diagnoses. Clinical use requires regulatory
> clearance, clinical validation, qualified professional review, and
> fairness/bias evaluation across demographic subgroups.

## Background — the MONAI Bundle format

A MONAI Bundle is a self-describing directory:

```
<bundle>/
    configs/
        metadata.json      # task, network, inputs/outputs, version, spatial dims
        inference.json      # inference workflow (or .yaml)
        train.json          # training workflow (or .yaml)  [optional]
        *.json / *.yaml     # additional/overlay configs (e.g. multi_gpu, softmax)
    models/
        model.pt            # checkpoint(s)
    scripts/                # optional bundle-specific python
    docs/
```

All bundles share one CLI:
`python -m monai.bundle run --config_file=configs/<name>.json [--<override>=<value>]`.
This uniformity is what makes generic support possible.

## Definitions

- **Bundle** — a directory in MONAI Bundle format (has `configs/` + typically
  `models/` + `metadata.json`).
- **Bundle source** — where a bundle comes from: an S3 `model.tar.gz`, a local
  path, or (future) a downloaded snapshot.
- **Bundle inspection** — reading a bundle's `metadata.json` and discovering its
  config files and checkpoints to drive it generically.

## Requirements

### Requirement 1 — Inspect and validate a bundle

**User story:** As a user, I want MedAnvil to understand a MONAI bundle's
structure automatically, so that it can drive training/inference without me
hand-specifying internals.

#### Acceptance Criteria

1. WHEN given a bundle directory THEN the system SHALL locate `configs/` (directly
   or nested one level down) and read `metadata.json` if present.
2. WHEN inspecting a bundle THEN the system SHALL enumerate available config files
   and classify them as training (`train`/`finetune`/`continual`) or inference
   (`infer`/`eval`) candidates.
3. WHEN inspecting a bundle THEN the system SHALL locate checkpoint files under
   `models/` (or the bundle root) by extension (`.pt/.pth/.ts/.safetensors`).
4. IF no `configs/` directory or no config files are found THEN the system SHALL
   report the bundle as invalid with a descriptive message and SHALL NOT attempt
   to run it.

### Requirement 2 — Package/import a bundle into a SageMaker artifact

**User story:** As a user, I want to stage a bundle as a SageMaker `model.tar.gz`
in S3, so that it can feed the existing training and deployment flows.

#### Acceptance Criteria

1. WHEN a bundle source is a local directory THEN the system SHALL package it into
   a `model.tar.gz` preserving the bundle layout (`configs/`, `models/`,
   `scripts/`) and upload it to a configured S3 URI.
2. WHEN packaging THEN the system SHALL validate the archive contains a discoverable
   bundle (Requirement 1) before upload.
3. WHEN a bundle source is an existing S3 `model.tar.gz` THEN it SHALL be usable
   directly as the training `model` channel / deployment artifact without
   re-packaging.
4. IF packaging or upload fails THEN the system SHALL clean up partial local/S3
   artifacts and report the failing step without leaking internal paths.

### Requirement 3 — Fine-tune a bundle on SageMaker

**User story:** As a researcher, I want to fine-tune a MONAI bundle on my dataset
using SageMaker, reusing MedAnvil's training service.

#### Acceptance Criteria

1. WHEN fine-tuning is started THEN the system SHALL launch a SageMaker training
   job via the EXISTING `SageMakerTrainingService`, staging the bundle as the
   `model` channel and the dataset as the `data` channel.
2. WHEN a training config name is provided THEN the entry point SHALL run
   `python -m monai.bundle run --config_file=configs/<name>` from the bundle
   directory; IF none is provided THEN it SHALL use the bundle's default training
   config (Requirement 1.2) and fail clearly if none exists.
3. WHEN the dataset follows the Decathlon layout THEN the system SHALL be able to
   generate a `training`/`validation` datalist from `dataset.json` and inject the
   datalist path, `dataset_dir`, and any config overrides into the chosen config.
4. WHERE a bundle requires task-specific config keys (e.g. `label_mappings`,
   `epochs`) THEN the entry point SHALL accept those as pass-through overrides
   without hardcoding any single bundle's schema.
5. WHEN the training config requires a GPU THEN deployment/launch SHALL use a GPU
   instance type; MONAI 1.3.x SHALL be available in the training image.
6. WHEN a fine-tuning job runs THEN it SHALL be tracked in the existing jobs store,
   report status/cost via the existing flow, and produce a training report.
7. WHEN training completes THEN the produced checkpoint and a `result.json`
   (recording config, dataset, overrides, and the research-use-only disclaimer)
   SHALL be written to `SM_MODEL_DIR` for bundling into `model.tar.gz`.

### Requirement 4 — Run inference with a bundle

**User story:** As a user, I want to deploy a MONAI bundle to a SageMaker endpoint
and run predictions, reusing MedAnvil's deployment flow.

#### Acceptance Criteria

1. WHEN a bundle artifact is deployed THEN the inference handler SHALL implement
   `model_fn`/`input_fn`/`predict_fn`/`output_fn` consistent with MedAnvil's
   existing SageMaker handler conventions.
2. WHEN a request is received THEN the handler SHALL run the bundle's inference
   config via `monai.bundle run`, using the packaged checkpoint (or an override
   `checkpoint_path`).
3. WHERE the task is 3D segmentation THEN the handler SHALL accept the established
   input formats (S3 URI, local path, or base64 NIfTI) and return the mask in the
   established NIfTI output format.
4. WHEN a prediction is returned THEN the response SHALL include the research-use-only
   disclaimer metadata.
5. IF the bundle exposes optional inference parameters (e.g. prompts) THEN the
   handler SHALL pass supported parameters through as config overrides, ignoring
   unsupported ones for bundles that do not use them.

### Requirement 5 — Training/inference image

**User story:** As a platform owner, I want a container image that can run any
supported MONAI bundle.

#### Acceptance Criteria

1. WHEN the image is built THEN it SHALL be based on MedAnvil's existing SageMaker
   Deep Learning Container base and SHALL include MONAI 1.3.x plus common bundle
   dependencies (nibabel, fire).
2. WHEN the image runs a bundle THEN the bundle's own `scripts/` and configs staged
   at runtime SHALL be importable/runnable (PYTHONPATH includes the bundle dir).
3. WHEN built for SageMaker THEN the image SHALL use the Docker V2 manifest format
   (BuildKit disabled), consistent with the existing build scripts.

### Requirement 6 — Security & safety

#### Acceptance Criteria

1. WHEN loading checkpoints THEN the system SHALL prefer `safetensors`, and load
   `.pt/.pth` with `weights_only=True` where supported; otherwise it SHALL document
   the trust boundary.
2. WHERE running a bundle executes bundle-provided code (scripts/configs) THEN this
   trust SHALL be explicit and documented, and bundles SHALL only be sourced from
   S3 prefixes the pipeline controls or user-provided sources the user vouches for.
3. WHEN constructing paths from bundle contents or user input THEN the system SHALL
   guard against path traversal and confine extraction/staging to controlled dirs.
4. WHEN inputs/outputs are handled THEN S3 URIs SHALL be validated against the
   `s3://bucket/key` pattern before use.

## Out of Scope (initial release)

- Automatic correctness validation of medical outputs (handlers are structural,
  not clinically validated).
- Non-MONAI-bundle models (covered by other work).
- A model catalog UI (may be added later; initial release is the runtime + entry
  points + image).
- Multi-node distributed training (single-node multi-GPU only, if the bundle ships
  a multi-GPU overlay config).
