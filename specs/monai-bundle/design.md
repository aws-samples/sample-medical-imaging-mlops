# Design — MONAI Bundle Support

## 1. Goal

Provide generic, model-agnostic **inference** and **fine-tuning** for MONAI
bundles on SageMaker, driven by each bundle's own configs via
`python -m monai.bundle run`. VISTA3D (NV-Segment-CT / CTMR) is the first
validation target, not a special code path.

## 2. Module layout

```
src/monai_bundle/
    __init__.py
    bundle_utils.py      # bundle introspection: configs, metadata, checkpoints
    datalist_gen.py      # Decathlon dataset.json -> train/val datalist
    train.py             # SageMaker entry point: run a bundle's training config
    infer.py             # SageMaker inference handler (model_fn/.../output_fn)
    package.py           # package a local bundle dir -> model.tar.gz (+ S3 upload)
    requirements.txt     # monai==1.3.*, nibabel, fire, ...
    README.md
Dockerfile.monai-bundle  # training/inference image (DLC base + MONAI 1.3.x)
```

Reuses existing MedAnvil infrastructure:
- `SageMakerTrainingService.start_training_job` (BYOC PyTorch estimator) for training.
- The deployment flow for inference endpoints.
- The `SM_CHANNEL_*` / `SM_MODEL_DIR` SageMaker container contract, matching
  `src/segmentation/train.py`.

## 3. Bundle introspection (`bundle_utils.py`) — implemented

`inspect_bundle(root) -> BundleInfo`:
- Locates `configs/` (directly or nested one level down, e.g. `NV-Segment-CT/configs`).
- Parses `metadata.json` (task, network, IO).
- Enumerates config files; classifies train (`train`/`finetune`/`continual`) vs
  inference (`infer`/`eval`) candidates.
- Locates checkpoints under `models/` by extension.
- `default_train_config()` / `default_inference_config()` pick sensible defaults.
- `is_valid` gates any run. `bundle_base_dir()` returns the dir to run from.

This is the abstraction that makes support generic: everything downstream asks
`BundleInfo` what to run rather than hardcoding filenames.

## 4. Training entry point (`train.py`)

SageMaker BYOC contract. Steps:
1. Stage the bundle from `SM_CHANNEL_MODEL` into a working dir; place the
   pretrained checkpoint at `models/model.pt` if the bundle's config expects it.
2. `inspect_bundle` to resolve the training config (CLI `--config_file` name, or
   `default_train_config()`).
3. If the dataset (`SM_CHANNEL_DATA`) is Decathlon-shaped, `build_datalist` and
   inject `data_list_file_path` + `dataset_dir` into the config.
4. Apply pass-through config overrides from hyperparameters (generic
   `--set key=value`), e.g. `epochs`, `label_mappings` — NOT hardcoded per model.
5. Run `python -m monai.bundle run --config_file=configs/<train> --dataset_dir=...`.
6. Copy the produced checkpoint + `result.json` into `SM_MODEL_DIR`.

Overrides are passed as a JSON hyperparameter (`config_overrides`) so any bundle's
config keys can be set without code changes.

## 5. Inference handler (`infer.py`)

Standard `model_fn`/`input_fn`/`predict_fn`/`output_fn`:
- `model_fn(model_dir)`: `inspect_bundle`, resolve inference config + checkpoint.
- `input_fn`: materialize input (S3 URI / local / base64 NIfTI) to a local file;
  collect optional pass-through inference params.
- `predict_fn`: `monai.bundle run --config_file=configs/<inference>
  --input_file=... --checkpoint_path=... --output_dir=...`.
- `output_fn`: read produced NIfTI, return base64 NIfTI + metadata + disclaimer.

Segmentation I/O mirrors the existing `src/inference/segmentation/inference.py`
formats so it plugs into the current inference UI.

## 6. Packaging (`package.py`)

- Validate a local dir is a bundle (`inspect_bundle().is_valid`).
- `tar czf model.tar.gz` preserving `configs/ models/ scripts/`.
- Optional S3 upload to a validated `s3://bucket/key`.
- On failure, clean up partial local/S3 artifacts.

An existing S3 `model.tar.gz` can be used directly as the training `model`
channel / deployment artifact with no repackaging.

## 7. Image (`Dockerfile.monai-bundle`)

- Base: the same AWS DLC as `Dockerfile.sagemaker`.
- Install `requirements.txt` (MONAI 1.3.x, nibabel, fire).
- Copy `src/monai_bundle/` to `/opt/program/`; `SAGEMAKER_PROGRAM=train.py`;
  `PYTHONPATH` includes `/opt/program` so a staged bundle's `scripts/` import.
- BuildKit disabled at build time (Docker V2 manifest) per existing build scripts.

## 8. Security (maps to Requirement 6)

- safetensors-first; `weights_only=True` for `.pt/.pth`; documented trust boundary
  where a bundle's load path executes bundle code.
- Bundles sourced from pipeline-controlled S3 prefixes or user-vouched sources.
- Path-traversal guards; confined staging dirs; `s3://bucket/key` validation.

## 9. VISTA3D as a validation case (not a code path)

VISTA3D fits the generic design as:
- training config: `configs/train_continual_softmax.json`
- override: `label_mappings` (e.g. pancreas→VISTA class 4, tumor→24) via
  `config_overrides`
- inference config: `configs/inference_softmax.json`
Only if a VISTA3D-specific need can't be expressed as a config/override do we add
a thin specialization — and the spec would be revised then.

## 10. Testing strategy

- Unit: `inspect_bundle` on synthetic bundle dirs (valid/nested/invalid); config
  classification; checkpoint discovery; `build_datalist` split; override parsing;
  S3 URI + path-traversal guards; package manifest contents.
- Handler unit: `infer.py` I/O contract with a stub bundle + mocked `monai.bundle`
  subprocess; disclaimer present.
- Integration: package a fixture bundle → (mocked) S3; a mocked training launch
  builds the correct `SageMakerTrainingConfig` and command.
- Follow existing `tests/unit` + `tests/integration` layout.

## 11. Phases

1. Introspection + datalist (done) + unit tests.
2. Training entry point + packaging + image.
3. Inference handler.
4. VISTA3D validation run (real GPU smoke test) — separate, cost-gated.
