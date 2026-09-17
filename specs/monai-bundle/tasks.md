# Implementation Plan — MONAI Bundle Support

Incremental, testable tasks. Each references the requirements it satisfies.

- [x] 1. Bundle introspection utility
  - `src/monai_bundle/bundle_utils.py`: `inspect_bundle()` + `BundleInfo`
    (locate `configs/`, parse `metadata.json`, classify train/inference configs,
    find checkpoints, default config pickers, `is_valid`, `bundle_base_dir`).
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Decathlon datalist generator
  - `src/monai_bundle/datalist_gen.py`: `build_datalist()` from `dataset.json`
    with train/val split + path normalization; CLI.
  - _Requirements: 3.3_

- [ ] 3. Unit tests for introspection + datalist
  - Tests over synthetic bundle dirs: valid, nested (`sub/configs`), invalid
    (no configs); config classification; checkpoint discovery; datalist split.
  - _Requirements: 1 (all), 3.3_

- [ ] 4. Training entry point
  - `src/monai_bundle/train.py`: stage bundle from `SM_CHANNEL_MODEL`, resolve
    training config (CLI arg or `default_train_config`), optional datalist
    injection, generic `config_overrides` (JSON) pass-through (epochs,
    label_mappings, ...), run `monai.bundle run`, collect checkpoint + result.json
    into `SM_MODEL_DIR`.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7_
  - _Unit test: override parsing; config resolution; missing-config error._

- [ ] 5. Packaging helper
  - `src/monai_bundle/package.py`: validate a local dir is a bundle, tar to
    `model.tar.gz` preserving layout, optional S3 upload, cleanup on failure;
    accept an existing S3 `model.tar.gz` unchanged.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 6.4_
  - _Unit test: manifest contents; invalid-bundle rejection; S3 URI validation._

- [ ] 6. Training/inference image
  - `Dockerfile.monai-bundle` (DLC base + MONAI 1.3.x + nibabel/fire), copy
    `src/monai_bundle/` to `/opt/program`, `SAGEMAKER_PROGRAM=train.py`,
    PYTHONPATH includes bundle dir; `src/monai_bundle/requirements.txt`.
  - Adapt `build_and_push.sh` pattern (BuildKit disabled / V2 manifest).
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 7. Inference handler
  - `src/monai_bundle/infer.py`: `model_fn`/`input_fn`/`predict_fn`/`output_fn`
    running the bundle's inference config; S3/local/base64 NIfTI I/O; pass-through
    inference params; disclaimer in output; safetensors-first / weights_only load.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 6.1_
  - _Unit test: I/O contract + disclaimer with stub bundle + mocked subprocess._

- [ ] 8. Security guards
  - Path-traversal guards + confined staging; `s3://bucket/key` validation;
    documented trust boundary for bundle-code execution.
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 9. VISTA3D validation (cost-gated, separate)
  - Package/point at NV-Segment-CT bundle; fine-tune on Task07_Pancreas with
    `config_overrides={"label_mappings": {"default": [[1,4],[2,24]]}}` and
    `--config_file=configs/train_continual_softmax.json`; short GPU run.
  - Confirm the generic path suffices; add a thin specialization ONLY if it does
    not (and revise the spec then).
  - _Requirements: validates 3 + 4 end-to-end. Requires explicit approval for the
    billable create-training-job call._

- [ ] 10. Verification pass
  - Run `pytest` + linters; clean temp files; confirm no secrets/bucket
    names/account IDs committed.
  - _Requirements: 6, verification standard_
