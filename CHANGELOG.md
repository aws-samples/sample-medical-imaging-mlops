# Changelog

All notable changes to MedAnvil are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

> **Research use only.** MedAnvil and the models it trains or deploys are not medical devices and must not be used for clinical diagnosis.

## [1.1.0] - 2026-11-13

This release adds MONAI Model Zoo and NVIDIA NV-Segment-CT bundle support, a preprocessing pipeline and a prediction viewer. It also hardens the dashboard and the SageMaker containers. **Rebuild and re-push all container images after upgrading** (see [Upgrade notes](#upgrade-notes)).

### Added

- **MONAI Bundle integration** (new *MONAI Bundle* tab, `/api/monai-bundle`). You can stage a MONAI Model Zoo bundle from its GitHub link, optionally fine-tune it on SageMaker, deploy it to an async endpoint and run inference that writes masks to S3.
  - Staged-bundle cards show the bundle's papers, data layout and source link, all read from its `metadata.json`.
  - The training-data layout is derived from each bundle's own train config.
  - Jobs are grouped into Active, Completed, Failed and Archived tabs.
  - Deployment shows live status and lets you scale or delete endpoints.
  - Each bundle's required packages are installed when the endpoint starts.
  - Classification bundles, 2D (JPEG/PNG) inputs and multi-output masks are supported.
  - Staged bundles can be deleted, which also removes the S3 artifact.
  - New `Dockerfile.monai-bundle`, `Dockerfile.monai-bundle-inference` and build scripts.
  - Validated bundles: spleen and whole-body CT segmentation, wholeBrainSeg (UNEST), breast density classification, pathology nuclei segmentation.
- **NVIDIA NV-Segment-CT (VISTA-3D) integration** (new *NVIDIA MedTech* tab, `/api/nvmedtech`).
  - Fine-tuning uses the bundle's own continual-learning recipe.
  - Async inference accepts optional label and point prompts.
  - Weights are fetched from Hugging Face at stage time and are never stored in git.
  - Dedicated `Dockerfile.nvmedtech` and `Dockerfile.nvmedtech-inference` images.
- **Preprocessing pipeline** (new *Preprocessing* tab).
  - Build a pipeline from 24 deterministic MONAI transforms and preview it on one image locally.
  - Run the same pipeline over an S3 dataset as a CPU SageMaker Processing Job.
  - Uses MSD/Decathlon `dataset.json`, falling back to ImageFolder for classification.
  - New `Dockerfile.preprocessing`.
- **Prediction Viewer** (experimental, `/viewer`).
  - Full-page NiiVue viewer for NIfTI and DICOM, with multiplanar, axial and 3D views.
  - Overlay opacity control, a recent-masks sidebar and `?image=&mask=` deep links.
  - Segmentation legend showing per-label colours and volumes, show/hide toggles and a crosshair label readout.
  - Volumes stream from S3 through the backend, so no bucket CORS setup is needed. DICOM is converted to NIfTI on the server.
- **Multi-class segmentation training.**
  - Configurable number of classes (0 auto-derives it from the label volumes).
  - Configurable patch ROI size and patches per volume.
  - Inference matches the patch-based training.
- The sidebar shows the connected AWS account ID.
- Optional SageMaker execution role ARN field on the MONAI bundle deploy form.
- `CHANGELOG.md`.

### Changed

- **Dependency upgrades for security fixes:**
  - Dashboard (`requirements.txt`): MONAI 1.6.1, MLflow 3.17.2, PyTorch ≥ 2.13, Pillow 13.0.0, Jinja2 3.1.6, scikit-learn 1.5.2.
  - Training images: MONAI 1.6.1. torch/torchvision are no longer pinned in the image requirements; they come from the AWS Deep Learning Container base.
  - Inference images: MONAI 1.5.2 (see [Known limitations](#known-limitations)).
  - NV-Segment-CT images: MONAI 1.5.2 and MLflow 3.17.2. The base images move from PyTorch 2.4 to PyTorch 2.9 (training) and 2.6 (inference), because MONAI ≥ 1.5 requires torch ≥ 2.4.1.
  - When a MONAI bundle is staged, versions it pins below a patched release are raised to that release.
- The Dockerfile base images are pinned by digest. The SageMaker Dockerfiles declare a `HEALTHCHECK`: `/ping` for inference images, `NONE` for batch training and processing images.
- Confirmation prompts and alerts use in-page dialogs and toast notifications instead of `window.confirm()` / `window.alert()`.
- Polling loops in the evaluation and endpoint warm-up scripts use boto3 waiters.
- Segmentation inference inverts spatial preprocessing, so saved masks match the original image geometry.
- Internal rename: `BundleInfo.is_valid` → `BundleInfo.valid` and `ResolvedDataset.is_segmentation` → `ResolvedDataset.segmentation`. The JSON field `is_segmentation` returned by the API is unchanged.

### Security

- **Browser (XSS):** all dynamic HTML goes through a new escape-by-default helper (`static/js/safe_html.js`).
  - The `html` tagged template escapes every interpolated value.
  - `setHTML` / `appendHTML` parse markup with `DOMParser` into inert fragments.
  - `jsArg()` safely encodes values for inline handlers, and `safeUrl()` blocks `javascript:` and `data:` URLs.
  - Every `innerHTML` and `document.write` use in the dashboard has been replaced.
- **Request-origin protection** for the loopback-only dashboard:
  - A Host-header allowlist blocks DNS rebinding.
  - Cross-site and cross-origin `POST`/`PUT`/`PATCH`/`DELETE` requests are rejected, which blocks CSRF.
- **Inference handlers:**
  - Request-supplied local file paths are accepted only under `LOCAL_INPUT_ROOTS`.
  - `ALLOWED_INPUT_BUCKETS` can restrict which S3 buckets requests may read.
  - Output filename patterns accept only known placeholders.
  - Per-request checkpoints must be inside the bundle's `models/` directory.
- **MONAI CLI arguments:** per-request `config_overrides` accept only literal values on non-reserved keys. MONAI `$` (expression), `@` (reference) and `%` (macro) values are rejected, and prompts must be numeric.
- **Archives:** tar extraction rejects path traversal, links that escape the target and device files, and uses Python's `data` extraction filter.
- **Bundle requirements:** only plain package specifiers are installed. pip options such as `--index-url`, `-e` and `-r`, and direct URLs, are dropped.
- **S3 bucket ownership:** before uploading job code (processing and evaluation scripts, staged bundles), MedAnvil checks the bucket belongs to `AWS_ACCOUNT_ID`, and the upload itself uses `ExpectedBucketOwner`.
- **Evaluation cache:** job names are validated and their paths kept inside the results directory before files are listed or deleted. `task_type` is allowlisted.
- **Model import:** S3 bucket and key are validated against S3 naming rules.
- **Temporary files:** created privately with `mkstemp`, never at fixed `/tmp` paths.
- **Network exposure:** the Docker Compose ports and the local MLflow server bind to `127.0.0.1` only.
- **Documentation:** example credentials and account IDs are replaced with placeholders.

### Fixed

- MONAI bundle staging and preprocessing preview used a separate boto3 client that could resolve a different AWS identity than the connected one, causing `AccessDenied`/`403`. They now use the dashboard's session.
- MONAI bundle and NV-Segment-CT inference requests now send the required `image` key. Previously they failed immediately, and the UI kept polling.
- `train.py` for MONAI bundles installs the artifact's requirements at start and detects non-default run IDs.
- Batch preprocessing skips missing files instead of aborting.
- Static JavaScript is cache-busted on each restart.

### Upgrade notes

1. **Rebuild and push every image:** `build_and_push*.sh` for training, inference, MONAI bundle, NV-Segment-CT and preprocessing. Re-validate fine-tuning and inference: MONAI moved from 1.3/1.4 to 1.5/1.6, and NV-Segment-CT moved off PyTorch 2.4.
2. Re-install the dashboard dependencies: `pip install -r requirements.txt`.
3. New optional settings for `.env.local` or the endpoint environment:

   | Variable | Where | Purpose |
   |---|---|---|
   | `AWS_ACCOUNT_ID` | dashboard | Enables the S3 bucket-ownership check (recommended) |
   | `REQUIRE_BUCKET_OWNER_CHECK` | dashboard | `true` refuses uploads when `AWS_ACCOUNT_ID` is unset |
   | `ALLOWED_HOSTS` | dashboard | Extra Host names allowed, e.g. when using an SSH tunnel under another name (default `localhost,127.0.0.1,[::1]`) |
   | `VIEWER_MASKS_BUCKET` | dashboard | Bucket listed in the viewer's recent-masks sidebar |
   | `LOCAL_INPUT_ROOTS` | inference endpoints | Directories from which local input paths are accepted (default `/opt/ml/input/data,/opt/ml/processing/input`) |
   | `ALLOWED_INPUT_BUCKETS` | inference endpoints | Comma-separated S3 buckets requests may read (unset = any bucket the endpoint role can read) |

4. Inference requests that sent a local `file_path` outside `LOCAL_INPUT_ROOTS` are now rejected. Send an S3 URI or the file bytes instead.

### Known limitations

- The SageMaker PyTorch inference images stay on MONAI 1.5.2. It fixes the advisories addressed in this release, but newer MONAI advisories are fixed only in 1.6.x, which needs PyTorch ≥ 2.8. Upgrade the inference images when a PyTorch ≥ 2.8 SageMaker inference container is available.
- The dashboard has no built-in authentication. Run it on loopback only, and reach it from another machine through an SSH tunnel.
- The Prediction Viewer is experimental.

## [1.0.0] - 2026-08-06

Initial public release: S3 data browsing, fairness analysis, SageMaker BYOC classification and segmentation training with fine-tuning, real-time and async deployment, inference, evaluation, SageMaker Model Monitor, MLflow integration and MkDocs documentation.

[1.1.0]: https://github.com/aws-samples/sample-medical-imaging-mlops/compare/bdb2798...HEAD
[1.0.0]: https://github.com/aws-samples/sample-medical-imaging-mlops/tree/bdb2798
