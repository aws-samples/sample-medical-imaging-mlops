# MedAnvil v1.1.0

> **Research use only.** MedAnvil and the models it trains or deploys are not medical devices and must not be used for clinical diagnosis.

This release adds MONAI Model Zoo and NVIDIA NV-Segment-CT bundle support, a preprocessing pipeline and a prediction viewer. It also hardens the dashboard and the SageMaker containers.

**Rebuild and re-push all container images after upgrading.** See [Upgrade notes](#upgrade-notes).

Full changelog: [`CHANGELOG.md`](CHANGELOG.md). Changes since the initial public release: [`bdb2798...v1.1.0`](https://github.com/aws-samples/sample-medical-imaging-mlops/compare/bdb2798...v1.1.0)

## Highlights

- **MONAI Bundle tab.** Stage a bundle from its MONAI Model Zoo GitHub link, optionally fine-tune it on SageMaker, deploy it to an async endpoint and run inference that writes masks to S3. Validated bundles: spleen and whole-body CT segmentation, wholeBrainSeg (UNEST), breast density classification and pathology nuclei segmentation.
- **NVIDIA MedTech tab (NV-Segment-CT / VISTA-3D).** Fine-tuning uses the bundle's continual-learning recipe, and async inference accepts optional label and point prompts. Weights are fetched from Hugging Face at stage time and are never stored in git. CT only.
- **Preprocessing pipeline.** Build a pipeline from 24 deterministic MONAI transforms, preview it on one image, then run it over an S3 dataset as a CPU SageMaker Processing Job.
- **Prediction Viewer (experimental, `/viewer`).** NiiVue viewer for NIfTI and DICOM with multiplanar and 3D views, mask overlays, a per-label legend and `?image=&mask=` deep links. Volumes stream from S3 through the backend, so no bucket CORS setup is needed.
- **Multi-class segmentation training.** Configurable class count, patch ROI size and patches per volume. Inference matches the patch-based training.

## Security

- **XSS:** all dynamic HTML goes through an escape-by-default helper (`static/js/safe_html.js`), and every `innerHTML` and `document.write` use was replaced.
- **DNS rebinding and CSRF:** the dashboard checks the `Host` header and rejects cross-site write requests.
- **Inference handlers:** local file paths are accepted only under `LOCAL_INPUT_ROOTS`, `ALLOWED_INPUT_BUCKETS` can restrict readable buckets, and MONAI `$`, `@` and `%` config overrides are rejected.
- **Archives and bundle requirements:** tar extraction rejects path traversal and escaping links, and only plain package specifiers are installed.
- **S3 bucket ownership:** job code uploads check the bucket belongs to `AWS_ACCOUNT_ID` and use `ExpectedBucketOwner`.
- **Network exposure:** Docker Compose ports and the local MLflow server bind to `127.0.0.1` only.
- **Dependencies:** upgraded for security fixes (MONAI, MLflow, PyTorch, Pillow, Jinja2, scikit-learn). Dockerfile base images are pinned by digest.

## Fixed

- MONAI bundle staging and preprocessing preview could use a different AWS identity than the connected one, causing `AccessDenied`/`403`. They now use the dashboard's session.
- MONAI bundle and NV-Segment-CT inference requests now send the required `image` key. Previously they failed immediately while the UI kept polling.
- Segmentation inference inverts spatial preprocessing, so saved masks match the original image geometry.
- Batch preprocessing skips missing files instead of aborting.

## Upgrade notes

1. Rebuild and push every image with the `build_and_push*.sh` scripts (training, inference, MONAI bundle, NV-Segment-CT, preprocessing). MONAI moved from 1.3/1.4 to 1.5/1.6 and NV-Segment-CT moved off PyTorch 2.4, so re-validate fine-tuning and inference.
2. Re-install the dashboard dependencies: `pip install -r requirements.txt`.
3. New optional settings: `AWS_ACCOUNT_ID`, `REQUIRE_BUCKET_OWNER_CHECK`, `ALLOWED_HOSTS`, `VIEWER_MASKS_BUCKET` (dashboard), `LOCAL_INPUT_ROOTS` and `ALLOWED_INPUT_BUCKETS` (inference endpoints). See [Configuration](docs/configuration.md#dashboard-security-settings).
4. Inference requests that send a local `file_path` outside `LOCAL_INPUT_ROOTS` are now rejected. Send an S3 URI or the file bytes instead.

## Known limitations

- The SageMaker PyTorch inference images stay on MONAI 1.5.2. Newer MONAI advisories are fixed only in 1.6.x, which needs PyTorch ≥ 2.8.
- The dashboard has no built-in authentication. Run it on loopback only and use an SSH tunnel for remote access.
- The Prediction Viewer is experimental.
