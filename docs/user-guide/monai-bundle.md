# MONAI Bundle Support

Bring **any [MONAI Bundle](https://docs.monai.io/en/stable/bundle_intro.html)** into
MedAnvil to run fine-tuning and inference on AWS SageMaker — driven entirely from
the dashboard's **MONAI Bundle** tab.

!!! warning "Research Use Only"
    MONAI bundle outputs are AI predictions for research and development only, not
    clinical diagnoses. Clinical use requires regulatory clearance, clinical
    validation, qualified professional review, and fairness/bias evaluation.

---

## What is a MONAI Bundle?

A MONAI Bundle is a **self-describing model package** with a standard layout:

```
<bundle>/
    configs/
        metadata.json      # task, network, inputs/outputs, version
        inference.json      # inference workflow
        train.json          # training workflow (optional)
        *.json / *.yaml     # additional/overlay configs
    models/
        model.pt            # checkpoint(s)
    scripts/                # optional bundle-specific python
    docs/
```

Every bundle is driven by one uniform CLI:

```bash
python -m monai.bundle run --config_file=configs/<name>.json [--<override>=<value>]
```

This uniformity is what lets MedAnvil support **any** bundle with a single generic
runtime — the NVIDIA Model Zoo bundles (`spleen_ct_segmentation`,
`wholeBody_ct_segmentation`, VISTA3D / NV-Segment-CT, …) and your own bundles alike.

---

## Architecture

MedAnvil ships two things and reuses SageMaker for the rest:

| Component | What it is | Where it lives |
|---|---|---|
| **Training image** | `sm-monai-bundle` — MONAI 1.3.x runtime + generic `train.py` | ECR |
| **Inference image** | `sm-monai-bundle-inference` — MONAI runtime + serving handler (`inference.py`) | ECR |
| **Bundle artifact** | `model.tar.gz` = `configs/` + `models/` (+ `code/inference.py`) | S3 |

The **image is model-agnostic**; the **artifact is the model**. SageMaker downloads
the artifact to `/opt/ml/model` at runtime, and MedAnvil's handler drives the
bundle's own configs via `monai.bundle run`. The same image serves the pretrained
bundle and any fine-tuned version — only the artifact's S3 URI changes.

### The lifecycle

```
Stage  ──▶  Fine-tune (optional)  ──▶  Deploy (async)  ──▶  Inference
 │                │                        │                    │
 model.tar.gz     adapted model.tar.gz     SageMaker endpoint   NIfTI mask → S3
 in S3            in S3                     (GPU)
```

You can **deploy the pretrained bundle directly** (Stage → Deploy) or **fine-tune
first** (Stage → Fine-tune → Deploy). Both use the same inference image.

---

## Prerequisites

- The two images built and pushed to ECR (see [Building the images](#building-the-images)).
- AWS credentials configured; a SageMaker execution role ARN (`SAGEMAKER_ROLE_ARN`).
- Environment (in `.env.local`):
    - `AWS_ACCOUNT_ID`, `AWS_DEFAULT_REGION`
    - `SAGEMAKER_ROLE_ARN`
    - `MONAI_BUNDLE_IMAGE_URI` (training image) — optional; derived from account/region if unset
    - `MONAI_BUNDLE_INFERENCE_IMAGE_URI` (inference image) — optional; derived if unset
- A GPU instance quota for `ml.g5.xlarge` (or larger) in your region.

---

## Bringing a new MONAI bundle to MedAnvil

### 1. Register the bundle (optional, for the UI dropdown)

To make a bundle appear in the dashboard's **MONAI Model Bundle** category and the
deploy source selector, add an entry to `SUPPORTED_BUNDLES` in
`app/services/monai_bundle_service.py`:

```python
SUPPORTED_BUNDLES = [
    {
        "name": "my_bundle",
        "repo_id": "MONAI/my_bundle",          # HF/Model Zoo repo id
        "task": "segmentation",
        "dimension": "3D",
        "description": "…",
        "parameters": "…",
        "input_size": "3D CT (NIfTI)",
        "default_train_config": "train.json",
        "default_inference_config": "inference.json",
    },
    # …
]
```

Any bundle can still be staged by repo id even without registering it — the registry
only drives the convenience dropdowns.

### 2. Stage the bundle to S3

**MONAI Bundle tab → 1. Stage a bundle.** Enter:

- **MONAI bundle GitHub link** — the bundle's page in the MONAI model-zoo, e.g.
  `https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation`.
  A Hugging Face URL (`https://huggingface.co/MONAI/<name>`) or a plain repo id
  (`MONAI/<name>`) is also accepted.
- **Destination S3 URI** — e.g. `s3://your-bucket/monai-bundles/spleen/model.tar.gz`

Staging downloads the bundle (from the Hugging Face `MONAI/<name>` mirror), validates
it, injects the serving handler into `code/`, merges the bundle's declared
dependencies into `code/requirements.txt`, packages it as `model.tar.gz`, and uploads
it to S3. The staged artifact is then deployable and appears under **Deploy source →
Base bundle**.

The stage result and the **Staged bundles** list show information read **directly
from the bundle's own `configs/metadata.json`** — key papers (`references`), the data
organization (modality, class map, original data source), and a GitHub link back to
the bundle in the model-zoo. After staging you are prompted to **confirm** the papers
and data organization look correct before fine-tuning.

!!! note "Staging runs on the dashboard host"
    It downloads the bundle (tens of MB to a few GB) and uploads to S3, so the host
    needs outbound access to Hugging Face / the Model Zoo and write access to the
    bucket. Staged bundles are tracked in `data/monai_bundle_jobs.json` and persist
    across app restarts.

### 3. (Optional) Fine-tune

**MONAI Bundle tab → 2. Fine-tune.** Provide:

- **Bundle artifact** — the staged `model.tar.gz`
- **Dataset S3 URI** — a MONAI **Decathlon**-format dataset (see [Data layout](#dataset-layout))
- **Hyperparameters** — epochs, cache rate, learning rate, batch size, instance type
- **Training config** (optional) — defaults to the bundle's own (e.g. `train.json`)
- **Extra overrides** (JSON) — any bundle config key, e.g. VISTA3D softmax
  `{"label_mappings": {"default": [[1, 4], [2, 24]]}}`

This launches a SageMaker training job (reusing MedAnvil's training service). The
output is a new, deployable `model.tar.gz` with the fine-tuned weights, tracked
under **Jobs** and selectable in Deploy under **Fine-tuned bundle**.

### 4. Deploy (async only)

**MONAI Bundle tab → 4. Deploy.** MONAI bundle inference is 3D/volumetric — slow and
large-payload — so deployment is **async-only**.

- **Deploy source** — **Base MONAI bundle** or **Fine-tuned bundle**
- **Artifact** — auto-filled from the selection (or paste an S3 URI)
- **Endpoint name**, **Async output S3 location** (required), instance type/count, max concurrent

The async endpoint runs the inference DLC + the bundle handler. Deploy takes a few
minutes to reach **InService** — the deploy card **polls and updates the status live**
(Creating → InService / Failed). MONAI bundle endpoints also appear in a **Deployed
models** list on the tab, where you can scale to 0, scale up, add/remove instances,
refresh, or delete — and on the main Deployment page.

!!! note "Jobs categories"
    Fine-tune jobs are shown under **Active / Completed / Failed / Archived** tabs
    (matching the training page). Statuses are refreshed from AWS on load, so a
    finished job moves out of Active automatically.

### 5. Run inference

Submit an async request whose body references the input image and (optionally) the
output location:

```json
{
  "image": "s3://your-bucket/dataset/imagesTs/case_1.nii.gz",
  "output_s3_uri": "s3://your-bucket/masks/",
  "output_filename": "case_1_seg.nii.gz"
}
```

- `image` — S3 URI, local path, or base64 NIfTI.
- The handler runs the bundle's inference config on GPU, then **writes the
  segmentation mask as a `.nii.gz` to S3** and returns `{"mask_s3_uri": "…"}`.
- If no output location is set, it falls back to returning a base64 NIfTI.

For interactive bundles (e.g. VISTA3D), pass prompts too: `label_prompt`, `points`,
`point_labels` — the handler forwards supported keys; bundles that don't use them
ignore them.

---

## Dataset layout

Fine-tuning expects a MONAI **Decathlon** dataset at the `data` channel:

```
dataset_dir/
    dataset.json            # has a "training" list of {image, label} pairs
    imagesTr/*.nii.gz
    labelsTr/*.nii.gz
    imagesTs/*.nii.gz       # optional (test)
```

MedAnvil generates a train/validation split (`datalist_gen.py`) from `dataset.json`
and injects the datalist + `dataset_dir` into the chosen training config.

---

## Building the images

From the project root (reads `AWS_ACCOUNT_ID`/region from `.env.local`):

```bash
# Training image
bash build_and_push_monai_bundle.sh --no-cache

# Inference (serving) image
bash build_and_push_monai_bundle_inference.sh --no-cache
```

!!! warning "Docker V2 manifest required"
    SageMaker rejects OCI image manifests. The scripts set `DOCKER_BUILDKIT=0`. On
    Docker 24+/containerd image stores, disable the containerd snapshotter first so
    the build produces a Docker V2 manifest:
    ```bash
    echo '{ "features": { "containerd-snapshotter": false } }' | sudo tee /etc/docker/daemon.json
    sudo systemctl restart docker
    ```
    The scripts verify the manifest type after push and fail if it's OCI.

The large CUDA base layers can stall on proxied/limited networks — build and push
from an EC2 instance or CloudShell in-region if a laptop push hangs.

---

## Metrics & tracking

MONAI bundles track training metrics through **MONAI/Ignite handlers defined in the
bundle's own `train.json`**, not through MedAnvil. For example,
`spleen_ct_segmentation` uses:

- **`StatsHandler`** — logs per-iteration training loss and per-epoch validation
  metrics to **stdout** (visible in the job's **CloudWatch logs**, e.g.
  `Epoch: 1/2, Iter: 10/10 -- train_loss: 0.61` and `val_mean_dice: 0.04`).
- **`TensorBoardStatsHandler`** — writes the same scalars as **TensorBoard event
  files** into the output artifact.
- **`MeanDice`** as the key validation metric; **`CheckpointSaver`** saves the best
  model by it.

So out of the box you get: **training loss + validation Dice** in CloudWatch and as
TensorBoard files in the output `model.tar.gz`.

!!! note "MLflow is not wired into the MONAI bundle path (yet)"
    Unlike the built-in classification/segmentation training (which logs to MLflow
    via `MLFLOW_TRACKING_URI`), MONAI bundle fine-tuning does **not** log to an
    MLflow tracking server by default — the stock bundles use StatsHandler +
    TensorBoard. To add MLflow, inject an `MLFlowHandler` into the bundle's handler
    list via a `config_overrides` key (works only if the bundle's `train.json`
    exposes an injectable handlers list, e.g. `train#handlers`) and pass
    `MLFLOW_TRACKING_URI` into the job — best-effort per bundle, plus optionally a
    wrapper run that logs params + the final artifact regardless. Not implemented by
    default; treat metric tracking as CloudWatch + TensorBoard unless you add it.

## Dependencies & version conflicts

MedAnvil handles bundle dependencies in three tiers. The right tier is driven by
what the bundle declares in `metadata.json` (`required_packages_version`,
`monai_version`, `pytorch_version`, `numpy_version`), surfaced in the **stage
response** (`deps`).

### Tier 1 — pip-installable deps (default, automatic)

Staging writes the bundle's `required_packages_version` into `code/requirements.txt`
inside the `model.tar.gz`. The inference DLC pip-installs them **at endpoint
startup**. No image rebuild; dependencies are isolated per bundle. Trade-off:
slower cold start. This covers the large majority of bundles (extra pure-Python
packages and version overrides like `nibabel`, `tensorboard`, `scikit-image`,
`numpy<2`).

### Tier 2 — base-critical overrides (works, but verify)

If a bundle pins a version of a base-image-critical package (e.g. `numpy`) that
pip downgrades/upgrades at startup, it usually works but can perturb packages the
DLC's `torch` was built against. The stage report shows the bundle's expected
framework versions so you can sanity-check. If inference then errors on import,
treat it as Tier 3.

### Tier 3 — irreconcilable framework / driver conflict (dedicated image)

**`code/requirements.txt` (Tier 1) CANNOT fix a torch/CUDA/GPU-driver mismatch.**
`torch` is compiled against the image's CUDA runtime, and the GPU driver lives on
the SageMaker host — neither can be swapped by pip at startup.

**What happens if you ignore it:** the endpoint does **not** degrade gracefully —
it fails hard. Symptoms:

- `CannotStartContainerError` (container dies at import / health check), or
- inference 500s in the logs with `CUDA error: driver version is insufficient…`,
  `Attempting to deserialize object on a CUDA device…`, or torch `ImportError`/ABI
  errors after a pip override.

**The fix — build a dedicated inference image for that bundle** from a base whose
torch/CUDA matches what the bundle needs, and deploy only that bundle against it:

1. Copy `Dockerfile.monai-bundle-inference` to `Dockerfile.monai-bundle-inference-<bundle>`
   and change the `FROM` to a matching AWS PyTorch inference DLC (e.g. if the
   bundle needs torch 2.4 / CUDA 12.4, use `pytorch-inference:2.4...cu124` instead
   of the default 2.6 image). Keep the rest identical (handler at `/opt/program`,
   `SAGEMAKER_PROGRAM=inference.py`).
2. Build & push it to a distinct repo/tag, e.g. `sm-monai-bundle-inference-<bundle>`
   (mirror `build_and_push_monai_bundle_inference.sh`).
3. Deploy that bundle's endpoint with a **per-bundle image override** — pass
   `image_uri` in the deploy request (or set `MONAI_BUNDLE_INFERENCE_IMAGE_URI`).
   Every other bundle keeps using the shared image.

**How to know before deploying:** the stage `deps` report lists the bundle's
`pytorch_version` / `monai_version` / `numpy_version`. If those diverge materially
from the shared image's baked versions, plan for a dedicated image (Tier 3) rather
than relying on Tier 1. The report is advisory — minor version drift is usually
fine (e.g. a bundle declaring torch 2.4 runs fine on the 2.6 image); a major
torch/CUDA gap is the red flag.

## API reference

All endpoints are under `/api/monai-bundle`:

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/supported` | List registered bundles (the "MONAI Model Bundle" category) |
| `POST` | `/stage` | Download a Model Zoo bundle, package + upload to S3 |
| `POST` | `/finetune` | Launch a fine-tuning job (hyperparameters + overrides) |
| `GET`  | `/jobs` / `/jobs/{name}` | List / get fine-tune job status |
| `GET`  | `/artifacts` | Deployable artifacts (base staged + fine-tuned) for the deploy page |
| `POST` | `/deploy` | Deploy an artifact to an **async** endpoint |

---

## Supported / not supported

**Supported:** any MONAI bundle whose inference/training runs via `monai.bundle run`
and whose dependencies are covered by the images (MONAI 1.3.x, `pytorch-ignite`,
`nibabel`). 3D CT/MR segmentation bundles, VISTA3D/NV-Segment-CT family, spleen /
whole-body Model Zoo bundles.

**Needs extra work:** bundles requiring libraries not in the images. Staging
auto-captures a bundle's `required_packages_version` (from `metadata.json`) into
`code/requirements.txt` inside the artifact, which the inference DLC pip-installs
at **endpoint startup** — so most dependency mismatches are handled automatically
(at the cost of slower cold start), with no image rebuild. The stage response
reports the extra packages and the bundle's expected framework versions. A hard
framework/native conflict (e.g. a torch/CUDA the DLC can't provide, or system
libraries) can't be resolved by pip at startup — that rare case needs a dedicated
inference image from a compatible base, deployed via a per-bundle `image_uri`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CannotStartContainerError`, no logs | Inference image not derived from the AWS PyTorch **inference** DLC, or entrypoint/init bypassed. Rebuild `Dockerfile.monai-bundle-inference` (based on the inference DLC). |
| `Exactly one .pth or .pt file is required` | The DLC's **default** handler ran — the handler wasn't found. Ensure `code/inference.py` is inside the `model.tar.gz` (staging/fine-tune inject it). |
| `Attempting to deserialize on a CUDA device … is False` | Running on a CPU host. On a GPU endpoint this doesn't occur; the handler passes `--device` matching the host. |
| `CUDA error: driver version is insufficient` / torch ABI or ImportError after startup | The bundle's required torch/CUDA doesn't match the base image and pip can't reconcile it. Build a **dedicated inference image** from a matching DLC base and deploy that bundle via a per-bundle `image_uri` — see [Dependencies & version conflicts → Tier 3](#tier-3-irreconcilable-framework-driver-conflict-dedicated-image). |
| OOM during "Loading dataset" (fine-tune) | The bundle caches the dataset in RAM (`cache_rate: 1.0`). Use a larger instance (`ml.g5.4xlarge`) or set `cache_rate` lower via a hyperparameter/override. |
| `ModuleNotFoundError: ignite...` | Missing `pytorch-ignite` — it's in `requirements-inference.txt`/training reqs; rebuild the image. |
| OCI manifest rejected by SageMaker | Disable the containerd image store and rebuild `--no-cache` (see [Building the images](#building-the-images)). |
| Async output has base64 instead of a file | Set `OUTPUT_S3_URI` on the endpoint (deploy page's output location) or pass `output_s3_uri` in the request. |

---

## Related

- [Deployment](deployment.md) — general SageMaker endpoint concepts
- [Training](training.md) — the built-in classification/segmentation training flow
- [Custom Model Workflow](../custom-model-workflow.md) — bring your own (non-bundle) architecture
