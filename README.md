# MedAnvil

```
                              ██████
                              ██████
              ████████████████████████
            ██                        █
           ██                          █
          █████████████████████████████████
          ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
                    ██████████
                    ██████████
                    ██████████
               ████████████████████
              ██████████████████████
```

**Train, deploy, and monitor medical imaging AI — from your browser.**

> **⚠️ Research Use Only** — This platform is intended for research and development of medical imaging AI models. Model outputs must not be used as the sole basis for clinical decisions without review by qualified healthcare professionals. Clinical deployment requires regulatory approval (e.g. FDA clearance), clinical validation studies, and human-in-the-loop oversight. See the [Responsible AI](#responsible-ai) section below.

---

## What is this?

MedAnvil is a web dashboard for the full medical imaging ML lifecycle on AWS. Point it at your data in S3, pick a model architecture, and go from raw DICOM/NIfTI files to a live SageMaker endpoint — with experiment tracking, fairness auditing, and model monitoring built in.

## Capabilities

| | |
|---|---|
| 🏋️ **Train** | Classification & segmentation on SageMaker (ResNet50, DenseNet121, EfficientNet, ViT, SegResNet, SwinUNETR) with MLflow tracking |
| 🚀 **Deploy** | One-click deployment to real-time or async SageMaker endpoints |
| 🔬 **Invoke** | Test endpoints from the UI — DICOM, NIfTI, S3 URIs, or local files |
| 📊 **Evaluate** | Run evaluation jobs against deployed endpoints with per-class metrics, confusion matrices, and ROC curves |
| 📈 **Monitor** | Data capture, data quality baselines, and drift detection schedules via SageMaker Model Monitor |
| ⚖️ **Fairness** | Scan DICOM metadata for demographic distributions and bias auditing |
| 📝 **Report** | Generate structured paper sections (abstract, methods, results, discussion) from completed jobs |
| 🔄 **Fine-tune** | Resume training from checkpoints with full lineage tracking |

## Quick Start

```bash
# Setup credentials
cp .env.example .env
cp .env.local.example .env.local   # add your AWS account ID, role ARN, ECR URIs

# Run (installs deps, starts app)
./start.sh
```

That's it. The start script handles virtual environment creation, dependency installation (uses [uv](https://github.com/astral-sh/uv) if available, falls back to pip), and launches the app. If `MLFLOW_TRACKING_URI` is set in `.env.local` (e.g. a SageMaker-managed MLflow ARN), it uses that — otherwise it starts a local MLflow server.

Open [http://localhost:8000](http://localhost:8000) 🎉

| Service | URL |
|---|---|
| Dashboard | http://localhost:8000 |
| MLflow | SageMaker Studio (or http://localhost:5000 if local) |

### Build & push SageMaker containers

```bash
./start.sh --build
```

This builds the BYOC training and inference Docker images and pushes them to ECR (requires `AWS_ACCOUNT_ID` in `.env.local`).

## Stack

- **Backend** — FastAPI · boto3 · PyTorch · MONAI · MLflow
- **Frontend** — Jinja2 templates · vanilla JS
- **Infra** — Docker · SageMaker · S3 · ECR

## Project Layout

```
app/                 → FastAPI routers + services
src/classification/  → SageMaker BYOC training (classification)
src/segmentation/    → SageMaker BYOC training (segmentation)
src/inference/       → SageMaker serving scripts
src/evaluation/      → Endpoint evaluation processing jobs
templates/           → Jinja2 HTML
static/              → CSS + JS
docs/                → Full documentation (Read the Docs format)
```

## Documentation

Full docs live in [`docs/`](docs/) — covers configuration, API reference, deployment guide, user guides, and development setup.

```bash
# Build locally
cd docs && make html
# → docs/_build/html/index.html
```

## Testing

```bash
# Unit tests
pytest

# With coverage
coverage run -m pytest && coverage report

# Integration tests (requires AWS credentials + live endpoints)
pytest tests/integration/ -v -s
```

## Responsible AI

This platform processes medical imaging data for AI model training and evaluation. Important considerations:

- **Not a medical device** — AI outputs are for research purposes only and must not be used as the sole basis for clinical decisions.
- **Human oversight required** — Any clinical deployment requires qualified healthcare professional review of all AI predictions.
- **Regulatory compliance** — Production medical AI systems require appropriate regulatory clearances (FDA, CE marking) and institutional review board approval.
- **Fairness and bias** — Use the built-in Fairness Analysis feature to audit demographic representation in training data. Monitor model performance across subgroups.
- **Validation** — Models must undergo clinical validation studies before use in patient care settings.
- **Data privacy** — Ensure compliance with HIPAA, GDPR, and applicable data protection regulations when handling patient data.

---

Built with [MONAI](https://monai.io/) · Powered by [AWS SageMaker](https://aws.amazon.com/sagemaker/)
