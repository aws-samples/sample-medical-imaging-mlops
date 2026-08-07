# MedAnvil

**Train, deploy, and monitor medical imaging AI — from your browser.**

!!! warning "Research Use Only"
    This platform is intended for research and development of medical imaging AI models. Model outputs must not be used as the sole basis for clinical decisions without review by qualified healthcare professionals. Clinical deployment requires regulatory approval (e.g. FDA clearance), clinical validation studies, and human-in-the-loop oversight.

---

## Capabilities

| Feature | Description |
|---|---|
| 🏋️ **Train** | Classification & segmentation on SageMaker (ResNet50, DenseNet121, EfficientNet, ViT, SegResNet, SwinUNETR) with MLflow tracking |
| 🚀 **Deploy** | One-click deployment to real-time or async SageMaker endpoints |
| 🔬 **Invoke** | Test endpoints from the UI — DICOM, NIfTI, S3 URIs, or local files |
| 📊 **Evaluate** | Run evaluation jobs against deployed endpoints with per-class metrics, confusion matrices, and ROC curves |
| 📈 **Monitor** | Data capture, data quality baselines, and drift detection schedules via SageMaker Model Monitor |
| ⚖️ **Fairness** | Scan DICOM metadata for demographic distributions and bias auditing |
| 📝 **Report** | Generate structured paper sections (abstract, methods, results, discussion) from completed jobs |
| 🔄 **Fine-tune** | Resume training from checkpoints with full lineage tracking |

---

## Documentation

### Getting Started

- [Getting Started](getting-started.md) — Installation, setup, and first run
- [Configuration](configuration.md) — Environment variables, AWS setup, and app settings

### User Guide

Step-by-step guides for every workflow:

- [User Guide Overview](user-guide/index.md)
- [Training](user-guide/training.md) — Launch local and SageMaker training jobs
- [Deployment](user-guide/deployment.md) — Deploy models to SageMaker endpoints
- [Inference](user-guide/inference.md) — Test endpoints with medical images
- [Evaluation](user-guide/evaluation.md) — Run evaluation jobs and view metrics
- [Monitoring](user-guide/monitoring.md) — Set up data capture and drift detection
- [Fairness Analysis](user-guide/fairness.md) — Audit demographic representation and bias
- [Report Generation](user-guide/report-generation.md) — Generate structured research reports

### Reference

- [Deployment Guide](deployment-guide.md) — Production deployment on AWS
- [Custom Model Workflow](custom-model-workflow.md) — Bring your own model architecture
- [API Reference](api-reference.md) — REST API endpoints and schemas
- [Development](development.md) — Contributing, testing, and project structure

---

## Tech Stack

- **Backend** — FastAPI · boto3 · PyTorch · MONAI · MLflow
- **Frontend** — Jinja2 templates · vanilla JS
- **Infra** — Docker · SageMaker · S3 · ECR

---

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
