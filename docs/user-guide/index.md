# MedML Training Dashboard — User Guide

```{toctree}
:maxdepth: 2
:caption: User Guide

training
deployment
inference
evaluation
monitoring
fairness
report-generation
```

> **⚠️ Medical AI Disclaimer** — Models trained and deployed through this platform are for research and development purposes. Clinical deployment requires regulatory review, validation against clinical standards, and human oversight. AI predictions must not be used as the sole basis for medical decisions.

This guide covers the end-to-end workflow for training, deploying, and monitoring medical image classification and segmentation models using the MedML Training Dashboard.

## Prerequisites

- Dashboard running at `http://localhost:8000` (see [Getting Started](../getting-started.md) for setup)
- AWS credentials configured (environment variables or entered via the dashboard)
- S3 bucket with training data uploaded
- SageMaker execution role ARN with appropriate permissions

## Sections

| Guide | Description |
|---|---|
| [Training](training.md) | Launch local or SageMaker training jobs for classification and segmentation models |
| [Deployment](deployment.md) | Deploy trained models to SageMaker real-time or async endpoints |
| [Inference](inference.md) | Invoke deployed endpoints with DICOM, NIfTI, or image files |
| [Evaluation](evaluation.md) | Run evaluation processing jobs against deployed endpoints |
| [Monitoring](monitoring.md) | Enable data capture, compute baselines, and schedule drift monitoring |
| [Fairness Analysis](fairness.md) | Scan DICOM metadata for demographic distributions and bias auditing |
| [Report Generation](report-generation.md) | Generate structured academic paper sections from completed jobs |

## Typical Workflow

1. **Connect to AWS** — Enter credentials or rely on environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`)
2. **Upload data to S3** — Organize training data in the expected folder structure
3. **Train a model** — Launch a SageMaker training job from the Training tab
4. **Deploy the model** — Deploy the resulting `model.tar.gz` to a SageMaker endpoint
5. **Test inference** — Invoke the endpoint with a sample image
6. **Evaluate** — Run a full evaluation job against a labeled test set
7. **Monitor** — Enable data capture and set up drift monitoring schedules
8. **Audit fairness** — Scan your DICOM dataset for demographic imbalances
9. **Generate report** — Produce paper sections summarizing your experiment
