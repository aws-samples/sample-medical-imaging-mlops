# Report Generation Guide

> **⚠️ Research Use Only** — Models trained and deployed through this platform are for research and development purposes. Clinical deployment requires regulatory review, validation against clinical standards, and human oversight. AI predictions must not be used as the sole basis for medical decisions.

This guide covers generating structured academic paper sections from completed training and evaluation jobs.

## Overview

The Report Generation feature produces formatted sections for research papers and technical reports based on your experiment data. It pulls metrics, hyperparameters, and results from completed jobs and structures them into standard academic sections.

## Prerequisites

- At least one completed (non-archived) training job
- (Optional) Completed evaluation jobs with metrics
- (Optional) Fairness analysis results

## Generate a Report

### Step 1 — Open the Report tab

Navigate to `http://localhost:8000` and click the **Report** tab.

### Step 2 — Select jobs

The job list shows only eligible jobs:
- **Training jobs** — completed, non-archived
- **Evaluation jobs** — completed

Select one or more training jobs and evaluation jobs to include.

### Step 3 — Configure the report

| Field | Description | Example |
|---|---|---|
| **Title** | Study/paper title | `"Spine X-ray Classification with DenseNet121"` |
| **Sections** | Which sections to generate | `abstract`, `methods`, `results`, `discussion` |
| **Include Fairness** | Include fairness analysis | `true` / `false` |

### Step 4 — Generate

Click **Generate Report**. The service compiles data from the selected jobs and produces the requested sections.

### API

```bash
curl -X POST http://localhost:8000/api/report/generate \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Spine X-ray Classification with DenseNet121",
    "training_job_names": ["spine-densenet-v1", "spine-densenet-v2"],
    "evaluation_job_names": ["eval-spine-v2"],
    "sections": ["abstract", "methods", "results", "discussion"],
    "include_fairness": true
  }'
```

## Available Sections

### Abstract

A concise summary of the study including:
- Task description
- Model architecture and key hyperparameters
- Dataset summary
- Main results (accuracy, AUC)

### Methods

Detailed methodology including:
- Model architecture and configuration
- Training setup (epochs, batch size, learning rate, instance type)
- Data source and preprocessing
- Evaluation protocol
- Fairness analysis methodology (if `include_fairness: true`)

### Results

Quantitative results including:
- Training metrics (final train/val accuracy, loss)
- Evaluation metrics (accuracy, macro AUC, per-class F1)
- Comparison across multiple runs (if multiple jobs selected)
- Fairness findings (if included)

### Discussion

Interpretation of results including:
- Performance analysis
- Comparison between model variants (if multiple jobs)
- Limitations
- Fairness implications (if included)

## List Available Jobs

To see which jobs are eligible for report generation:

```bash
curl http://localhost:8000/api/report/jobs
```

Returns only:
- Completed, non-archived training jobs
- Completed evaluation jobs

## Tips

- **Select related jobs** — compare variants of the same model or different architectures on the same dataset
- **Include evaluation** — reports are richer when evaluation metrics are available
- **Enable fairness** — adds demographic distribution discussion to Methods and any performance disparities to Discussion
- **Use multiple training jobs** — the report can compare hyperparameter choices and their impact on performance
