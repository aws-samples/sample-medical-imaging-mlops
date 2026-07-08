# Evaluation Guide

This guide covers running SageMaker Processing Jobs to evaluate deployed classification endpoints against labeled test sets.

## Prerequisites

- A deployed classification endpoint in `InService` status
- A labeled test dataset in S3 organized by class folders
- A SageMaker execution role with processing job permissions

## Test Data Format

Organize your test data in S3 with one subfolder per class:

```
s3://your-bucket/test-data/
├── class_a/
│   ├── image001.dcm
│   ├── image002.dcm
│   └── image003.dcm
├── class_b/
│   ├── image004.dcm
│   └── image005.dcm
└── class_c/
    ├── image006.dcm
    └── image007.dcm
```

Each subfolder name is used as the ground-truth label.

## Submit an Evaluation Job

### Step 1 — Open the Evaluation tab

Navigate to `http://localhost:8000` and click the **Evaluation** tab.

### Step 2 — Fill out the evaluation form

| Field | Required | Description | Example |
|---|---|---|---|
| **Endpoint Name** | Yes | The deployed endpoint to evaluate | `spine-classifier-v2` |
| **Test Data S3 URI** | Yes | S3 path to the labeled test dataset | `s3://bucket/test-data/` |
| **Output S3 URI** | Yes | S3 prefix where results are written | `s3://bucket/eval-output/` |
| **Processing Image URI** | Yes | ECR URI of the processing container | Same as inference image |
| **Role ARN** | Yes | SageMaker execution role | from `.env.local` |
| **Instance Type** | No | Processing instance type | `ml.m5.xlarge` |

### Step 3 — Submit

Click **Run Evaluation**. A SageMaker Processing Job is launched that:
1. Discovers class folders under the test data URI
2. Invokes the endpoint for every file
3. Collects predictions and probability scores
4. Computes metrics and generates output artifacts

### Step 4 — Wait for completion

The job typically takes a few minutes depending on the dataset size. Status is polled from AWS.

## Output Artifacts

The evaluation job writes four files to the output S3 URI:

| File | Description |
|---|---|
| `metrics.json` | Accuracy, macro AUC, per-class precision/recall/F1, and support counts |
| `predictions.csv` | Per-file ground truth vs predicted class with per-class probabilities |
| `confusion_matrix.png` | Confusion matrix heatmap visualization |
| `roc_auc.png` | ROC-AUC curves (one per class, one-vs-rest) |

### `metrics.json` structure

```json
{
  "accuracy": 0.8750,
  "macro_auc": 0.9412,
  "per_class": {
    "class_a": {
      "precision": 0.90,
      "recall": 0.85,
      "f1": 0.87,
      "support": 20
    },
    "class_b": {
      "precision": 0.82,
      "recall": 0.88,
      "f1": 0.85,
      "support": 25
    }
  }
}
```

### `predictions.csv` structure

```text
file,true_class,predicted_class,prob_class_a,prob_class_b,prob_class_c
image001.dcm,class_a,class_a,0.89,0.07,0.04
image004.dcm,class_b,class_c,0.12,0.35,0.53
```

## Viewing Results

Once the evaluation job completes:

1. The results panel shows overall **accuracy** and **macro AUC**
2. Click on a job to view:
   - **Confusion matrix** — visual heatmap of predictions vs ground truth
   - **ROC-AUC curves** — per-class curves with AUC values
   - **Per-class metrics table** — precision, recall, F1, support for each class
3. Download individual artifacts for further analysis

### Endpoint Response Formats Handled

The evaluation script supports two endpoint response formats transparently:

| Format | Structure |
|---|---|
| **Batched arrays** | `{"predictions": [[p0, p1, ...]], "predicted_class": [idx], "confidence": [c]}` |
| **Named probabilities** | `{"probabilities": {"class_a": 0.3, ...}, "predicted_class": "class_a"}` |

## Sync from AWS

Click **Sync from AWS** (or `POST /api/evaluation/sync`) to pull evaluation processing jobs from AWS. Use `?name_contains=` to filter.

## Tips

- **Use a representative test set** — include edge cases and all class labels
- **Check class balance** — imbalanced test sets can produce misleading accuracy; look at per-class metrics
- **Compare across runs** — use the Report Generation feature to compile results from multiple jobs
- **Subgroup analysis** — combine with Fairness Analysis to check if performance varies across demographic groups
