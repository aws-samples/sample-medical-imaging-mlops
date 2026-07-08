# Fairness Analysis Guide

This guide covers scanning DICOM datasets for demographic distributions and identifying potential biases in your training data.

## Overview

The Fairness Analysis feature reads DICOM metadata from files in S3 and extracts demographic attributes to help you understand the composition of your dataset. Imbalanced demographic distributions can lead to biased model performance across patient subgroups.

## What It Extracts

The scanner reads the following DICOM tags from each file:

| DICOM Tag | Field | Example Values |
|---|---|---|
| `PatientSex` | Sex/Gender | `M`, `F`, `O` |
| `PatientAge` | Age | `045Y`, `6M`, `2W` |
| `EthnicGroup` | Ethnicity | Varies by institution |
| `PatientRace` | Race | Varies by institution |
| `Modality` | Imaging modality | `CT`, `MR`, `CR`, `DX` |
| `BodyPartExamined` | Body part | `CHEST`, `SPINE`, `HEAD` |

## Running a Fairness Scan

### Step 1 — Open the Fairness tab

Navigate to `http://localhost:8000` and click the **Fairness Analysis** tab.

### Step 2 — Configure the scan

| Field | Description | Default |
|---|---|---|
| **S3 Prefix** | S3 path containing DICOM files (scanned recursively) | — |
| **Max Files** | Maximum number of DICOM files to scan | `200` |

### Step 3 — Run the scan

Click **Scan**. The service:
1. Lists DICOM files under the S3 prefix
2. Downloads each file with `stop_before_pixels=True` (efficient — skips pixel data)
3. Extracts demographic tags from up to `max_files` files
4. Returns aggregated distributions

### API

```bash
curl -X POST http://localhost:8000/api/fairness/scan \
  -H "Content-Type: application/json" \
  -d '{
    "s3_prefix": "s3://my-bucket/training-data/",
    "max_files": 200
  }'
```

## Understanding the Results

### Per-Attribute Value Counts

Shows the distribution of each demographic attribute:

```
PatientSex:
  M: 120 (60%)
  F: 75 (37.5%)
  O: 5 (2.5%)

Modality:
  CR: 150 (75%)
  DX: 50 (25%)
```

### Age Bucketing

DICOM age strings (`045Y`, `6M`, `2W`) are normalized to integer years and bucketed:

| Bucket | Age Range |
|---|---|
| `0-17` | Pediatric |
| `18-29` | Young adult |
| `30-44` | Adult |
| `45-59` | Middle-aged |
| `60-74` | Senior |
| `75+` | Elderly |

Age format parsing:
- `045Y` → 45 years
- `6M` → 0 years (infant)
- `2W` → 0 years (neonate)

### Missing Rate Analysis

Shows the percentage of files where each tag was absent or empty:

```
Missing Rates:
  PatientSex: 2.5%
  PatientAge: 8.0%
  EthnicGroup: 65.0%
  PatientRace: 72.0%
  Modality: 0.0%
  BodyPartExamined: 1.5%
```

High missing rates for `EthnicGroup` and `PatientRace` are common — many institutions do not populate these fields.

### Sample Records

Up to 50 file-level rows showing per-file demographic values for manual inspection.

## Bias Auditing Workflow

### 1. Scan your training data

Run a fairness scan on the S3 prefix containing your training dataset.

### 2. Identify imbalances

Look for:
- **Severe skew** — e.g. 90% male / 10% female
- **Missing subgroups** — e.g. no pediatric patients
- **High missing rates** — e.g. race data missing for 70%+ of files

### 3. Cross-reference with evaluation

After running an evaluation job:
- Compare per-class performance metrics across demographic subgroups
- Check if the model underperforms on underrepresented groups

### 4. Take action

| Finding | Possible Action |
|---|---|
| Severe sex imbalance | Collect more data from underrepresented sex |
| Missing age groups | Augment dataset or document limitation |
| High missing rates | Contact data source about completeness |
| Performance disparity | Retrain with balanced sampling or stratified splits |

### 5. Include in reports

Set `include_fairness: true` when generating paper reports to automatically include the fairness analysis in the Methods and Discussion sections.

## Limitations

- Only scans DICOM files (NIfTI volumes do not contain demographic metadata)
- Relies on tags being populated by the source institution
- `EthnicGroup` and `PatientRace` are often empty in practice
- Max 200 files by default — increase `max_files` for larger datasets (trades off speed)
- Does not compute statistical significance — use as a screening tool
