"""
SageMaker Processing Job script for classification endpoint evaluation.

Expected S3 input structure (mounted at /opt/ml/processing/input):
    <class_name>/
        image1.dcm
        image2.dcm
        ...

Outputs written to /opt/ml/processing/output:
    metrics.json          — accuracy, per-class precision/recall/F1, macro AUC
    confusion_matrix.png  — confusion matrix heatmap
    roc_auc.png           — ROC-AUC curves (one per class, OvR)
    predictions.csv       — per-file ground truth vs predicted class + probabilities
"""

import argparse
import json
import logging
import os
import csv
from pathlib import Path

import boto3
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR  = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")


# ── Endpoint invocation ───────────────────────────────────────────────────────

def invoke_endpoint(runtime_client, endpoint_name: str, s3_uri: str) -> dict:
    """Send a single S3 URI to the endpoint and return parsed response."""
    payload  = json.dumps({"file_path": s3_uri})
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def parse_probabilities(result: dict, class_names: list) -> list:
    """Normalise endpoint response to a probability vector aligned with class_names."""
    # Format A: {"predictions": [[p0, p1, ...]], "predicted_class": [idx], "confidence": [c]}
    if "predictions" in result:
        probs = result["predictions"][0]
        return [float(p) for p in probs]

    # Format B: {"probabilities": {"class_a": 0.3, ...}, "predicted_class": "class_a", ...}
    if "probabilities" in result and isinstance(result["probabilities"], dict):
        prob_map = result["probabilities"]
        return [float(prob_map.get(c, 0.0)) for c in class_names]

    return []


def get_predicted_index(result: dict, class_names: list) -> int:
    """Return the predicted class index from the endpoint response."""
    if "predicted_class" in result:
        pc = result["predicted_class"]
        if isinstance(pc, list):
            pc = pc[0]
        if isinstance(pc, int):
            return pc
        # string label → index
        if pc in class_names:
            return class_names.index(pc)
    return -1


# ── Metrics & plots ───────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, class_names):
    """Compute accuracy, per-class P/R/F1, macro AUC."""
    from sklearn.metrics import (
        accuracy_score, classification_report,
        roc_auc_score, confusion_matrix,
    )
    import warnings

    n_classes = len(class_names)
    acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(n_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Macro AUC (OvR) — requires probability scores
    auc = None
    if y_prob and len(y_prob) == len(y_true):
        prob_matrix = np.array(y_prob)
        if prob_matrix.shape[1] == n_classes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    auc = float(roc_auc_score(
                        y_true, prob_matrix,
                        multi_class="ovr", average="macro",
                        labels=list(range(n_classes)),
                    ))
                except Exception as e:
                    logger.warning("AUC computation failed: %s", e)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return acc, report, auc, cm


def plot_confusion_matrix(cm, class_names, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", output_path)


def plot_roc_auc(y_true, y_prob, class_names, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import warnings

    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))
    prob_mat  = np.array(y_prob)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if y_bin.shape[1] <= i or prob_mat.shape[1] <= i:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], prob_mat[:, i])
                roc_auc     = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=1.5,
                        label=f"{cls} (AUC = {roc_auc:.3f})")
            except Exception:
                pass

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC-AUC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC-AUC plot → %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name",  required=True)
    parser.add_argument("--data-s3-uri",    required=True,
                        help="S3 URI of the test dataset root (class subfolders)")
    parser.add_argument("--output-s3-uri",  required=True,
                        help="S3 URI prefix where results will be uploaded")
    parser.add_argument("--region",         default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s3_client      = boto3.client("s3", region_name=args.region)
    runtime_client = boto3.client("sagemaker-runtime", region_name=args.region)

    # ── Discover class folders from S3 ───────────────────────────────────────
    bucket, prefix = args.data_s3_uri.replace("s3://", "").split("/", 1)
    prefix = prefix.rstrip("/") + "/"

    paginator = s3_client.get_paginator("list_objects_v2")
    all_keys  = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        all_keys.extend(obj["Key"] for obj in page.get("Contents", []))

    # Group by immediate subfolder (= class name)
    class_files: dict[str, list[str]] = {}
    for key in all_keys:
        rel = key[len(prefix):]
        parts = rel.split("/")
        if len(parts) >= 2 and parts[-1]:          # skip folder markers
            cls = parts[0]
            class_files.setdefault(cls, []).append(key)

    if not class_files:
        raise RuntimeError(f"No class subfolders found under {args.data_s3_uri}")

    class_names = sorted(class_files.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    logger.info("Classes (%d): %s", len(class_names), class_names)

    # ── Run inference ─────────────────────────────────────────────────────────
    y_true, y_pred, y_prob = [], [], []
    rows = []   # for predictions.csv

    for cls, keys in class_files.items():
        true_idx = class_to_idx[cls]
        logger.info("Evaluating class '%s' (%d files)...", cls, len(keys))
        for key in keys:
            s3_uri = f"s3://{bucket}/{key}"
            try:
                result    = invoke_endpoint(runtime_client, args.endpoint_name, s3_uri)
                probs     = parse_probabilities(result, class_names)
                pred_idx  = get_predicted_index(result, class_names)

                y_true.append(true_idx)
                y_pred.append(pred_idx)
                if probs:
                    y_prob.append(probs)

                rows.append({
                    "file":            s3_uri,
                    "true_class":      cls,
                    "predicted_class": class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx),
                    "correct":         true_idx == pred_idx,
                    **{f"prob_{c}": round(p, 4) for c, p in zip(class_names, probs)},
                })
            except Exception as e:
                logger.warning("Failed on %s: %s", s3_uri, e)

    if not y_true:
        raise RuntimeError("No predictions collected — check endpoint and data URIs")

    logger.info("Total samples: %d", len(y_true))

    # ── Compute metrics ───────────────────────────────────────────────────────
    acc, report, macro_auc, cm = compute_metrics(
        y_true, y_pred, y_prob if y_prob else None, class_names
    )
    logger.info("Accuracy: %.4f  |  Macro AUC: %s", acc, f"{macro_auc:.4f}" if macro_auc else "N/A")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # metrics.json
    metrics = {
        "endpoint_name":  args.endpoint_name,
        "data_s3_uri":    args.data_s3_uri,
        "num_samples":    len(y_true),
        "accuracy":       round(acc, 4),
        "macro_auc":      round(macro_auc, 4) if macro_auc else None,
        "class_names":    class_names,
        "per_class":      {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1":        round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }
            for cls in class_names if cls in report
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # predictions.csv
    if rows:
        csv_path = OUTPUT_DIR / "predictions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # confusion_matrix.png
    plot_confusion_matrix(cm, class_names, OUTPUT_DIR / "confusion_matrix.png")

    # roc_auc.png
    if y_prob:
        plot_roc_auc(y_true, y_prob, class_names, OUTPUT_DIR / "roc_auc.png")
    else:
        logger.warning("No probability scores returned — skipping ROC-AUC plot")

    # ── Upload to S3 ──────────────────────────────────────────────────────────
    out_bucket, out_prefix = args.output_s3_uri.replace("s3://", "").split("/", 1)
    out_prefix = out_prefix.rstrip("/")
    for f in OUTPUT_DIR.iterdir():
        dest_key = f"{out_prefix}/{f.name}"
        s3_client.upload_file(str(f), out_bucket, dest_key)
        logger.info("Uploaded → s3://%s/%s", out_bucket, dest_key)

    logger.info("Evaluation complete. Results at %s", args.output_s3_uri)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
