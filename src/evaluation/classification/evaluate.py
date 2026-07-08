"""
SageMaker Processing Job script for classification endpoint evaluation.

Expected S3 input structure (mounted at /opt/ml/processing/input):
    <class_name>/
        image1.dcm
        image2.dcm
        ...

Outputs written to /opt/ml/processing/output:
    metrics.json          — accuracy, per-class precision/recall/F1, macro AUC
    subgroup_metrics.json — per-demographic-group accuracy/F1 (sex, age, ethnicity)
    confusion_matrix.png  — confusion matrix heatmap
    roc_auc.png           — ROC-AUC curves (one per class, OvR)
    subgroup_accuracy.png — bar chart of accuracy by demographic group
    predictions.csv       — per-file ground truth vs predicted class + probabilities + demographics
"""

# ── Medical AI Safety Notice ─────────────────────────────────────────────────
# This evaluation script assesses model performance for RESEARCH purposes only.
# Evaluation metrics do not constitute clinical validation. Production medical
# AI systems require regulatory clearance, clinical validation studies, and
# human-in-the-loop oversight before deployment in patient care settings.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import os
import csv
import io
import tempfile
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR  = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")

# DICOM demographic tags
DEMO_TAGS = {
    "PatientSex":  (0x0010, 0x0040),
    "PatientAge":  (0x0010, 0x1010),
    "EthnicGroup": (0x0010, 0x2160),
}

AGE_BUCKETS = [(0,17,"0-17"),(18,29,"18-29"),(30,44,"30-44"),
               (45,59,"45-59"),(60,74,"60-74"),(75,120,"75+")]


# ── DICOM demographics ────────────────────────────────────────────────────────

def _parse_age(age_str: str):
    if not age_str:
        return None
    s = age_str.strip().upper()
    try:
        if s.endswith("Y"): return int(s[:-1])
        if s.endswith("M"): return max(0, int(s[:-1]) // 12)
        if s.endswith("W"): return max(0, int(s[:-1]) // 52)
        if s.endswith("D"): return 0
        return int(s)
    except ValueError:
        return None


def _age_bucket(age):
    if age is None:
        return "Unknown"
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return "Unknown"


def extract_demographics(s3_client, bucket: str, key: str) -> dict:
    """Download DICOM header only and extract demographic tags. Returns {} on failure."""
    try:
        import pydicom
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read()
        ds  = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)

        def _tag(tag):
            try:
                v = ds[tag].value if tag in ds else None
                return str(v).strip() if v else "Unknown"
            except Exception:
                return "Unknown"

        sex       = _tag(DEMO_TAGS["PatientSex"]) or "Unknown"
        age_raw   = _tag(DEMO_TAGS["PatientAge"])
        ethnicity = _tag(DEMO_TAGS["EthnicGroup"]) or "Unknown"

        age_int    = _parse_age(age_raw if age_raw != "Unknown" else "")
        age_bucket = _age_bucket(age_int)

        return {"sex": sex, "age_bucket": age_bucket, "ethnicity": ethnicity}
    except Exception as e:
        logger.debug("Could not read DICOM header for %s: %s", key, e)
        return {"sex": "Unknown", "age_bucket": "Unknown", "ethnicity": "Unknown"}


# ── Endpoint invocation ───────────────────────────────────────────────────────

def invoke_endpoint(runtime_client, endpoint_name: str, s3_uri: str) -> dict:
    payload  = json.dumps({"file_path": s3_uri})
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def parse_probabilities(result: dict, class_names: list) -> list:
    if "predictions" in result:
        return [float(p) for p in result["predictions"][0]]
    if "probabilities" in result and isinstance(result["probabilities"], dict):
        return [float(result["probabilities"].get(c, 0.0)) for c in class_names]
    return []


def get_predicted_index(result: dict, class_names: list) -> int:
    if "predicted_class" in result:
        pc = result["predicted_class"]
        if isinstance(pc, list): pc = pc[0]
        if isinstance(pc, int): return pc
        if pc in class_names: return class_names.index(pc)
    return -1


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, class_names):
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
    import warnings

    n_classes = len(class_names)
    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=list(range(n_classes)),
        target_names=class_names, output_dict=True, zero_division=0,
    )

    auc = None
    if y_prob and len(y_prob) == len(y_true):
        prob_matrix = np.array(y_prob)
        if prob_matrix.shape[1] == n_classes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    auc = float(roc_auc_score(
                        y_true, prob_matrix, multi_class="ovr",
                        average="macro", labels=list(range(n_classes)),
                    ))
                except Exception as e:
                    logger.warning("AUC computation failed: %s", e)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return acc, report, auc, cm


def compute_subgroup_metrics(rows: list, class_names: list) -> dict:
    """
    For each demographic attribute (sex, age_bucket, ethnicity), compute
    accuracy and macro-F1 per group value. Only groups with >= 2 samples included.
    """
    from sklearn.metrics import accuracy_score, f1_score
    import warnings

    demo_attrs = ["sex", "age_bucket", "ethnicity"]
    subgroups  = {}

    for attr in demo_attrs:
        # Group row indices by attribute value
        groups: dict[str, list] = defaultdict(list)
        for row in rows:
            val = row.get(attr, "Unknown")
            groups[val].append(row)

        attr_results = {}
        for group_val, group_rows in groups.items():
            if group_val == "Unknown" or len(group_rows) < 2:
                continue
            gt   = [r["true_idx"]  for r in group_rows]
            pred = [r["pred_idx"]  for r in group_rows]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acc = float(accuracy_score(gt, pred))
                f1  = float(f1_score(gt, pred, average="macro",
                                     labels=list(range(len(class_names))),
                                     zero_division=0))
            attr_results[group_val] = {
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "n":        len(group_rows),
            }

        if attr_results:
            subgroups[attr] = attr_results

    return subgroups


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
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
    y_bin    = label_binarize(y_true, classes=list(range(n_classes)))
    prob_mat = np.array(y_prob)

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
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC-AUC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC-AUC plot → %s", output_path)


def plot_subgroup_accuracy(subgroups: dict, output_path: Path):
    """Bar chart of accuracy per group, one subplot per demographic attribute."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attrs = [a for a in subgroups if subgroups[a]]
    if not attrs:
        return

    fig, axes = plt.subplots(1, len(attrs),
                             figsize=(max(6, 5 * len(attrs)), 5),
                             squeeze=False)

    for ax, attr in zip(axes[0], attrs):
        data   = subgroups[attr]
        labels = list(data.keys())
        accs   = [data[l]["accuracy"] for l in labels]
        ns     = [data[l]["n"] for l in labels]
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

        bars = ax.bar(labels, accs, color=colors, edgecolor="white", width=0.6)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(attr.replace("_", " ").title(), fontsize=12)
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(y=sum(accs) / len(accs), color="gray", linestyle="--",
                   linewidth=1, label="Mean")
        ax.legend(fontsize=8)

        # Annotate bars with accuracy + n
        for bar, acc, n in zip(bars, accs, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{acc:.2f}\n(n={n})",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Accuracy by Demographic Group", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved subgroup accuracy plot → %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--data-s3-uri",   required=True)
    parser.add_argument("--output-s3-uri", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s3_client      = boto3.client("s3", region_name=args.region)
    runtime_client = boto3.client("sagemaker-runtime", region_name=args.region)

    # ── Discover class folders ────────────────────────────────────────────────
    bucket, prefix = args.data_s3_uri.replace("s3://", "").split("/", 1)
    prefix = prefix.rstrip("/") + "/"

    paginator = s3_client.get_paginator("list_objects_v2")
    all_keys  = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        all_keys.extend(obj["Key"] for obj in page.get("Contents", []))

    class_files: dict[str, list[str]] = {}
    for key in all_keys:
        rel   = key[len(prefix):]
        parts = rel.split("/")
        if len(parts) >= 2 and parts[-1]:
            cls = parts[0]
            class_files.setdefault(cls, []).append(key)

    if not class_files:
        raise RuntimeError(f"No class subfolders found under {args.data_s3_uri}")

    class_names  = sorted(class_files.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    logger.info("Classes (%d): %s", len(class_names), class_names)

    # ── Run inference + extract demographics ──────────────────────────────────
    y_true, y_pred, y_prob = [], [], []
    rows = []

    for cls, keys in class_files.items():
        true_idx = class_to_idx[cls]
        logger.info("Evaluating class '%s' (%d files)...", cls, len(keys))
        for key in keys:
            s3_uri = f"s3://{bucket}/{key}"
            try:
                result   = invoke_endpoint(runtime_client, args.endpoint_name, s3_uri)
                probs    = parse_probabilities(result, class_names)
                pred_idx = get_predicted_index(result, class_names)

                # Extract demographics from DICOM header (best-effort)
                is_dicom = key.lower().endswith((".dcm", ".dicom"))
                demo     = extract_demographics(s3_client, bucket, key) if is_dicom else \
                           {"sex": "Unknown", "age_bucket": "Unknown", "ethnicity": "Unknown"}

                y_true.append(true_idx)
                y_pred.append(pred_idx)
                if probs:
                    y_prob.append(probs)

                rows.append({
                    "file":            s3_uri,
                    "true_class":      cls,
                    "true_idx":        true_idx,
                    "predicted_class": class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx),
                    "pred_idx":        pred_idx,
                    "correct":         true_idx == pred_idx,
                    "sex":             demo["sex"],
                    "age_bucket":      demo["age_bucket"],
                    "ethnicity":       demo["ethnicity"],
                    **{f"prob_{c}": round(p, 4) for c, p in zip(class_names, probs)},
                })
            except Exception as e:
                logger.warning("Failed on %s: %s", s3_uri, e)

    if not y_true:
        raise RuntimeError("No predictions collected — check endpoint and data URIs")

    logger.info("Total samples: %d", len(y_true))

    # ── Overall metrics ───────────────────────────────────────────────────────
    acc, report, macro_auc, cm = compute_metrics(
        y_true, y_pred, y_prob if y_prob else None, class_names
    )
    logger.info("Accuracy: %.4f  |  Macro AUC: %s", acc, f"{macro_auc:.4f}" if macro_auc else "N/A")

    # ── Subgroup metrics ──────────────────────────────────────────────────────
    subgroups = compute_subgroup_metrics(rows, class_names)
    for attr, groups in subgroups.items():
        logger.info("Subgroup [%s]: %s", attr,
                    {g: f"acc={v['accuracy']} n={v['n']}" for g, v in groups.items()})

    # ── Save outputs ──────────────────────────────────────────────────────────
    metrics = {
        "endpoint_name": args.endpoint_name,
        "data_s3_uri":   args.data_s3_uri,
        "num_samples":   len(y_true),
        "accuracy":      round(acc, 4),
        "macro_auc":     round(macro_auc, 4) if macro_auc else None,
        "class_names":   class_names,
        "per_class": {
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
    (OUTPUT_DIR / "subgroup_metrics.json").write_text(json.dumps(subgroups, indent=2))

    # predictions.csv — include demographic columns
    if rows:
        # Remove internal idx fields before writing CSV
        csv_rows = [{k: v for k, v in r.items() if k not in ("true_idx", "pred_idx")}
                    for r in rows]
        with open(OUTPUT_DIR / "predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    plot_confusion_matrix(cm, class_names, OUTPUT_DIR / "confusion_matrix.png")

    if y_prob:
        plot_roc_auc(y_true, y_prob, class_names, OUTPUT_DIR / "roc_auc.png")
    else:
        logger.warning("No probability scores — skipping ROC-AUC plot")

    if subgroups:
        plot_subgroup_accuracy(subgroups, OUTPUT_DIR / "subgroup_accuracy.png")
    else:
        logger.info("No demographic tags found in DICOM files — skipping subgroup plot")

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
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
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
