"""
SageMaker Processing Job script for segmentation endpoint evaluation.

Expected S3 input structure (mounted at /opt/ml/processing/input):
    <subject_id>/
        img.nii.gz      — input volume
        mask.nii.gz     — ground truth binary mask

Outputs written to /opt/ml/processing/output:
    metrics.json          — mean Dice, mean IoU, per-subject scores
    dice_distribution.png — histogram of per-subject Dice scores
    predictions.csv       — per-subject ground truth vs predicted Dice/IoU
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
from pathlib import Path

import boto3
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/opt/ml/processing/output")


# ── Endpoint invocation ───────────────────────────────────────────────────────

def invoke_endpoint(runtime_client, endpoint_name: str, s3_uri: str, s3_client=None) -> dict:
    """Send a NIfTI S3 URI to the segmentation endpoint and return parsed response.
    Supports both real-time and async endpoints."""
    import time

    payload = json.dumps({"s3_uri": s3_uri, "file_path": s3_uri})

    # Try real-time first
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
        return json.loads(response["Body"].read().decode("utf-8"))
    except runtime_client.exceptions.ClientError as e:
        if "does not support this inference type" not in str(e):
            raise

    # Async fallback — upload input to S3 and invoke async
    if not s3_client:
        s3_client = boto3.client("s3")

    # Upload input payload
    input_key = f"evaluation-tmp/{endpoint_name}/input_{int(time.time())}.json"
    input_bucket = s3_uri.replace("s3://", "").split("/")[0]
    s3_client.put_object(Bucket=input_bucket, Key=input_key, Body=payload.encode("utf-8"), ContentType="application/json")

    response = runtime_client.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=f"s3://{input_bucket}/{input_key}",
        ContentType="application/json",
        Accept="application/json",
    )

    output_location = response.get("OutputLocation", "")
    out_parts = output_location.replace("s3://", "").split("/", 1)
    out_bucket, out_key = out_parts[0], out_parts[1]

    # Poll for result (max 5 min)
    for _ in range(60):
        time.sleep(5)
        try:
            obj = s3_client.get_object(Bucket=out_bucket, Key=out_key)
            return json.loads(obj["Body"].read().decode("utf-8"))
        except s3_client.exceptions.NoSuchKey:
            continue

    raise TimeoutError(f"Async inference timed out for {s3_uri}")


# ── Metrics ───────────────────────────────────────────────────────────────────

def decode_mask(result: dict, s3_client=None) -> np.ndarray:
    """Decode segmentation mask from endpoint response.
    Supports both:
      - mask_nifti_b64: base64-encoded NIfTI (legacy)
      - mask_s3_uri: S3 path to .nii.gz file (current)
    """
    import base64, tempfile, nibabel as nib

    # New format: mask saved to S3
    mask_s3_uri = result.get("mask_s3_uri")
    if mask_s3_uri and s3_client:
        parts = mask_s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            f.flush()
            tmp_path = f.name
        s3_client.download_file(bucket, key, tmp_path)
        mask = nib.load(tmp_path).get_fdata().astype(np.uint8)
        os.unlink(tmp_path)
        return mask

    # Legacy format: base64-encoded mask in response
    b64 = result.get("mask_nifti_b64") or result.get("segmentation_mask_base64")
    if b64:
        raw = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            f.write(raw)
            f.flush()
            tmp_path = f.name
        mask = nib.load(tmp_path).get_fdata().astype(np.uint8)
        os.unlink(tmp_path)
        return mask

    raise ValueError("Response missing both 'mask_s3_uri' and 'mask_nifti_b64'")


def load_gt_mask(s3_client, bucket: str, key: str) -> np.ndarray:
    """Download ground truth NIfTI mask from S3 and return as binary numpy array."""
    import tempfile, nibabel as nib

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.flush()
        tmp_path = f.name
    s3_client.download_file(bucket, key, tmp_path)
    mask = nib.load(tmp_path).get_fdata().astype(np.uint8)
    os.unlink(tmp_path)
    return mask


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = (pred * gt).sum()
    union = ((pred + gt) > 0).sum()
    return float(intersection / union) if union > 0 else 1.0


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_dice_distribution(dice_scores: list, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dice_scores, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(np.mean(dice_scores), color="red", linestyle="--",
               label=f"Mean = {np.mean(dice_scores):.3f}")
    ax.set_xlabel("Dice Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Per-Subject Dice Score Distribution", fontsize=14)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved Dice distribution → %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name",  required=True)
    parser.add_argument("--data-s3-uri",    required=True,
                        help="S3 URI root — subject subfolders each containing img.nii.gz + mask.nii.gz")
    parser.add_argument("--output-s3-uri",  required=True)
    parser.add_argument("--img-filename",   default="img.nii.gz",
                        help="Filename of the input volume inside each subject folder")
    parser.add_argument("--mask-filename",  default="mask.nii.gz",
                        help="Filename of the ground truth mask inside each subject folder")
    parser.add_argument("--region",         default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    s3_client      = boto3.client("s3", region_name=args.region)
    runtime_client = boto3.client("sagemaker-runtime", region_name=args.region)

    # ── Discover subject folders from S3 ─────────────────────────────────────
    bucket, prefix = args.data_s3_uri.replace("s3://", "").split("/", 1)
    prefix = prefix.rstrip("/") + "/"

    paginator = s3_client.get_paginator("list_objects_v2")
    all_keys  = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        all_keys.extend(obj["Key"] for obj in page.get("Contents", []))

    # Group by subject subfolder
    subjects: dict[str, dict] = {}
    for key in all_keys:
        rel   = key[len(prefix):]
        parts = rel.split("/")
        if len(parts) >= 2 and parts[-1]:
            subj = parts[0]
            fname = parts[-1]
            subjects.setdefault(subj, {})[fname] = key

    # Keep only subjects that have both img and mask
    valid_subjects = {
        s: files for s, files in subjects.items()
        if args.img_filename in files and args.mask_filename in files
    }

    if not valid_subjects:
        raise RuntimeError(
            f"No valid subjects found under {args.data_s3_uri} "
            f"(expected subfolders with {args.img_filename} and {args.mask_filename})"
        )
    logger.info("Found %d subjects", len(valid_subjects))

    # ── Run inference ─────────────────────────────────────────────────────────
    dice_scores, iou_scores, rows = [], [], []

    for subj, files in valid_subjects.items():
        img_s3_uri  = f"s3://{bucket}/{files[args.img_filename]}"
        mask_key    = files[args.mask_filename]
        logger.info("Evaluating subject '%s'...", subj)
        try:
            result    = invoke_endpoint(runtime_client, args.endpoint_name, img_s3_uri)
            pred_mask = decode_mask(result, s3_client=s3_client)
            gt_mask   = load_gt_mask(s3_client, bucket, mask_key)

            # Resize pred to gt shape if needed
            if pred_mask.shape != gt_mask.shape:
                from skimage.transform import resize
                pred_mask = (resize(pred_mask.astype(float), gt_mask.shape) > 0.5).astype(np.uint8)

            dice = dice_score(pred_mask, gt_mask)
            iou  = iou_score(pred_mask, gt_mask)
            dice_scores.append(dice)
            iou_scores.append(iou)
            rows.append({"subject": subj, "dice": round(dice, 4), "iou": round(iou, 4)})
        except Exception as e:
            logger.warning("Failed on subject %s: %s", subj, e)

    if not dice_scores:
        raise RuntimeError("No predictions collected — check endpoint and data URIs")

    # ── Compute summary metrics ───────────────────────────────────────────────
    metrics = {
        "endpoint_name": args.endpoint_name,
        "data_s3_uri":   args.data_s3_uri,
        "num_subjects":  len(dice_scores),
        "mean_dice":     round(float(np.mean(dice_scores)), 4),
        "std_dice":      round(float(np.std(dice_scores)), 4),
        "median_dice":   round(float(np.median(dice_scores)), 4),
        "mean_iou":      round(float(np.mean(iou_scores)), 4),
    }
    logger.info("Mean Dice: %(mean_dice).4f  |  Mean IoU: %(mean_iou).4f", metrics)

    # ── Save outputs ──────────────────────────────────────────────────────────
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if rows:
        csv_path = OUTPUT_DIR / "predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "dice", "iou"])
            writer.writeheader()
            writer.writerows(rows)

    plot_dice_distribution(dice_scores, OUTPUT_DIR / "dice_distribution.png")

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
