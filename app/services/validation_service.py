"""
Pre-training validation service.
Checks AWS credentials, S3 data structure, IAM role, and ECR image accessibility.
"""
import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

REQUIRED_CLASSIFICATION_SPLITS = ["train", "val"]
OPTIONAL_SPLITS = ["test"]


def _parse_s3_uri(uri: str):
    """Return (bucket, prefix) from s3://bucket/prefix."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Not a valid S3 URI: {uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def _list_immediate_subdirs(s3_client, bucket: str, prefix: str) -> List[str]:
    """Return immediate subdirectory names under prefix (no recursion)."""
    normalized = prefix.rstrip("/") + "/" if prefix else ""
    paginator = s3_client.get_paginator("list_objects_v2")
    subdirs = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"][len(normalized):].rstrip("/")
            if name:
                subdirs.add(name)
    return sorted(subdirs)


def _count_objects(s3_client, bucket: str, prefix: str) -> int:
    """Count all objects under prefix."""
    normalized = prefix.rstrip("/") + "/" if prefix else ""
    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    total_size = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized):
        count += page.get("KeyCount", 0)
        for obj in page.get("Contents", []):
            total_size += obj.get("Size", 0)
    return count, total_size


def validate_training_data(
    s3_client,
    iam_client,
    data_s3_uri: str,
    task_type: str,
    role_arn: str,
    image_uri: str,
) -> Dict:
    """
    Run all pre-training validations and return a structured report.
    """
    checks = []
    passed = True

    # ── 1. AWS credentials ──────────────────────────────────────────────────
    try:
        s3_client.list_buckets()
        checks.append(_ok("AWS Credentials", "Credentials are valid and active."))
    except Exception as e:
        checks.append(_fail("AWS Credentials", f"Could not authenticate with AWS: {e}"))
        passed = False

    # ── 2. Parse S3 URI ──────────────────────────────────────────────────────
    try:
        bucket, prefix = _parse_s3_uri(data_s3_uri)
    except ValueError as e:
        checks.append(_fail("S3 URI Format", str(e)))
        return {"passed": False, "checks": checks, "statistics": {}}

    checks.append(_ok("S3 URI Format", f"Parsed as bucket={bucket}, prefix={prefix or '/'}"))

    # ── 3. Bucket accessibility ──────────────────────────────────────────────
    try:
        s3_client.head_bucket(Bucket=bucket)
        checks.append(_ok("S3 Bucket Access", f"Bucket '{bucket}' is accessible."))
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = "Bucket not found." if code == "404" else f"Access denied ({code})."
        checks.append(_fail("S3 Bucket Access", msg))
        passed = False
        return {"passed": False, "checks": checks, "statistics": {}}

    # ── 4. Data structure & statistics ──────────────────────────────────────
    statistics = {}
    top_level_dirs = _list_immediate_subdirs(s3_client, bucket, prefix)
    statistics["top_level_folders"] = top_level_dirs

    if task_type == "classification":
        split_check, split_stats, split_ok = _validate_classification_structure(
            s3_client, bucket, prefix, top_level_dirs
        )
        checks.extend(split_check)
        statistics.update(split_stats)
        if not split_ok:
            passed = False
    else:
        # Segmentation — just report what's there
        total_files, total_size = _count_objects(s3_client, bucket, prefix)
        statistics["total_files"] = total_files
        statistics["total_size_mb"] = round(total_size / 1024 / 1024, 2)
        checks.append(_ok("S3 Data Structure", f"Found {total_files} files ({statistics['total_size_mb']} MB) under {data_s3_uri}"))

    # ── 5. IAM Role ──────────────────────────────────────────────────────────
    role_check = _validate_iam_role(iam_client, role_arn)
    checks.append(role_check)
    if not role_check["passed"]:
        passed = False

    # ── 6. ECR Image URI format ──────────────────────────────────────────────
    ecr_check = _validate_image_uri(image_uri)
    checks.append(ecr_check)
    if not ecr_check["passed"]:
        passed = False

    return {"passed": passed, "checks": checks, "statistics": statistics}


def _validate_classification_structure(s3_client, bucket, prefix, top_level_dirs):
    checks = []
    stats = {}
    ok = True

    # Check required splits exist
    missing = [s for s in REQUIRED_CLASSIFICATION_SPLITS if s not in top_level_dirs]
    if missing:
        checks.append(_fail(
            "Split Folders",
            f"Missing required folders: {missing}. Found: {top_level_dirs}. "
            f"Expected at least: {REQUIRED_CLASSIFICATION_SPLITS}"
        ))
        ok = False
        return checks, stats, ok

    checks.append(_ok("Split Folders", f"Found splits: {top_level_dirs} (train and val required, test optional)"))

    # Collect classes per split
    split_classes = {}
    split_stats = {}
    for split in top_level_dirs:
        split_prefix = f"{prefix.rstrip('/')}/{split}" if prefix else split
        classes = _list_immediate_subdirs(s3_client, bucket, split_prefix)
        split_classes[split] = set(classes)

        class_counts = {}
        split_total = 0
        split_size = 0
        for cls in classes:
            cls_prefix = f"{split_prefix.rstrip('/')}/{cls}"
            count, size = _count_objects(s3_client, bucket, cls_prefix)
            class_counts[cls] = count
            split_total += count
            split_size += size

        split_stats[split] = {
            "classes": class_counts,
            "total_files": split_total,
            "total_size_mb": round(split_size / 1024 / 1024, 2),
            "num_classes": len(classes),
        }

    stats["splits"] = split_stats

    # Validate class consistency between train and val
    required_splits_present = [s for s in REQUIRED_CLASSIFICATION_SPLITS if s in split_classes]
    if len(required_splits_present) >= 2:
        train_classes = split_classes.get("train", set())
        val_classes = split_classes.get("val", set())

        if train_classes == val_classes:
            checks.append(_ok(
                "Class Consistency",
                f"train and val both contain {len(train_classes)} classes: {sorted(train_classes)}"
            ))
        else:
            only_train = train_classes - val_classes
            only_val = val_classes - train_classes
            msg = f"Class mismatch between train and val."
            if only_train:
                msg += f" Only in train: {sorted(only_train)}."
            if only_val:
                msg += f" Only in val: {sorted(only_val)}."
            checks.append(_fail("Class Consistency", msg))
            ok = False

    # Per-split summary checks
    for split, s in split_stats.items():
        if s["num_classes"] == 0:
            checks.append(_fail(f"{split} Classes", f"No class subfolders found under '{split}/'."))
            ok = False
        else:
            checks.append(_ok(
                f"{split} Data",
                f"{s['num_classes']} classes, {s['total_files']} files, {s['total_size_mb']} MB"
            ))

    return checks, stats, ok


def _validate_iam_role(iam_client, role_arn: str) -> Dict:
    arn_pattern = r"^arn:aws:iam::\d{12}:role/.+"
    if not re.match(arn_pattern, role_arn):
        return _fail("IAM Role ARN", f"Invalid ARN format: '{role_arn}'. Expected: arn:aws:iam::<account>:role/<name>")
    try:
        role_name = role_arn.split("/")[-1]
        iam_client.get_role(RoleName=role_name)
        return _ok("IAM Role", f"Role '{role_name}' exists and is accessible.")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "NoSuchEntity":
            return _fail("IAM Role", f"Role not found: '{role_arn}'")
        return _fail("IAM Role", f"Could not verify role ({code}): {e}")
    except Exception as e:
        return _fail("IAM Role", f"Could not verify role: {e}")


def _validate_image_uri(image_uri: str) -> Dict:
    # Basic ECR URI pattern: <account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
    ecr_pattern = r"^\d+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/.+"
    if re.match(ecr_pattern, image_uri):
        return _ok("Container Image URI", f"ECR URI format is valid: {image_uri}")
    if image_uri:
        return _ok("Container Image URI", f"Non-ECR image URI provided: {image_uri} (format not validated)")
    return _fail("Container Image URI", "Image URI is empty.")


def _ok(name: str, message: str) -> Dict:
    return {"name": name, "passed": True, "message": message}


def _fail(name: str, message: str) -> Dict:
    return {"name": name, "passed": False, "message": message}
