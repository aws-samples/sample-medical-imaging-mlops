"""MSD / Decathlon dataset.json resolver.

The dataset contract for preprocessing: every dataset ships a dataset.json at
its root that explicitly lists image (and, for segmentation, label) entries.
The preprocessing job reads it rather than guessing pairs from filenames.

Schema (MSD / Medical Segmentation Decathlon):

    {
      "name": "AMOS",
      "tensorImageSize": "3D",
      "labels": {"0": "background", "1": "spleen", ...},
      "training":   [ {"image": "./imagesTr/x.nii.gz", "label": "./labelsTr/x.nii.gz"}, ... ],
      "validation": [ {"image": "...", "label": "..."} , ... ],   # optional
      "test":       [ "./imagesTs/y.nii.gz", ... ]                # optional, image-only
    }

Paths are relative to the dataset.json location. Task type is DERIVED: if the
training entries carry a "label", the dataset is segmentation; otherwise it is
image-only (classification).

RESEARCH USE ONLY.
"""
from __future__ import annotations

import json
import os
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_S3_RE = re.compile(r"^s3://([a-z0-9][a-z0-9.\-]{1,61}[a-z0-9])/(.+)$")


class DatasetResolveError(Exception):
    """Raised with a client-safe message."""


class DatasetJsonMissingError(DatasetResolveError):
    """Raised specifically when dataset.json is absent at the dataset root.

    Carries a ready-to-use example so the caller/UI can tell the user how to
    create one.
    """

    def __init__(self, message: str | None = None):
        super().__init__(message or "dataset.json not found at the dataset root")


# A canonical MSD/Decathlon dataset.json the UI can show as a template.
EXAMPLE_DATASET_JSON: dict = {
    "name": "MyDataset",
    "description": "Short description of the dataset",
    "tensorImageSize": "3D",
    "modality": {"0": "CT"},
    "labels": {"0": "background", "1": "organ_of_interest"},
    "training": [
        {"image": "./imagesTr/case_0001.nii.gz", "label": "./labelsTr/case_0001.nii.gz"},
        {"image": "./imagesTr/case_0002.nii.gz", "label": "./labelsTr/case_0002.nii.gz"},
    ],
    "validation": [
        {"image": "./imagesTr/case_0003.nii.gz", "label": "./labelsTr/case_0003.nii.gz"},
    ],
    "test": ["./imagesTs/case_0100.nii.gz"],
}


def example_dataset_json_text() -> str:
    """Return the example dataset.json as pretty-printed text."""
    import json as _json
    return _json.dumps(EXAMPLE_DATASET_JSON, indent=2)


@dataclass
class ResolvedPair:
    image: str                    # normalized relative path (posix)
    label: Optional[str] = None   # None for image-only entries
    split: str = "training"       # training | validation | test


@dataclass
class ResolvedDataset:
    name: str
    is_segmentation: bool
    labels: Dict[str, str] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    pairs: List[ResolvedPair] = field(default_factory=list)
    source: str = "dataset_json"        # dataset_json | imagefolder
    classes: Dict[str, int] = field(default_factory=dict)   # class name -> image count (ImageFolder)

    def task_type(self) -> str:
        return "segmentation" if self.is_segmentation else "classification"

    def summary(self, sample_n: int = 5) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type(),
            "is_segmentation": self.is_segmentation,
            "source": self.source,
            "labels": self.labels,
            "classes": self.classes,
            "counts": self.counts,
            "sample_pairs": [
                {"image": p.image, "label": p.label, "split": p.split}
                for p in self.pairs[:sample_n]
            ],
            "total_pairs": len(self.pairs),
        }


# ── Parsing ────────────────────────────────────────────────────────────────────

def _norm_rel(path: str) -> str:
    """Normalize a relative path from dataset.json (strip leading ./, no abs)."""
    if not isinstance(path, str) or not path.strip():
        raise DatasetResolveError("dataset.json contains an empty or non-string path")
    p = path.strip().replace("\\", "/")
    if p.startswith("s3://") or p.startswith("/"):
        raise DatasetResolveError("dataset.json paths must be relative to the dataset root")
    p = posixpath.normpath(p)
    if p.startswith(".."):
        raise DatasetResolveError("dataset.json paths must not escape the dataset root")
    return p


def _entry_to_pair(entry: Any, split: str, require_label: Optional[bool]) -> ResolvedPair:
    """Convert one dataset.json entry to a ResolvedPair.

    An entry is either a string (image-only) or an object with "image" and
    optionally "label".
    """
    if isinstance(entry, str):
        return ResolvedPair(image=_norm_rel(entry), label=None, split=split)
    if isinstance(entry, dict):
        if "image" not in entry:
            raise DatasetResolveError(f"{split} entry is missing an 'image' field")
        label = entry.get("label")
        return ResolvedPair(
            image=_norm_rel(entry["image"]),
            label=_norm_rel(label) if label not in (None, "") else None,
            split=split,
        )
    raise DatasetResolveError(f"{split} entry must be a string or an object")


def parse_dataset_json(text: str) -> ResolvedDataset:
    """Parse and validate a dataset.json string into a ResolvedDataset.

    Task type is derived from whether training entries carry labels.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DatasetResolveError(f"dataset.json is not valid JSON: {e.msg}") from None
    if not isinstance(data, dict):
        raise DatasetResolveError("dataset.json must be a JSON object")

    training = data.get("training")
    if not isinstance(training, list) or not training:
        raise DatasetResolveError("dataset.json must contain a non-empty 'training' list")

    # Derive task type from the first training entry.
    first = training[0]
    is_seg = isinstance(first, dict) and bool(first.get("label"))

    pairs: List[ResolvedPair] = []
    for e in training:
        pair = _entry_to_pair(e, "training", require_label=is_seg)
        if is_seg and pair.label is None:
            raise DatasetResolveError(
                "Inconsistent dataset: some training entries have a 'label' and others do not"
            )
        pairs.append(pair)

    for split in ("validation", "test"):
        entries = data.get(split)
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise DatasetResolveError(f"'{split}' must be a list if present")
        for e in entries:
            # test is conventionally image-only; validation mirrors training.
            pairs.append(_entry_to_pair(e, split, require_label=False))

    counts = {
        "training": sum(1 for p in pairs if p.split == "training"),
        "validation": sum(1 for p in pairs if p.split == "validation"),
        "test": sum(1 for p in pairs if p.split == "test"),
    }

    labels = data.get("labels") or {}
    if labels and not isinstance(labels, dict):
        raise DatasetResolveError("'labels' must be an object mapping id -> name")

    return ResolvedDataset(
        name=str(data.get("name", "dataset")),
        is_segmentation=is_seg,
        labels={str(k): str(v) for k, v in labels.items()},
        counts=counts,
        pairs=pairs,
    )


# ── Local filesystem resolution ────────────────────────────────────────────────

def resolve_from_dir(dataset_dir: str, *, verify_exists: bool = True) -> ResolvedDataset:
    """Read dataset.json from a local directory and resolve it.

    If verify_exists, checks that referenced image/label files exist.
    """
    dj = os.path.join(dataset_dir, "dataset.json")
    if not os.path.isfile(dj):
        raise DatasetJsonMissingError()
    with open(dj, "r") as f:
        resolved = parse_dataset_json(f.read())

    if verify_exists:
        missing = []
        for p in resolved.pairs:
            if not os.path.isfile(os.path.join(dataset_dir, p.image)):
                missing.append(p.image)
            if p.label and not os.path.isfile(os.path.join(dataset_dir, p.label)):
                missing.append(p.label)
            if len(missing) >= 5:
                break
        if missing:
            raise DatasetResolveError(
                "dataset.json references files that do not exist (e.g. "
                + ", ".join(missing[:5]) + ")"
            )
    return resolved


# ── S3 resolution (reads ONLY dataset.json — cheap dry-run) ─────────────────────

def read_dataset_json_from_s3(input_s3_uri: str, s3_client=None) -> ResolvedDataset:
    """Fetch only <input_s3_uri>/dataset.json from S3 and resolve it.

    Does NOT list or verify the referenced files (that would be expensive);
    this is the cheap "resolve dataset" dry-run.
    """
    uri = input_s3_uri.rstrip("/") + "/dataset.json"
    m = _S3_RE.match(uri)
    if not m:
        raise DatasetResolveError("Invalid S3 URI (expected s3://bucket/prefix)")
    bucket, key = m.group(1), m.group(2)

    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        text = obj["Body"].read().decode("utf-8")
    except Exception as e:
        # Distinguish "no dataset.json" from other errors, without leaking internals.
        code = getattr(getattr(e, "response", {}), "get", lambda *_: None)("Error")
        if isinstance(code, dict) and code.get("Code") in ("NoSuchKey", "404", "NoSuchBucket"):
            raise DatasetJsonMissingError() from None
        raise DatasetResolveError("Could not read dataset.json from S3") from None

    return parse_dataset_json(text)


# ── ImageFolder classification discovery (no dataset.json) ──────────────────────
# Convention: class = immediate subdirectory name, containing image files.
#   dataset/<class>/<image files>
# This mirrors how the classification training derives classes from folder names.

_IMAGE_EXTS = (".nii", ".nii.gz", ".dcm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _is_image_key(key: str) -> bool:
    k = key.lower()
    return any(k.endswith(ext) for ext in _IMAGE_EXTS)


def discover_imagefolder_s3(input_s3_uri: str, s3_client=None, sample_per_class: int = 3) -> ResolvedDataset:
    """Discover a classification dataset laid out as <root>/<class>/<images>.

    Lists ONE level of prefixes under the input URI and counts image files per
    class folder. Returns a classification ResolvedDataset (source="imagefolder").

    Raises DatasetResolveError if no class folders with images are found.
    """
    root = input_s3_uri.rstrip("/") + "/"
    m = _S3_RE.match(root)
    if not m:
        raise DatasetResolveError("Invalid S3 URI (expected s3://bucket/prefix)")
    bucket, prefix = m.group(1), m.group(2)

    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3")

    paginator = s3_client.get_paginator("list_objects_v2")

    # 1) list immediate subfolders (class candidates) using Delimiter="/"
    class_prefixes: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            class_prefixes.append(cp["Prefix"])   # e.g. "root/pneumonia/"

    classes: Dict[str, int] = {}
    pairs: List[ResolvedPair] = []
    for cp in class_prefixes:
        class_name = cp[len(prefix):].strip("/")
        if not class_name:
            continue
        count = 0
        samples_taken = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=cp):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") or not _is_image_key(key):
                    continue
                count += 1
                if samples_taken < sample_per_class:
                    rel = key[len(prefix):]   # relative to dataset root
                    pairs.append(ResolvedPair(image=rel, label=None, split="training"))
                    samples_taken += 1
        if count > 0:
            classes[class_name] = count

    if not classes:
        raise DatasetResolveError(
            "No class subfolders with images found for ImageFolder classification"
        )

    total = sum(classes.values())
    name = prefix.rstrip("/").split("/")[-1] or "dataset"
    return ResolvedDataset(
        name=name,
        is_segmentation=False,
        labels={},
        counts={"training": total, "validation": 0, "test": 0},
        pairs=pairs,
        source="imagefolder",
        classes=classes,
    )
