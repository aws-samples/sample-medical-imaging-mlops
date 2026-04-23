"""Service for extracting demographic information from DICOM files in S3."""
import io
import logging
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# DICOM tags of interest for fairness analysis
DEMOGRAPHIC_TAGS = {
    "PatientSex":     (0x0010, 0x0040),
    "PatientAge":     (0x0010, 0x1010),
    "EthnicGroup":    (0x0010, 0x2160),
    "PatientRace":    (0x0010, 0x2110),  # not standard but sometimes present
    "Modality":       (0x0008, 0x0060),
    "BodyPartExamined": (0x0018, 0x0015),
}

AGE_BUCKETS = [
    (0,   17,  "0-17"),
    (18,  29,  "18-29"),
    (30,  44,  "30-44"),
    (45,  59,  "45-59"),
    (60,  74,  "60-74"),
    (75,  120, "75+"),
]


def _parse_age(age_str: str) -> Optional[int]:
    """Parse DICOM age string like '045Y', '6M', '2W' to years."""
    if not age_str:
        return None
    age_str = age_str.strip().upper()
    try:
        if age_str.endswith("Y"):
            return int(age_str[:-1])
        if age_str.endswith("M"):
            return max(0, int(age_str[:-1]) // 12)
        if age_str.endswith("W"):
            return max(0, int(age_str[:-1]) // 52)
        if age_str.endswith("D"):
            return 0
        return int(age_str)
    except ValueError:
        return None


def _age_bucket(age: Optional[int]) -> str:
    if age is None:
        return "Unknown"
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return "Unknown"


def _extract_tag(ds, tag: tuple) -> str:
    """Safely extract a DICOM tag value as a string."""
    try:
        val = ds[tag].value if tag in ds else None
        if val is None:
            return "Unknown"
        val = str(val).strip()
        return val if val else "Unknown"
    except Exception:
        return "Unknown"


class FairnessService:

    def scan_s3_demographics(
        self,
        s3_uri: str,
        s3_client,
        max_files: int = 200,
        file_extensions: tuple = (".dcm", ".dicom"),
    ) -> Dict[str, Any]:
        """
        Scan DICOM files under an S3 prefix and return demographic distributions.

        Returns a dict with:
          - total_scanned: int
          - total_found: int
          - attributes: { attr_name: { value: count, ... }, ... }
          - age_distribution: { bucket: count }
          - missing_rates: { attr_name: pct_missing }
          - files_sample: list of { key, PatientSex, PatientAge, EthnicGroup }
        """
        import pydicom

        # Parse bucket and prefix from s3://bucket/prefix
        path   = s3_uri.replace("s3://", "").rstrip("/")
        parts  = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        # List DICOM objects
        paginator = s3_client.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if any(key.lower().endswith(ext) for ext in file_extensions):
                    keys.append(key)
                if len(keys) >= max_files * 5:  # stop listing early
                    break
            if len(keys) >= max_files * 5:
                break

        total_found = len(keys)
        keys = keys[:max_files]

        # Aggregate demographics
        attributes: Dict[str, Dict[str, int]] = {k: defaultdict(int) for k in DEMOGRAPHIC_TAGS}
        age_distribution: Dict[str, int] = defaultdict(int)
        files_sample: List[dict] = []
        errors = 0

        for key in keys:
            try:
                obj      = s3_client.get_object(Bucket=bucket, Key=key)
                raw      = obj["Body"].read()
                ds       = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)

                row = {"key": key}
                for attr, tag in DEMOGRAPHIC_TAGS.items():
                    val = _extract_tag(ds, tag)
                    attributes[attr][val] += 1
                    row[attr] = val

                # Age bucketing
                age_val = _extract_tag(ds, DEMOGRAPHIC_TAGS["PatientAge"])
                age_int = _parse_age(age_val if age_val != "Unknown" else "")
                bucket_label = _age_bucket(age_int)
                age_distribution[bucket_label] += 1
                row["AgeBucket"] = bucket_label

                files_sample.append(row)

            except Exception as e:
                logger.debug("Could not read %s: %s", key, e)
                errors += 1

        total_scanned = len(keys)

        # Missing rates
        missing_rates = {}
        for attr in DEMOGRAPHIC_TAGS:
            unknown_count = attributes[attr].get("Unknown", 0)
            missing_rates[attr] = round(unknown_count / total_scanned * 100, 1) if total_scanned else 0

        return {
            "s3_uri":           s3_uri,
            "total_found":      total_found,
            "total_scanned":    total_scanned,
            "errors":           errors,
            "attributes":       {k: dict(v) for k, v in attributes.items()},
            "age_distribution": dict(age_distribution),
            "missing_rates":    missing_rates,
            "files_sample":     files_sample[:50],  # cap sample for response size
        }
