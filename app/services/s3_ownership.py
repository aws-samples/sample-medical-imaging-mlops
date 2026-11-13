"""S3 bucket-ownership guard (bucket-squatting protection).

Before the dashboard uploads job CODE (processing/evaluation scripts, staged
bundles that SageMaker later executes) to a bucket named in a request, confirm
the bucket belongs to this AWS account. A bucket with a predictable name that
was registered by someone else would otherwise receive the upload, and the
job would run whatever that account puts back in its place.

The expected account comes from AWS_ACCOUNT_ID (.env.local). S3 enforces the
check server-side via the ExpectedBucketOwner parameter (403 on mismatch).
"""
import logging
import os
import re

logger = logging.getLogger(__name__)


class BucketOwnershipError(ValueError):
    """The target bucket is not owned by the configured AWS account."""


def expected_bucket_owner() -> str | None:
    acct = os.environ.get("AWS_ACCOUNT_ID", "").strip()
    return acct if re.fullmatch(r"\d{12}", acct) else None


def assert_bucket_owned(s3_client, bucket: str) -> None:
    """Raise BucketOwnershipError unless `bucket` is owned by AWS_ACCOUNT_ID.

    Fails closed when REQUIRE_BUCKET_OWNER_CHECK=true and AWS_ACCOUNT_ID is unset;
    otherwise logs a warning and skips (local setups without the variable).
    """
    owner = expected_bucket_owner()
    if not owner:
        if os.environ.get("REQUIRE_BUCKET_OWNER_CHECK", "").lower() in ("1", "true", "yes"):
            raise BucketOwnershipError("AWS_ACCOUNT_ID must be set to verify S3 bucket ownership")
        logger.warning("AWS_ACCOUNT_ID not set — skipping ownership check for s3://%s", bucket)
        return
    from botocore.exceptions import ClientError
    try:
        s3_client.head_bucket(Bucket=bucket, ExpectedBucketOwner=owner)
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("403", "AccessDenied", "Forbidden"):
            raise BucketOwnershipError(
                f"s3://{bucket} is not owned by account {owner} (or is not accessible); refusing to upload job code"
            ) from e
        if code in ("404", "NoSuchBucket", "NotFound"):
            raise BucketOwnershipError(f"s3://{bucket} does not exist") from e
        raise


def owner_extra_args() -> dict:
    """ExtraArgs for upload_file so S3 also enforces ownership on the write itself."""
    owner = expected_bucket_owner()
    return {"ExpectedBucketOwner": owner} if owner else {}
