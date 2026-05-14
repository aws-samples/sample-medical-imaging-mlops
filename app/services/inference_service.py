"""Service for running inference against SageMaker endpoints"""
import json
import logging
import boto3
from botocore.config import Config
from typing import Optional

logger = logging.getLogger(__name__)

# 3D segmentation inference can be slow — allow up to 5 minutes
_INFERENCE_CONFIG = Config(
    read_timeout=300,
    connect_timeout=10,
    retries={"max_attempts": 1},
)


class InferenceService:

    def invoke_endpoint(
        self,
        endpoint_name: str,
        input_data: dict,
        task_type: str,
        output_s3_uri: Optional[str],
        boto_session,
    ) -> dict:
        """Invoke a SageMaker real-time endpoint and optionally save result to S3."""
        runtime = boto_session.client("sagemaker-runtime", config=_INFERENCE_CONFIG)

        # Handle raw bytes (local file upload) vs JSON payload
        if "bytes" in input_data:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/dicom",
                Accept="application/json",
                Body=input_data["bytes"],
            )
        else:
            s3_uri    = input_data.get("s3_uri")
            file_path = input_data.get("file_path")
            if s3_uri:
                body = json.dumps({"s3_uri": s3_uri})
            else:
                body = json.dumps({"file_path": file_path or ""})
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=body,
            )
        result_str = response["Body"].read().decode("utf-8")
        result     = json.loads(result_str)

        # Normalise response — handle both inference_realtime.py and inference.py formats
        if "predictions" in result and "predicted_class" in result:
            labels = [
                'Surgical_implant', 'Vertebral_collapse', 'Spondylolysthesis',
                'No_finding', 'Foraminal_stenosis', 'Other_lesions',
                'Disc_space_narrowing', 'Osteophytes'
            ]
            probs     = result["predictions"][0]
            pred_idx  = result["predicted_class"][0]
            conf      = result["confidence"][0]
            result = {
                "predicted_class_index": pred_idx,
                "predicted_class":       labels[pred_idx] if pred_idx < len(labels) else str(pred_idx),
                "confidence":            round(conf, 4),
                "probabilities":         {l: round(p, 4) for l, p in zip(labels, probs)},
            }

        # Save to S3 if requested
        if output_s3_uri:
            try:
                parts  = output_s3_uri.replace("s3://", "").split("/", 1)
                bucket = parts[0]
                prefix = parts[1].rstrip("/") if len(parts) > 1 else "inference-results"
                key    = f"{prefix}/{endpoint_name}-result.json"
                s3     = boto_session.client("s3")
                s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=result_str.encode("utf-8"),
                    ContentType="application/json",
                )
                result["saved_to"] = f"s3://{bucket}/{key}"
                logger.info("Saved inference result to s3://%s/%s", bucket, key)
            except Exception as e:
                logger.warning("Could not save result to S3: %s", e)
                result["save_error"] = str(e)

        result["task_type"] = task_type
        return result
