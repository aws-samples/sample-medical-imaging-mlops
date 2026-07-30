"""Service for running inference against SageMaker endpoints"""
import json
import logging
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
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
        """Invoke a SageMaker endpoint. Tries real-time first; falls back to async
        if the endpoint only supports async inference."""
        runtime = boto_session.client("sagemaker-runtime", config=_INFERENCE_CONFIG)

        try:
            return self._invoke_realtime(runtime, endpoint_name, input_data, task_type, output_s3_uri, boto_session)
        except ClientError as e:
            error_msg = str(e)
            # "does not support this inference type" means it's an async endpoint
            if "does not support this inference type" in error_msg or "ValidationError" in e.response.get("Error", {}).get("Code", ""):
                logger.info("Endpoint '%s' does not support real-time inference, trying async...", endpoint_name)
                return self._invoke_async(runtime, endpoint_name, input_data, task_type, output_s3_uri, boto_session)
            raise

    def _invoke_realtime(
        self, runtime, endpoint_name, input_data, task_type, output_s3_uri, boto_session
    ) -> dict:
        """Invoke a real-time SageMaker endpoint."""
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

    def _invoke_async(
        self, runtime, endpoint_name, input_data, task_type, output_s3_uri, boto_session
    ) -> dict:
        """Invoke an async SageMaker endpoint. Uploads input to S3 first,
        then calls invoke_endpoint_async."""
        s3 = boto_session.client("s3")

        # Determine input S3 location
        s3_uri = input_data.get("s3_uri")

        if s3_uri:
            # Build a JSON input file — send both keys for compatibility with
            # BYOC images (expects s3_uri) and DLC code/ approach (expects file_path)
            input_body = json.dumps({"s3_uri": s3_uri, "file_path": s3_uri})
        elif "bytes" in input_data:
            input_body = input_data["bytes"]
        elif input_data.get("file_path"):
            input_body = json.dumps({"file_path": input_data["file_path"]})
        else:
            raise ValueError("No input data provided for async inference")

        # Upload input to S3 for async invocation
        # Use output_s3_uri bucket or infer from endpoint config
        if output_s3_uri:
            parts = output_s3_uri.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            prefix = parts[1].rstrip("/") if len(parts) > 1 else "inference-input"
        else:
            # Default bucket — caller should provide output_s3_uri for async
            raise ValueError(
                f"Endpoint '{endpoint_name}' requires async inference. "
                "Please provide an Output S3 URI where results will be stored."
            )

        import uuid
        input_key = f"{prefix}/async-input/{uuid.uuid4()}.json"
        content_type = "application/json"

        if isinstance(input_body, bytes):
            content_type = "application/octet-stream"
            input_key = f"{prefix}/async-input/{uuid.uuid4()}.bin"

        s3.put_object(
            Bucket=bucket,
            Key=input_key,
            Body=input_body if isinstance(input_body, bytes) else input_body.encode("utf-8"),
            ContentType=content_type,
        )
        input_s3_location = f"s3://{bucket}/{input_key}"
        logger.info("Uploaded async input to %s", input_s3_location)

        # Invoke async
        response = runtime.invoke_endpoint_async(
            EndpointName=endpoint_name,
            InputLocation=input_s3_location,
            ContentType=content_type,
            Accept="application/json",
        )

        result = {
            "async":           True,
            "inference_id":    response.get("InferenceId"),
            "output_location": response.get("OutputLocation"),
            "input_location":  input_s3_location,
            "status":          "Submitted",
            "task_type":       task_type,
            "message":         f"Async inference submitted. Results will be written to: {response.get('OutputLocation')}",
        }

        logger.info("Async inference submitted: id=%s output=%s",
                    result["inference_id"], result["output_location"])
        return result
