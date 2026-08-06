"""Service for SageMaker Model Monitor — data capture, baselines, and monitoring schedules."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MONITORING_FILE = Path("data/monitoring.json")


class MonitoringService:
    def __init__(self):
        self.schedules: Dict[str, dict] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if MONITORING_FILE.exists():
            try:
                self.schedules = json.loads(MONITORING_FILE.read_text())
                logger.info("Loaded %d monitoring schedules", len(self.schedules))
            except Exception as e:
                logger.warning("Could not load monitoring file: %s", e)

    def _save(self):
        MONITORING_FILE.parent.mkdir(parents=True, exist_ok=True)
        MONITORING_FILE.write_text(json.dumps(self.schedules, indent=2, default=str))

    # ── Data Capture Viewer ───────────────────────────────────────────────────

    def get_capture_summary(
        self,
        capture_s3_uri: str,
        endpoint_name: str,
        s3_client,
        max_files: int = 20,
    ) -> dict:
        """
        Read the latest capture .jsonl files from S3, decode base64 outputs,
        and return aggregated prediction distribution stats.
        """
        import base64

        bucket, prefix = capture_s3_uri.replace("s3://", "").split("/", 1)
        prefix = prefix.rstrip("/") + f"/{endpoint_name}/AllTraffic/"

        # List latest .jsonl files (sorted by key = date/hour order)
        paginator = s3_client.get_paginator("list_objects_v2")
        all_keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".jsonl"):
                    all_keys.append((obj["LastModified"], obj["Key"]))

        if not all_keys:
            return {"total_records": 0, "message": "No capture files found yet"}

        # Sort by last modified, take most recent files
        all_keys.sort(key=lambda x: x[0], reverse=True)
        recent_keys = [k for _, k in all_keys[:max_files]]

        # Parse records
        records = []
        errors  = 0
        for key in recent_keys:
            try:
                obj  = s3_client.get_object(Bucket=bucket, Key=key)
                body = obj["Body"].read().decode("utf-8")
                for line in body.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        records.append(entry)
                    except Exception:
                        errors += 1
            except Exception as e:
                logger.warning("Could not read capture file %s: %s", key, e)

        if not records:
            return {"total_records": 0, "message": "Capture files found but no records parsed"}

        # Decode and aggregate outputs
        class_counts: dict = {}
        confidence_scores: list = []
        recent_rows: list = []
        label_map = [
            "Disc_space_narrowing", "Foraminal_stenosis", "No_finding",
            "Osteophytes", "Other_lesions", "Spondylolysthesis",
            "Surgical_implant", "Vertebral_collapse",
        ]

        for rec in records:
            try:
                capture = rec.get("captureData", {})
                output  = capture.get("endpointOutput", {})
                raw     = output.get("data", "")
                enc     = output.get("encoding", "BASE64")

                if enc == "BASE64":
                    decoded = base64.b64decode(raw).decode("utf-8")
                else:
                    decoded = raw

                result = json.loads(decoded)

                # Handle both response formats
                if "predictions" in result and "predicted_class" in result:
                    probs    = result["predictions"][0] if result["predictions"] else []
                    pred_idx = result["predicted_class"][0] if result["predicted_class"] else -1
                    conf     = result["confidence"][0] if result["confidence"] else 0.0
                elif "probabilities" in result:
                    probs    = list(result["probabilities"].values())
                    pred_idx = list(result["probabilities"].keys()).index(result.get("predicted_class", ""))
                    conf     = result.get("confidence", 0.0)
                else:
                    continue

                pred_label = label_map[pred_idx] if 0 <= pred_idx < len(label_map) else str(pred_idx)
                class_counts[pred_label] = class_counts.get(pred_label, 0) + 1
                confidence_scores.append(round(float(conf), 4))

                inference_time = rec.get("eventMetadata", {}).get("inferenceTime", "")
                recent_rows.append({
                    "time":            inference_time,
                    "predicted_class": pred_label,
                    "confidence":      round(float(conf), 4),
                    "probabilities":   {label_map[i]: round(float(p), 4)
                                        for i, p in enumerate(probs) if i < len(label_map)},
                })
            except Exception:
                errors += 1

        # Confidence histogram buckets
        buckets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist = [0] * (len(buckets) - 1)
        for c in confidence_scores:
            for i in range(len(buckets) - 1):
                if buckets[i] <= c < buckets[i + 1]:
                    hist[i] += 1
                    break
            else:
                hist[-1] += 1  # 1.0 goes in last bucket

        avg_conf = round(sum(confidence_scores) / len(confidence_scores), 4) if confidence_scores else 0

        return {
            "total_records":       len(records),
            "parsed_records":      len(recent_rows),
            "errors":              errors,
            "files_scanned":       len(recent_keys),
            "class_distribution":  class_counts,
            "confidence_histogram": {
                "buckets": [f"{buckets[i]:.1f}-{buckets[i+1]:.1f}" for i in range(len(hist))],
                "counts":  hist,
            },
            "avg_confidence":      avg_conf,
            "recent_records":      recent_rows[:50],  # cap for response size
        }

    # ── Data Capture ─────────────────────────────────────────────────────────

    def enable_data_capture(
        self,
        endpoint_name: str,
        capture_s3_uri: str,
        sampling_percentage: int,
        sagemaker_client,
    ) -> dict:
        """Enable or update data capture on an existing endpoint."""
        ep = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = ep["EndpointConfigName"]
        config = sagemaker_client.describe_endpoint_config(EndpointConfigName=config_name)

        # Data capture is not supported on async endpoints
        if config.get("AsyncInferenceConfig"):
            raise ValueError("Data capture is not supported on async endpoints. SageMaker does not allow DataCaptureConfig with AsyncInferenceConfig.")

        new_config_name = f"{endpoint_name}-dc-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        # Strip read-only fields from ProductionVariants before reusing
        _readonly = {
            "CurrentServerlessConfig", "ManagedInstanceScaling",
            "RoutingConfig", "CurrentWeight",
        }
        clean_variants = [
            {k: v for k, v in variant.items() if k not in _readonly}
            for variant in config["ProductionVariants"]
        ]

        create_kwargs = {
            "EndpointConfigName": new_config_name,
            "ProductionVariants": clean_variants,
            "DataCaptureConfig": {
                "EnableCapture":             True,
                "InitialSamplingPercentage": sampling_percentage,
                "DestinationS3Uri":          capture_s3_uri,
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
                "CaptureContentTypeHeader": {
                    "JsonContentTypes": ["application/json"],
                },
            },
        }
        if config.get("KmsKeyId"):
            create_kwargs["KmsKeyId"] = config["KmsKeyId"]
        # Preserve AsyncInferenceConfig if the original endpoint had it
        if config.get("AsyncInferenceConfig"):
            create_kwargs["AsyncInferenceConfig"] = config["AsyncInferenceConfig"]

        sagemaker_client.create_endpoint_config(**create_kwargs)
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=new_config_name,
        )
        logger.info("Enabled data capture on %s → %s", endpoint_name, capture_s3_uri)

        # Persist to monitoring.json
        record = {
            "endpoint_name":   endpoint_name,
            "capture_s3_uri":  capture_s3_uri,
            "sampling_pct":    sampling_percentage,
            "config_name":     new_config_name,
            "type":            "data_capture",
            "status":          "Updating",
            "enabled_at":      datetime.now(timezone.utc).isoformat(),
        }
        self.schedules[f"dc-{endpoint_name}"] = record
        self._save()
        return record

    # ── Baseline ─────────────────────────────────────────────────────────────

    def create_data_quality_baseline(
        self,
        job_name: str,
        baseline_data_s3_uri: str,
        output_s3_uri: str,
        role_arn: str,
        instance_type: str,
        sagemaker_client,
    ) -> dict:
        """Submit a data quality baseline job using the SageMaker Model Monitor prebuilt container."""
        image_uri = _get_monitor_image_uri(sagemaker_client)

        sagemaker_client.create_processing_job(
            ProcessingJobName=job_name,
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount":  1,
                    "InstanceType":   instance_type,
                    "VolumeSizeInGB": 20,
                }
            },
            # The prebuilt container has its own entrypoint — no custom script needed
            AppSpecification={
                "ImageUri": image_uri,
            },
            ProcessingInputs=[
                {
                    "InputName": "baseline_data_input_1",
                    "S3Input": {
                        "S3Uri":       baseline_data_s3_uri,
                        "LocalPath":   "/opt/ml/processing/input/baseline_data_input_1",
                        "S3DataType":  "S3Prefix",
                        "S3InputMode": "File",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                }
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "monitoring_output_1",
                        "S3Output": {
                            "S3Uri":        output_s3_uri,
                            "LocalPath":    "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
            RoleArn=role_arn,
            # These env vars tell the prebuilt container what to do
            Environment={
                "dataset_format":             '{"csv": {"header": true, "output_columns_position": "START"}}',
                "dataset_source":             "/opt/ml/processing/input/baseline_data_input_1",
                "output_path":                "/opt/ml/processing/output",
                "publish_cloudwatch_metrics": "Disabled",
            },
        )
        record = {
            "job_name":             job_name,
            "type":                 "baseline",
            "monitor_type":         "data_quality",
            "baseline_data_s3_uri": baseline_data_s3_uri,
            "output_s3_uri":        output_s3_uri,
            "instance_type":        instance_type,
            "status":               "InProgress",
            "created_at":           datetime.now(timezone.utc).isoformat(),
        }
        self.schedules[job_name] = record
        self._save()
        return record

    # ── Monitoring Schedules ──────────────────────────────────────────────────

    def create_data_quality_schedule(
        self,
        schedule_name: str,
        endpoint_name: str,
        baseline_constraints_s3_uri: str,
        baseline_statistics_s3_uri: str,
        output_s3_uri: str,
        role_arn: str,
        instance_type: str,
        schedule_expression: str,
        sagemaker_client,
    ) -> dict:
        """Create a data quality monitoring schedule."""
        sagemaker_client.create_monitoring_schedule(
            MonitoringScheduleName=schedule_name,
            MonitoringScheduleConfig={
                "ScheduleConfig": {
                    "ScheduleExpression": schedule_expression,
                },
                "MonitoringJobDefinition": {
                    "BaselineConfig": {
                        "ConstraintsResource": {"S3Uri": baseline_constraints_s3_uri},
                        "StatisticsResource":  {"S3Uri": baseline_statistics_s3_uri},
                    },
                    "MonitoringInputs": [
                        {
                            "EndpointInput": {
                                "EndpointName":  endpoint_name,
                                "LocalPath":     "/opt/ml/processing/input/endpoint",
                            }
                        }
                    ],
                    "MonitoringOutputConfig": {
                        "MonitoringOutputs": [
                            {
                                "S3Output": {
                                    "S3Uri":        output_s3_uri,
                                    "LocalPath":    "/opt/ml/processing/output",
                                    "S3UploadMode": "EndOfJob",
                                }
                            }
                        ]
                    },
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceCount":  1,
                            "InstanceType":   instance_type,
                            "VolumeSizeInGB": 20,
                        }
                    },
                    "MonitoringAppSpecification": {
                        "ImageUri": _get_monitor_image_uri(sagemaker_client),
                    },
                    "RoleArn": role_arn,
                },
                "MonitoringType": "DataQuality",
            },
        )
        record = {
            "schedule_name":               schedule_name,
            "type":                        "schedule",
            "monitor_type":                "data_quality",
            "endpoint_name":               endpoint_name,
            "output_s3_uri":               output_s3_uri,
            "schedule_expression":         schedule_expression,
            "instance_type":               instance_type,
            "status":                      "Scheduled",
            "created_at":                  datetime.now(timezone.utc).isoformat(),
            "last_execution_status":       None,
            "last_execution_time":         None,
        }
        self.schedules[schedule_name] = record
        self._save()
        return record

    def create_model_quality_schedule(
        self,
        schedule_name: str,
        endpoint_name: str,
        ground_truth_s3_uri: str,
        output_s3_uri: str,
        role_arn: str,
        instance_type: str,
        schedule_expression: str,
        problem_type: str,
        inference_attribute: str,
        probability_attribute: Optional[str],
        ground_truth_attribute: str,
        sagemaker_client,
    ) -> dict:
        """Create a model quality monitoring schedule."""
        job_def: dict = {
            "MonitoringInputs": [
                {
                    "EndpointInput": {
                        "EndpointName":        endpoint_name,
                        "LocalPath":           "/opt/ml/processing/input/endpoint",
                        "InferenceAttribute":  inference_attribute,
                    }
                }
            ],
            "MonitoringOutputConfig": {
                "MonitoringOutputs": [
                    {
                        "S3Output": {
                            "S3Uri":        output_s3_uri,
                            "LocalPath":    "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        }
                    }
                ]
            },
            "MonitoringResources": {
                "ClusterConfig": {
                    "InstanceCount":  1,
                    "InstanceType":   instance_type,
                    "VolumeSizeInGB": 20,
                }
            },
            "MonitoringAppSpecification": {
                "ImageUri": _get_monitor_image_uri(sagemaker_client),
            },
            "RoleArn": role_arn,
            "GroundTruthS3Input": {"S3Uri": ground_truth_s3_uri},
        }
        if probability_attribute:
            job_def["MonitoringInputs"][0]["EndpointInput"]["ProbabilityAttribute"] = probability_attribute

        sagemaker_client.create_monitoring_schedule(
            MonitoringScheduleName=schedule_name,
            MonitoringScheduleConfig={
                "ScheduleConfig": {"ScheduleExpression": schedule_expression},
                "MonitoringJobDefinition": job_def,
                "MonitoringType": "ModelQuality",
            },
        )
        record = {
            "schedule_name":         schedule_name,
            "type":                  "schedule",
            "monitor_type":          "model_quality",
            "endpoint_name":         endpoint_name,
            "ground_truth_s3_uri":   ground_truth_s3_uri,
            "output_s3_uri":         output_s3_uri,
            "schedule_expression":   schedule_expression,
            "problem_type":          problem_type,
            "instance_type":         instance_type,
            "status":                "Scheduled",
            "created_at":            datetime.now(timezone.utc).isoformat(),
            "last_execution_status": None,
            "last_execution_time":   None,
        }
        self.schedules[schedule_name] = record
        self._save()
        return record

    # ── Status & Results ──────────────────────────────────────────────────────

    def refresh_schedule(self, schedule_name: str, sagemaker_client) -> dict:
        """Refresh schedule status and latest execution from AWS."""
        self._load()
        resp = sagemaker_client.describe_monitoring_schedule(
            MonitoringScheduleName=schedule_name
        )
        status = resp["MonitoringScheduleStatus"]
        record = self.schedules.get(schedule_name, {"schedule_name": schedule_name})
        record["status"] = status

        # Get latest execution
        execs = sagemaker_client.list_monitoring_executions(
            MonitoringScheduleName=schedule_name,
            MaxResults=1,
            SortBy="CreationTime",
            SortOrder="Descending",
        ).get("MonitoringExecutionSummaries", [])

        if execs:
            ex = execs[0]
            record["last_execution_status"] = ex.get("MonitoringExecutionStatus")
            record["last_execution_time"]   = ex.get("CreationTime", "").isoformat() if hasattr(ex.get("CreationTime", ""), "isoformat") else str(ex.get("CreationTime", ""))
            record["last_execution_output"] = ex.get("ProcessingJobArn", "")
            if ex.get("FailureReason"):
                record["last_failure_reason"] = ex["FailureReason"]

        self.schedules[schedule_name] = record
        self._save()
        return record

    def delete_schedule(self, schedule_name: str, sagemaker_client) -> dict:
        """Stop and delete a monitoring schedule."""
        try:
            sagemaker_client.stop_monitoring_schedule(MonitoringScheduleName=schedule_name)
        except Exception:
            pass
        sagemaker_client.delete_monitoring_schedule(MonitoringScheduleName=schedule_name)
        self.schedules.pop(schedule_name, None)
        self._save()
        return {"deleted": schedule_name}

    def list_schedules(self, sagemaker_client=None) -> list:
        """Return all schedules, refreshing active ones from AWS."""
        self._load()
        if sagemaker_client:
            for name, record in list(self.schedules.items()):
                if record.get("type") == "schedule" and record.get("status") not in ("Stopped", "Failed"):
                    try:
                        self.refresh_schedule(name, sagemaker_client)
                    except Exception as e:
                        logger.warning("Could not refresh schedule %s: %s", name, e)
        return sorted(self.schedules.values(), key=lambda r: r.get("created_at", ""), reverse=True)


def _get_monitor_image_uri(sagemaker_client) -> str:
    """Return the SageMaker Model Monitor prebuilt container URI for us-east-1."""
    # Standard prebuilt image — region-specific
    # See: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-pre-built-container.html
    try:
        region = sagemaker_client.meta.region_name or "us-east-1"
    except Exception:
        region = "us-east-1"
    account_map = {
        "us-east-1":      "156813124566",
        "us-east-2":      "777275614652",
        "us-west-1":      "890145073186",
        "us-west-2":      "159807026194",
        "eu-west-1":      "468650794304",
        "eu-central-1":   "048819808253",
        "ap-northeast-1": "574779866223",
        "ap-southeast-1": "245545462676",
        "ap-southeast-2": "563025443158",
        "ap-south-1":     "126357580389",
    }
    account = account_map.get(region, "156813124566")
    return f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-model-monitor-analyzer"
