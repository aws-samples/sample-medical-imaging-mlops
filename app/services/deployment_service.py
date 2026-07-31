"""Service for deploying models to SageMaker endpoints"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

DEPLOYMENTS_FILE = Path("data/deployments.json")


class DeploymentService:
    def __init__(self):
        self.deployments: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if DEPLOYMENTS_FILE.exists():
            try:
                data = json.loads(DEPLOYMENTS_FILE.read_text())
                # Drop any entries that were already marked Deleted
                self.deployments = {k: v for k, v in data.items() if v.get("status") != "Deleted"}
                logger.info("Loaded %d persisted deployments", len(self.deployments))
            except Exception as e:
                logger.warning("Could not load deployments file: %s", e)

    def _save(self):
        try:
            DEPLOYMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            DEPLOYMENTS_FILE.write_text(json.dumps(self.deployments, indent=2, default=str))
        except Exception as e:
            logger.warning("Could not save deployments file: %s", e)

    def create_endpoint(self, config: dict, sagemaker_session, role_arn: str) -> dict:
        """Deploy a model to a SageMaker real-time or async endpoint using boto3 directly."""
        if config.get("async_endpoint"):
            return self._create_async_endpoint(config, sagemaker_session, role_arn)
        return self._create_realtime_endpoint(config, sagemaker_session, role_arn)

    def _create_realtime_endpoint(self, config: dict, sagemaker_session, role_arn: str) -> dict:
        """Deploy a real-time endpoint."""
        sagemaker_client = sagemaker_session.boto_session.client("sagemaker")
        endpoint_name   = config["endpoint_name"]
        model_artifacts = config["model_artifacts"]
        image_uri       = config["image_uri"]
        entry_point     = config.get("inference_script", "inference.py")
        instance_type   = config.get("instance_type", "ml.m5.xlarge")
        instance_count  = config.get("instance_count", 1)

        model_name  = f"{endpoint_name}-model"
        config_name = f"{endpoint_name}-config"

        # model.tar.gz already contains inference.py (copied by train.py at end of training)
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image":        image_uri,
                "ModelDataUrl": model_artifacts,
                "Environment": {
                    "SAGEMAKER_PROGRAM":             entry_point,
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                },
            },
            ExecutionRoleArn=role_arn,
        )
        endpoint_config_kwargs: dict = {
            "EndpointConfigName": config_name,
            "ProductionVariants": [{
                "VariantName":          "AllTraffic",
                "ModelName":            model_name,
                "InstanceType":         instance_type,
                "InitialInstanceCount": instance_count,
            }],
        }

        # Optional data capture
        if config.get("enable_data_capture") and config.get("capture_s3_uri"):
            endpoint_config_kwargs["DataCaptureConfig"] = {
                "EnableCapture":             True,
                "InitialSamplingPercentage": config.get("capture_sampling_pct", 100),
                "DestinationS3Uri":          config["capture_s3_uri"],
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
                "CaptureContentTypeHeader": {
                    "JsonContentTypes": ["application/json"],
                },
            }

        sagemaker_client.create_endpoint_config(**endpoint_config_kwargs)
        sagemaker_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        logger.info("Creating real-time endpoint: %s", endpoint_name)

        # Infer task type from image URI or endpoint name
        task_type = "classification"
        combined = f"{image_uri or ''} {endpoint_name}".lower()
        if "segmentation" in combined or "segment" in combined:
            task_type = "segmentation"

        record = {
            "endpoint_name":        endpoint_name,
            "endpoint_type":        "realtime",
            "task_type":            task_type,
            "model_artifacts":      model_artifacts,
            "image_uri":            image_uri,
            "instance_type":        instance_type,
            "instance_count":       instance_count,
            "data_capture_enabled": bool(config.get("enable_data_capture") and config.get("capture_s3_uri")),
            "capture_s3_uri":       config.get("capture_s3_uri"),
            "status":               "Creating",
            "created_at":           datetime.utcnow().isoformat(),
        }
        self.deployments[endpoint_name] = record
        self._save()
        return record

    def _create_async_endpoint(self, config: dict, sagemaker_session, role_arn: str) -> dict:
        """Deploy a SageMaker async inference endpoint."""
        sagemaker_client = sagemaker_session.boto_session.client("sagemaker")
        endpoint_name   = config["endpoint_name"]
        model_artifacts = config["model_artifacts"]
        image_uri       = config["image_uri"]
        source_dir      = config.get("source_dir", "src/classification")
        entry_point     = config.get("inference_script", "inference.py")
        instance_type   = config.get("instance_type", "ml.m5.xlarge")
        instance_count  = config.get("instance_count", 1)
        output_s3_uri   = config.get("output_s3_uri")
        max_concurrent  = config.get("max_concurrent", 10)

        if not output_s3_uri:
            raise ValueError("output_s3_uri is required for async endpoints")

        model_name  = f"{endpoint_name}-model"
        config_name = f"{endpoint_name}-config"
        output_pattern = config.get("output_pattern", "{patient_id}_segmentation.nii.gz")

        # model.tar.gz already contains inference.py (copied by train.py at end of training)
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image":        image_uri,
                "ModelDataUrl": model_artifacts,
                "Environment": {
                    "SAGEMAKER_PROGRAM":             entry_point,
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "OUTPUT_S3_URI":                 output_s3_uri,
                    "OUTPUT_FILENAME_PATTERN":       output_pattern,
                },
            },
            ExecutionRoleArn=role_arn,
        )
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                "VariantName":          "AllTraffic",
                "ModelName":            model_name,
                "InstanceType":         instance_type,
                "InitialInstanceCount": instance_count,
            }],
            AsyncInferenceConfig={
                "OutputConfig": {
                    "S3OutputPath": output_s3_uri,
                },
                "ClientConfig": {
                    "MaxConcurrentInvocationsPerInstance": max_concurrent,
                },
            },
        )
        sagemaker_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        logger.info("Creating async endpoint: %s", endpoint_name)

        # Infer task type from image URI or endpoint name
        task_type = "classification"
        combined = f"{image_uri or ''} {endpoint_name}".lower()
        if "segmentation" in combined or "segment" in combined:
            task_type = "segmentation"

        record = {
            "endpoint_name":   endpoint_name,
            "endpoint_type":   "async",
            "task_type":       task_type,
            "model_artifacts": model_artifacts,
            "image_uri":       image_uri,
            "instance_type":   instance_type,
            "instance_count":  instance_count,
            "output_s3_uri":   output_s3_uri,
            "output_pattern":  output_pattern,
            "status":          "Creating",
            "created_at":      datetime.utcnow().isoformat(),
        }
        self.deployments[endpoint_name] = record
        self._save()
        return record

    def invoke_async(self, endpoint_name: str, input_s3_uri: str, output_s3_uri: str, boto_session) -> dict:
        """Submit an async inference request. Input must be a pre-uploaded S3 object."""
        client = boto_session.client("sagemaker-runtime")
        response = client.invoke_endpoint_async(
            EndpointName=endpoint_name,
            InputLocation=input_s3_uri,
            ContentType="application/json",
            Accept="application/json",
        )
        return {
            "inference_id":    response.get("InferenceId"),
            "output_location": response.get("OutputLocation"),
            "status":          "Submitted",
        }

    def get_endpoint_status(self, endpoint_name: str, sagemaker_client) -> dict:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status   = response["EndpointStatus"]
            record   = self.deployments.get(endpoint_name, {})
            record["status"] = status
            record["endpoint_name"] = endpoint_name
            record["endpoint_url"] = (
                f"https://runtime.sagemaker.{response.get('EndpointArn','').split(':')[3]}"
                f".amazonaws.com/endpoints/{endpoint_name}/invocations"
                if status == "InService" else None
            )
            if status == "Failed":
                record["failure_reason"] = response.get("FailureReason")

            # Get instance count from production variants
            variants = response.get("ProductionVariants", [])
            if variants:
                record["instance_count"] = variants[0].get("CurrentInstanceCount", variants[0].get("DesiredInstanceCount", 1))
                record["instance_type"] = record.get("instance_type") or variants[0].get("InstanceType")

            # Check endpoint config for async and data capture settings
            config_name = response.get("EndpointConfigName")
            if config_name:
                try:
                    cfg_resp = sagemaker_client.describe_endpoint_config(EndpointConfigName=config_name)

                    # Async inference config
                    async_cfg = cfg_resp.get("AsyncInferenceConfig", {})
                    if async_cfg:
                        record["endpoint_type"] = "async"
                        output_cfg = async_cfg.get("OutputConfig", {})
                        if output_cfg.get("S3OutputPath"):
                            record["output_s3_uri"] = output_cfg["S3OutputPath"]
                    else:
                        record["endpoint_type"] = record.get("endpoint_type") or "realtime"

                    # Instance type from production variants in config
                    cfg_variants = cfg_resp.get("ProductionVariants", [])
                    if cfg_variants and not record.get("instance_type"):
                        record["instance_type"] = cfg_variants[0].get("InstanceType")

                    # Data capture config
                    dc = cfg_resp.get("DataCaptureConfig", {})
                    if dc.get("EnableCapture"):
                        record["data_capture_enabled"] = True
                        record["capture_s3_uri"] = dc.get("DestinationS3Uri", "")
                        record["capture_sampling_pct"] = dc.get("InitialSamplingPercentage", 100)
                    else:
                        record["data_capture_enabled"] = False
                except Exception:
                    pass

            self.deployments[endpoint_name] = record
            self._save()
            return record
        except Exception as e:
            err_str = str(e)
            if "Could not find endpoint" in err_str or "ValidationException" in err_str or "does not exist" in err_str:
                # Endpoint deleted externally — remove from local store
                self.deployments.pop(endpoint_name, None)
                self._save()
                return {"endpoint_name": endpoint_name, "status": "Deleted"}
            raise

    def scale_endpoint(self, endpoint_name: str, desired_count: int, sagemaker_client) -> dict:
        """Scale an endpoint up or down. Setting desired_count=0 scales to zero (no billing)."""
        # Get current endpoint config
        desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        config_name = desc.get("EndpointConfigName")

        if not config_name:
            raise ValueError(f"Could not determine endpoint config for '{endpoint_name}'")

        # Describe current config to get variant info
        config_desc = sagemaker_client.describe_endpoint_config(EndpointConfigName=config_name)
        variants = config_desc.get("ProductionVariants", [])
        if not variants:
            raise ValueError(f"No production variants found for endpoint '{endpoint_name}'")

        variant_name = variants[0]["VariantName"]

        if desired_count == 0:
            # Scale to zero using update_endpoint_weights_and_capacities
            sagemaker_client.update_endpoint_weights_and_capacities(
                EndpointName=endpoint_name,
                DesiredWeightsAndCapacities=[{
                    "VariantName": variant_name,
                    "DesiredInstanceCount": 0,
                }],
            )
            logger.info("Scaling endpoint '%s' to 0 instances", endpoint_name)
        else:
            sagemaker_client.update_endpoint_weights_and_capacities(
                EndpointName=endpoint_name,
                DesiredWeightsAndCapacities=[{
                    "VariantName": variant_name,
                    "DesiredInstanceCount": desired_count,
                }],
            )
            logger.info("Scaling endpoint '%s' to %d instance(s)", endpoint_name, desired_count)

        # Update local record
        record = self.deployments.get(endpoint_name, {})
        record["instance_count"] = desired_count
        record["status"] = "Updating"
        self.deployments[endpoint_name] = record
        self._save()

        return {
            "endpoint_name": endpoint_name,
            "desired_count": desired_count,
            "status": "Updating",
        }

    def delete_endpoint(self, endpoint_name: str, sagemaker_client):
        try:
            try:
                desc        = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                config_name = desc.get("EndpointConfigName")
            except Exception:
                config_name = self.deployments.get(endpoint_name, {}).get("config_name")

            model_name = f"{endpoint_name}-model"

            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info("Deleted endpoint: %s", endpoint_name)

            if config_name:
                try:
                    sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
                    logger.info("Deleted endpoint config: %s", config_name)
                except Exception as e:
                    logger.warning("Could not delete endpoint config %s: %s", config_name, e)

            try:
                sagemaker_client.delete_model(ModelName=model_name)
                logger.info("Deleted model: %s", model_name)
            except Exception as e:
                logger.warning("Could not delete model %s: %s", model_name, e)
        finally:
            self.deployments.pop(endpoint_name, None)
            self._save()

    def list_deployments(self) -> list:
        return list(self.deployments.values())

    def import_endpoint(self, endpoint_name: str, sagemaker_client) -> dict:
        """Register an existing SageMaker endpoint for tracking without deploying."""
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status   = response["EndpointStatus"]
        record   = self.deployments.get(endpoint_name) or {
            "endpoint_name":  endpoint_name,
            "endpoint_type":  "realtime",
            "model_artifacts": None,
            "image_uri":      None,
            "instance_type":  None,
            "instance_count": None,
            "created_at":     response.get("CreationTime", "").isoformat()
                              if response.get("CreationTime") else None,
        }
        record["status"] = status
        if status == "InService":
            region = sagemaker_client.meta.region_name
            record["endpoint_url"] = (
                f"https://runtime.sagemaker.{region}.amazonaws.com"
                f"/endpoints/{endpoint_name}/invocations"
            )
        self.deployments[endpoint_name] = record
        self._save()
        logger.info("Imported existing endpoint: %s (%s)", endpoint_name, status)
        return record

    def sync_from_aws(self, sagemaker_client, name_contains: str = None) -> dict:
        """Pull endpoints from AWS and merge into the local store.

        Endpoints already tracked locally have their status updated.
        Endpoints not yet tracked are added with a minimal record.
        Returns {"added": [...], "updated": [...]} with affected endpoint names.
        """
        kwargs = {
            "SortBy": "CreationTime",
            "SortOrder": "Descending",
            "MaxResults": 100,
        }
        if name_contains:
            kwargs["NameContains"] = name_contains

        response = sagemaker_client.list_endpoints(**kwargs)
        summaries = response.get("Endpoints", [])

        added = []
        updated = []

        for summary in summaries:
            endpoint_name = summary["EndpointName"]
            status = summary["EndpointStatus"]
            creation_time = summary.get("CreationTime")

            if endpoint_name in self.deployments:
                self.deployments[endpoint_name]["status"] = status
                if creation_time:
                    self.deployments[endpoint_name]["created_at"] = creation_time.isoformat() if hasattr(creation_time, 'isoformat') else str(creation_time)
                if status == "InService":
                    region = sagemaker_client.meta.region_name
                    self.deployments[endpoint_name]["endpoint_url"] = (
                        f"https://runtime.sagemaker.{region}.amazonaws.com"
                        f"/endpoints/{endpoint_name}/invocations"
                    )
                updated.append(endpoint_name)
            else:
                record = {
                    "endpoint_name":  endpoint_name,
                    "endpoint_type":  "realtime",
                    "model_artifacts": None,
                    "image_uri":      None,
                    "instance_type":  None,
                    "instance_count": None,
                    "status":         status,
                    "created_at":     creation_time.isoformat() if hasattr(creation_time, 'isoformat') else str(creation_time) if creation_time else None,
                }
                if status == "InService":
                    region = sagemaker_client.meta.region_name
                    record["endpoint_url"] = (
                        f"https://runtime.sagemaker.{region}.amazonaws.com"
                        f"/endpoints/{endpoint_name}/invocations"
                    )
                self.deployments[endpoint_name] = record
                added.append(endpoint_name)

        self._save()
        logger.info("sync_from_aws (deployments): added %d, updated %d", len(added), len(updated))
        return {"added": added, "updated": updated}
