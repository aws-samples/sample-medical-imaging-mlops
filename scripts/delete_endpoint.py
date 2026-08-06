"""Delete the segmentation endpoint, endpoint config, and model."""
import boto3

endpoint_name = "segmentation"
region = "us-east-1"

sm = boto3.client("sagemaker", region_name=region)

try:
    # Get endpoint config and model names before deleting
    desc = sm.describe_endpoint(EndpointName=endpoint_name)
    config_name = desc["EndpointConfigName"]

    config_desc = sm.describe_endpoint_config(EndpointConfigName=config_name)
    model_name = config_desc["ProductionVariants"][0]["ModelName"]

    # Delete endpoint
    sm.delete_endpoint(EndpointName=endpoint_name)
    print(f"✅ Deleted endpoint: {endpoint_name}")

    # Delete endpoint config
    sm.delete_endpoint_config(EndpointConfigName=config_name)
    print(f"✅ Deleted endpoint config: {config_name}")

    # Delete model
    sm.delete_model(ModelName=model_name)
    print(f"✅ Deleted model: {model_name}")

except sm.exceptions.ClientError as e:
    print(f"❌ Error: {e}")
