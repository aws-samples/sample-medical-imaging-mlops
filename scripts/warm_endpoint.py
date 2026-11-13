"""Scale the segmentation endpoint to 1 instance and wait for it to be InService."""
import boto3
import time

endpoint_name = "segmentation"
sm = boto3.client("sagemaker", region_name="us-east-1")

# Check current status
desc = sm.describe_endpoint(EndpointName=endpoint_name)
status = desc["EndpointStatus"]
instance_count = desc["ProductionVariants"][0].get("CurrentInstanceCount", 0)
print(f"Current status: {status}, instances: {instance_count}")

if instance_count >= 1 and status == "InService":
    print("Endpoint is already warm. Ready to go.")
else:
    print("Scaling to 1 instance...")
    sm.update_endpoint_weights_and_capacities(
        EndpointName=endpoint_name,
        DesiredWeightsAndCapacities=[{
            "VariantName": "primary",
            "DesiredInstanceCount": 1,
        }]
    )
    print("Waiting for InService (this may take 8-12 min for GPU instances)...")

    # SageMaker's own waiter polls describe_endpoint (30 s x 30 = 15 min) and
    # fails fast if the endpoint goes to Failed.
    from botocore.exceptions import WaiterError
    start = time.time()
    try:
        sm.get_waiter("endpoint_in_service").wait(
            EndpointName=endpoint_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 30},
        )
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        count = desc["ProductionVariants"][0].get("CurrentInstanceCount", 0)
        print(f"\n✅ Endpoint is warm! ({int(time.time() - start)}s, instances: {count})")
    except WaiterError as e:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        if desc["EndpointStatus"] == "Failed":
            print(f"\n❌ Endpoint failed: {desc.get('FailureReason', 'Unknown')}")
        else:
            print(f"\n❌ Timed out waiting for endpoint to warm up: {e}")
