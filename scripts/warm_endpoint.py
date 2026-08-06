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

    start = time.time()
    while time.time() - start < 900:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        count = desc["ProductionVariants"][0].get("CurrentInstanceCount", 0)
        elapsed = int(time.time() - start)
        print(f"   [{elapsed}s] Status: {status}, Instances: {count}")

        if status == "InService" and count >= 1:
            print(f"\n✅ Endpoint is warm! ({elapsed}s)")
            break
        elif status == "Failed":
            print(f"\n❌ Endpoint failed: {desc.get('FailureReason', 'Unknown')}")
            break

        time.sleep(30)
    else:
        print("\n❌ Timed out waiting for endpoint to warm up.")
