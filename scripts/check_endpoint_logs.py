"""Check CloudWatch logs for the segmentation endpoint — look for Python errors."""
import boto3
import time

endpoint_name = "segmentation"
region = "us-east-1"

logs_client = boto3.client("logs", region_name=region)
log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"

print(f"Checking logs in: {log_group}")
print("=" * 60)

try:
    # Get ALL log streams
    streams = logs_client.describe_log_streams(
        logGroupName=log_group,
        orderBy="LastEventTime",
        descending=True,
        limit=5,
    )

    for stream in streams["logStreams"]:
        stream_name = stream["logStreamName"]
        print(f"\n{'='*60}")
        print(f"Stream: {stream_name}")
        print(f"{'='*60}")

        # Get first 100 events (startup/error logs)
        events = logs_client.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            limit=100,
            startFromHead=True,
        )

        for event in events["events"]:
            ts = time.strftime("%H:%M:%S", time.localtime(event["timestamp"] / 1000))
            msg = event["message"].strip()
            # Show everything — we need to find the traceback
            print(f"  [{ts}] {msg}")

except logs_client.exceptions.ResourceNotFoundException:
    print(f"Log group '{log_group}' not found.")
except Exception as e:
    print(f"Error: {e}")
