"""
Integration test for SageMaker endpoint inference.
Invokes a real endpoint with a real S3 image or local file and validates the response.

Usage:
    pytest tests/integration/test_endpoint_inference.py -v -s

Environment variables (or set directly below):
    AWS_DEFAULT_REGION   default: us-east-1
    ENDPOINT_NAME        default: test-endp
    TEST_IMAGE_S3_URI    default: s3://public-datasets-imaging-us-east-1/vindr-spinexr-subset/test/Foraminal_stenosis/412cb981a940a9689347c614cb1489dd.dcm
    TEST_LOCAL_IMAGE     path to a local image file (DICOM, PNG, JPG); if not set a synthetic PNG is used
"""
import base64
import json
import os
import tempfile
import boto3
import numpy as np
import pytest
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
REGION        = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "test-endp")
TEST_IMAGE    = os.environ.get(
    "TEST_IMAGE_S3_URI",
    "s3://public-datasets-imaging-us-east-1/vindr-spinexr-subset/test/"
    "Foraminal_stenosis/412cb981a940a9689347c614cb1489dd.dcm",
)
TEST_LOCAL_IMAGE = os.environ.get("TEST_LOCAL_IMAGE", "")


@pytest.fixture(scope="module")
def runtime_client():
    return boto3.client(
        "sagemaker-runtime",
        region_name=REGION,
        config=boto3.session.Config(
            connect_timeout=10,
            read_timeout=60,
        )
    )


@pytest.fixture(scope="module")
def sagemaker_client():
    return boto3.client("sagemaker", region_name=REGION)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_endpoint_is_inservice(sagemaker_client):
    """Verify the endpoint exists and is InService before invoking."""
    response = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    status   = response["EndpointStatus"]
    assert status == "InService", (
        f"Endpoint '{ENDPOINT_NAME}' is not InService (status: {status}). "
        "Deploy the endpoint first."
    )
    print(f"\n✓ Endpoint '{ENDPOINT_NAME}' is InService")


LABELS = [
    'Surgical_implant', 'Vertebral_collapse', 'Spondylolysthesis',
    'No_finding', 'Foraminal_stenosis', 'Other_lesions',
    'Disc_space_narrowing', 'Osteophytes'
]


def test_invoke_with_s3_uri(runtime_client):
    """Invoke the endpoint with an S3 URI and validate the response structure."""
    # The deployed endpoint uses inference_realtime.py which expects "file_path" key
    payload = json.dumps({"file_path": TEST_IMAGE})

    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    body   = response["Body"].read().decode("utf-8")
    result = json.loads(body)

    # Save result to file
    out_path = f"tests/integration/result_{ENDPOINT_NAME}.json"
    with open(out_path, "w") as f:
        json.dump({"endpoint": ENDPOINT_NAME, "input": TEST_IMAGE, "result": result}, f, indent=2)
    print(f"\nResult saved to: {out_path}")
    print(f"\nRaw response:\n{json.dumps(result, indent=2)}")

    # Parse — inference_realtime.py returns {"predictions": [[...]], "predicted_class": [idx], "confidence": [val]}
    predictions   = result.get("predictions", [[]])[0]
    predicted_idx = result.get("predicted_class", [None])[0]
    confidence    = result.get("confidence", [None])[0]

    assert predictions,    "Response missing predictions"
    assert predicted_idx is not None, "Response missing predicted_class"
    assert confidence    is not None, "Response missing confidence"
    assert 0.0 <= confidence <= 1.0,  f"Confidence out of range: {confidence}"

    predicted_label = LABELS[predicted_idx] if predicted_idx < len(LABELS) else str(predicted_idx)
    probs_dict      = dict(zip(LABELS, predictions))

    print(f"\n✓ Predicted class : {predicted_label}")
    print(f"✓ Confidence      : {confidence:.1%}")
    print(f"✓ Probabilities   :")
    for cls, prob in sorted(probs_dict.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"    {cls:<30} {bar:<20} {prob:.1%}")


def test_invoke_response_time(runtime_client):
    """Verify the endpoint responds within 10 seconds."""
    import time
    payload = json.dumps({"file_path": TEST_IMAGE})

    start    = time.time()
    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    response["Body"].read()
    elapsed = time.time() - start

    print(f"\n✓ Response time: {elapsed:.2f}s")
    assert elapsed < 10.0, f"Endpoint too slow: {elapsed:.2f}s (limit: 10s)"


def _get_local_image_b64() -> tuple[str, str]:
    """
    Return (base64_encoded_bytes, filename) for the test image.
    Uses TEST_LOCAL_IMAGE env var if set, otherwise generates a synthetic 224x224 PNG.
    """
    if TEST_LOCAL_IMAGE and os.path.isfile(TEST_LOCAL_IMAGE):
        with open(TEST_LOCAL_IMAGE, "rb") as f:
            data = f.read()
        filename = os.path.basename(TEST_LOCAL_IMAGE)
    else:
        # Synthetic grayscale PNG — simulates a medical image when no real file is provided
        arr = (np.random.rand(224, 224) * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = tempfile.SpooledTemporaryFile()
        img.save(buf, format="PNG")
        buf.seek(0)
        data = buf.read()
        filename = "synthetic_test.png"
        print(f"\n  (no TEST_LOCAL_IMAGE set — using synthetic {filename})")

    return base64.b64encode(data).decode("utf-8"), filename


def test_invoke_with_local_file(runtime_client):
    """Invoke the endpoint with a base64-encoded local file and validate the response."""
    b64_data, filename = _get_local_image_b64()
    payload = json.dumps({
        "image_data": b64_data,
        "filename": filename,
    })

    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    body = response["Body"].read().decode("utf-8")
    result = json.loads(body)

    # Save result
    out_path = f"tests/integration/result_local_{ENDPOINT_NAME}.json"
    with open(out_path, "w") as f:
        json.dump({"endpoint": ENDPOINT_NAME, "input": filename, "result": result}, f, indent=2)
    print(f"\nResult saved to: {out_path}")
    print(f"\nRaw response:\n{json.dumps(result, indent=2)}")

    # Validate response structure
    assert "predicted_class" in result or "predicted_class_index" in result, (
        f"Response missing predicted_class. Got: {list(result.keys())}"
    )
    assert "confidence" in result, f"Response missing confidence. Got: {list(result.keys())}"
    assert "probabilities" in result, f"Response missing probabilities. Got: {list(result.keys())}"

    confidence = result["confidence"]
    assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"

    print(f"\n✓ File            : {filename}")
    print(f"✓ Predicted class : {result.get('predicted_class', result.get('predicted_class_index'))}")
    print(f"✓ Confidence      : {confidence:.1%}")
    if "probabilities" in result:
        print("✓ Probabilities   :")
        for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"    {cls:<30} {bar:<20} {prob:.1%}")
