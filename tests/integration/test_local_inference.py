"""
Simple local inference test — runs the inference pipeline directly
without needing a SageMaker endpoint.

Usage:
    pytest tests/integration/test_local_inference.py -v -s
    TEST_LOCAL_IMAGE=/path/to/image.dcm pytest tests/integration/test_local_inference.py -v -s
"""
import json
import os
import sys
import pytest

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

LOCAL_IMAGE = os.environ.get(
    "TEST_LOCAL_IMAGE",
    "data/test_data/classification-spine/412cb981a940a9689347c614cb1489dd.dcm",
)
MODEL_DIR = os.environ.get("MODEL_DIR", "data/models")


@pytest.fixture(scope="module")
def model_components():
    """Load model once for all tests in this module."""
    from inference.classification.inference import model_fn
    if not os.path.isdir(MODEL_DIR) or not any(
        f.endswith(".pth") for f in os.listdir(MODEL_DIR)
    ):
        pytest.skip(f"No model found in {MODEL_DIR} — set MODEL_DIR env var")
    return model_fn(MODEL_DIR)


def test_local_file_exists():
    """Check the test image is present before running inference."""
    assert os.path.isfile(LOCAL_IMAGE), (
        f"Test image not found: {LOCAL_IMAGE}\n"
        "Set TEST_LOCAL_IMAGE env var to point to a valid DICOM/PNG/JPG file."
    )
    print(f"\n✓ Found local image: {LOCAL_IMAGE}")


def test_input_fn_local_file():
    """input_fn should return a valid file path for a local file."""
    from inference.classification.inference import input_fn

    if not os.path.isfile(LOCAL_IMAGE):
        pytest.skip(f"Test image not found: {LOCAL_IMAGE}")

    payload = json.dumps({"file_path": LOCAL_IMAGE})
    result_path = input_fn(payload, "application/json")

    assert os.path.isfile(result_path), f"input_fn returned non-existent path: {result_path}"
    print(f"\n✓ input_fn returned: {result_path}")


def test_full_inference_local_file(model_components):
    """Run the full inference pipeline on a local file."""
    from inference.classification.inference import input_fn, predict_fn, output_fn

    if not os.path.isfile(LOCAL_IMAGE):
        pytest.skip(f"Test image not found: {LOCAL_IMAGE}")

    # input
    payload   = json.dumps({"file_path": LOCAL_IMAGE})
    file_path = input_fn(payload, "application/json")

    # predict
    prediction = predict_fn(file_path, model_components)

    # output
    raw    = output_fn(prediction, "application/json", model_components)
    result = json.loads(raw)

    print(f"\nRaw response:\n{json.dumps(result, indent=2)}")

    assert "predicted_class" in result or "predicted_class_index" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert 0.0 <= result["confidence"] <= 1.0

    print(f"\n✓ Predicted class : {result.get('predicted_class', result.get('predicted_class_index'))}")
    print(f"✓ Confidence      : {result['confidence']:.1%}")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"    {cls:<30} {bar:<20} {prob:.1%}")
