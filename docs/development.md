# Development Guide

## Project Structure

```
├── app/
│   ├── models/
│   │   └── schemas.py                      # All Pydantic request/response models and enums
│   ├── routers/
│   │   ├── aws_router.py                   # /api/aws — credentials, S3 browsing
│   │   ├── training_router.py              # /api/training — local + SageMaker training jobs
│   │   ├── deployment_router.py            # /api/deployment — SageMaker endpoint deployment
│   │   ├── inference_router.py             # /api/inference — real-time endpoint invocation
│   │   ├── evaluation_router.py            # /api/evaluation — Processing Job evaluation
│   │   ├── monitoring_router.py            # /api/monitoring — Model Monitor lifecycle
│   │   └── fairness_router.py             # /api/fairness — DICOM demographic analysis
│   └── services/
│       ├── aws_service.py                  # boto3 client management (S3, SageMaker)
│       ├── training_service.py             # Local model validation, import, registration
│       ├── sagemaker_training_service.py   # SageMaker BYOC job lifecycle
│       ├── deployment_service.py           # SageMaker endpoint deployment
│       ├── inference_service.py            # Real-time endpoint invocation
│       ├── evaluation_service.py           # Processing Job evaluation lifecycle
│       ├── monitoring_service.py           # Model Monitor — capture, baselines, schedules
│       ├── fairness_service.py             # DICOM demographic scanning
│       ├── validation_service.py           # Pre-launch validation (S3, IAM, ECR)
│       └── paper_report_service.py         # Academic report section generation
├── static/
│   ├── css/                                # Stylesheets
│   └── js/                                 # Frontend JavaScript (vanilla)
├── templates/                              # Jinja2 HTML templates
├── data/
│   ├── raw/                                # Raw input data (gitignored)
│   ├── processed/                          # Preprocessed datasets (gitignored)
│   ├── models/                             # Imported model artifacts (gitignored)
│   ├── custom_models/                      # Extracted custom training code uploads
│   ├── sagemaker_jobs.json                 # Persisted training job records
│   ├── deployments.json                    # Persisted endpoint records
│   ├── monitoring.json                     # Persisted Monitor schedule records
│   └── training_reports/                   # Per-job JSON reports
├── src/
│   ├── classification/                     # SageMaker training scripts (classification)
│   ├── segmentation/                       # SageMaker training scripts (segmentation)
│   ├── evaluation/
│   │   └── evaluate.py                     # Processing Job evaluation script
│   └── inference/
│       ├── classification/inference.py     # Classification serving hooks
│       └── segmentation/inference.py       # Segmentation serving hooks
├── tests/
│   ├── unit/                               # Unit tests with moto mocking
│   ├── integration/                        # Tests against live AWS resources
│   ├── fixtures/mock_data.py              # Shared test fixtures
│   └── invoke_endpoint.py                 # CLI endpoint invocation utility
├── scripts/
│   └── pre-commit                          # Git pre-commit hook
├── .github/workflows/
│   └── scrub-check.yml                    # CI: blocks hardcoded credentials
├── main.py                                 # App entry point
├── Dockerfile                              # Dashboard container
├── Dockerfile.sagemaker                    # Training BYOC container
├── Dockerfile.inference                    # Inference BYOC container
├── docker-compose.yml                      # Full stack (dashboard + MLflow + TensorBoard)
├── build_and_push.sh                       # Build & push training image
├── build_and_push_inference.sh             # Build & push inference image
└── requirements.txt                        # Python dependencies
```

---

## Code Conventions

### Schemas

All Pydantic models and enums live in `app/models/schemas.py`. This includes:
- Request models (e.g., `AWSCredentials`, `SageMakerTrainingConfig`, `ModelImportRequest`)
- Response models (e.g., `ImportedModel`, `TrainingJob`)
- Enums (e.g., `ModelType`, `TaskType`, `DimensionType`, `JobStatus`)

Router-local request models (e.g., `EndpointDeployRequest`, `InferenceRequest`) are defined in their respective router files when they are specific to that router only.

### Routers

- One file per API prefix (e.g., `training_router.py` → `/api/training`)
- Import service instances from sibling routers when sharing state:
  ```python
  from app.routers.aws_router import aws_service
  ```
- Each router creates its service instance(s) at module level
- Routers handle HTTP concerns (status codes, exceptions) — business logic stays in services

### Services

- Plain Python classes, no FastAPI dependencies
- Instantiated as module-level singletons in their respective router
- `AWSService` is a shared singleton imported from `aws_router`
- Always check `s3_client is not None` before use; raise `ValueError("AWS credentials not configured")` if not set
- State persistence: services that persist data use `_load()` / `_save()` methods with JSON files in `data/`

### Error Handling

- Services raise `ValueError` for bad input and `FileNotFoundError` for missing resources
- Routers catch exceptions and convert to `HTTPException`:
  - `ValueError` → 400 Bad Request
  - `FileNotFoundError` / `KeyError` → 404 Not Found
  - General `Exception` → 500 Internal Server Error
- Always log errors before re-raising:
  ```python
  except Exception as e:
      logger.error(f"Operation failed: {e}")
      raise HTTPException(status_code=500, detail=str(e))
  ```

### Logging

Each module creates its own logger:
```python
import logging
logger = logging.getLogger(__name__)
```

Use appropriate levels:
- `logger.info()` — successful operations, job starts
- `logger.warning()` — recoverable issues, fallbacks
- `logger.error()` — failures that result in error responses

### Environment Variables

- Loaded via `python-dotenv` in `main.py` (`.env` first, then `.env.local` with override)
- Never hardcode credentials, account IDs, or bucket names
- Access via `os.environ.get("VAR_NAME", "default")`

---

## Adding a New Model Architecture

1. **Register in `model_def.py`** — Add an entry to the `MODELS` dict in `src/classification/model_def.py` or `src/segmentation/model_def.py`:

```python
MODELS = {
    # ... existing models ...
    "my_new_model": {
        "description": "My custom architecture",
        "parameters": "12.5M",
        "input_size": "224x224",
        "pretrained": True,
        "task": "classification",
        "dimension": "2D",
    },
}
```

2. **Implement the factory function** — In the same `model_def.py`, add a case to the `get_model()` function:

```python
def get_model(model_name, num_classes, pretrained=True):
    if model_name == "my_new_model":
        model = MyNewModel(num_classes=num_classes, pretrained=pretrained)
        return model
    # ... existing cases ...
```

3. **No router/service changes needed** — The `/api/training/models` endpoint dynamically reads the `MODELS` dict via AST parsing.

4. **Rebuild the Docker image** — If using SageMaker:
```bash
./build_and_push.sh <ACCOUNT_ID> <REGION> classification
```

---

## Adding a New API Endpoint

### Step 1: Define the schema (if needed)

Add request/response models to `app/models/schemas.py`:

```python
class MyNewRequest(BaseModel):
    field_one: str
    field_two: int = 10
```

### Step 2: Add service logic

Create or extend a service in `app/services/`:

```python
# app/services/my_service.py
import logging

logger = logging.getLogger(__name__)

class MyService:
    def do_something(self, param: str) -> dict:
        if not param:
            raise ValueError("param is required")
        # business logic here
        return {"result": param}
```

### Step 3: Add the router endpoint

In the appropriate router file:

```python
from app.services.my_service import MyService

my_service = MyService()

@router.post("/my-endpoint")
async def my_endpoint(request: MyNewRequest):
    try:
        result = my_service.do_something(request.field_one)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 4: Register the router (if new file)

In `main.py`:

```python
from app.routers import my_router
app.include_router(my_router.router, prefix="/api/my-prefix", tags=["MyTag"])
```

### Step 5: Add tests

Add a unit test in `tests/unit/` with mocked AWS dependencies and an integration test if the endpoint calls AWS.

---

## Testing

### Unit Tests (pytest + moto)

Unit tests mock AWS services using `moto` and test service/router logic in isolation.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/unit/test_training_service.py

# Run with coverage
coverage run -m pytest && coverage report
```

Test files live in `tests/unit/`. Shared fixtures are in `tests/fixtures/mock_data.py`.

Example test pattern:

```python
import pytest
from moto import mock_s3
import boto3
from app.services.training_service import TrainingService

@mock_s3
def test_import_model():
    # Setup mock S3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    s3.put_object(Bucket="test-bucket", Key="model.pth", Body=b"fake model")

    service = TrainingService()
    # ... test logic ...
```

### Integration Tests

Integration tests hit real AWS resources and require valid credentials.

#### Local inference tests

Test the inference pipeline directly (no SageMaker endpoint required):

```bash
pytest tests/integration/test_local_inference.py -v -s
```

Environment variables:
- `TEST_LOCAL_IMAGE` — path to a test DICOM/PNG/JPG file
- `MODEL_DIR` — directory containing a `.pth` checkpoint

#### Endpoint inference tests

Test a live SageMaker endpoint:

```bash
pytest tests/integration/test_endpoint_inference.py -v -s

# Or use the convenience wrapper
./tests/integration/run_inference_tests.sh /path/to/image.dcm
```

Environment variables:
- `ENDPOINT_NAME` — name of an `InService` endpoint
- `TEST_IMAGE_S3_URI` — S3 URI of a test image
- `TEST_LOCAL_IMAGE` — local file path (optional; generates synthetic if unset)

#### Quick endpoint test (no pytest)

```bash
# Classification
python tests/integration/test_endpoint.py --endpoint my-endpoint --local /path/to/image.dcm

# Segmentation
python test_endpoint_segmentation.py --endpoint my-seg-endpoint \
    --s3-uri s3://bucket/volume.nii.gz
```

#### CLI invocation utility

```bash
python tests/invoke_endpoint.py --image /path/to/image.dcm --endpoint my-endpoint
python tests/invoke_endpoint.py --s3-uri s3://bucket/image.dcm --output result.json
```

---

## Security

### Pre-commit Hook for Secret Detection

Install the pre-commit hook to block commits containing account-specific values:

```bash
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The hook checks staged changes for:
- Hardcoded AWS account IDs
- Account-specific S3 bucket names
- IAM Role ARNs (excluding test fixture account `123456789012`)
- AWS access key patterns (`AKIA...`)

If any are detected, the commit is blocked with a descriptive error message.

### CI/CD Workflow (scrub-check)

The `.github/workflows/scrub-check.yml` GitHub Actions workflow runs on every PR to `main` and performs the same checks as the pre-commit hook across the entire codebase:

1. **Hardcoded AWS account IDs** — scans `.py`, `.js`, `.html`, `.sh`, `.md`, `.yml`, `.yaml`, `.json` files
2. **Hardcoded IAM Role ARNs** — matches `arn:aws:iam::<12-digit>:role/` patterns
3. **Account-specific S3 buckets** — checks for known bucket name patterns
4. **Leaked secrets** — detects AWS access key patterns (`AKIA[0-9A-Z]{16}`)

The workflow excludes the `.github/` directory itself from scanning.

### Frontend XSS Prevention (escapeHtml)

All user-supplied or API-returned data rendered in the DOM is passed through the `escapeHtml()` function defined in `static/js/main.js`:

```javascript
function escapeHtml(str) {
    // Escapes <, >, &, ", ' to prevent XSS
}
```

This is used consistently across all frontend JavaScript files (`training.js`, `deployment.js`, `evaluation.js`, `monitoring.js`, `inference.js`, `fairness.js`, `data.js`).

### Subresource Integrity (SRI)

External CDN resources include `integrity` attributes and `crossorigin="anonymous"`:

```html
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
      rel="stylesheet"
      integrity="sha384-3B6NwesSXE7YJlcLI9RpRqGf2p/EgVH8BgoKTaUrmKNDkHPStTQ3EyoYjCGXaOTS"
      crossorigin="anonymous">

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
        integrity="sha384-e6nUZLBkQ86NJ6TVVKAeSaK8jWa3NhkYWZFomE39AvDbQWeie9PlQqM3pmYW5d1g"
        crossorigin="anonymous"></script>
```

This ensures CDN-served files haven't been tampered with.

### No Hardcoded Credentials

- AWS credentials are loaded from environment variables or instance profiles
- Account-specific config lives in `.env.local` (gitignored)
- The `scrub-check` CI workflow and pre-commit hook enforce this at commit and PR time
- Use `<YOUR_AWS_ACCOUNT_ID>` or `123456789012` as placeholders in documentation and examples

---

## Common Development Tasks

### Running the app locally

```bash
source .venv/bin/activate
python main.py
# → http://localhost:8000
```

### Running with Docker

```bash
docker-compose up --build
```

### Rebuilding SageMaker images after code changes

```bash
# After changing src/classification/
./build_and_push.sh <ACCOUNT_ID> us-east-1 classification

# After changing src/segmentation/
./build_and_push.sh <ACCOUNT_ID> us-east-1 segmentation

# After changing src/inference/classification/
./build_and_push_inference.sh <ACCOUNT_ID> us-east-1 classification

# After changing src/inference/segmentation/
./build_and_push_inference.sh <ACCOUNT_ID> us-east-1 segmentation
```

### Syncing state from AWS

On first launch (or after clearing local data), click **Sync from AWS** in the UI or call:

```bash
# Training jobs
curl -X POST http://localhost:8000/api/training/sagemaker/sync

# Deployments
curl -X POST http://localhost:8000/api/deployment/sync

# Evaluation jobs
curl -X POST http://localhost:8000/api/evaluation/sync
```

### Adding a new JS file

1. Create the file in `static/js/`
2. Include it in `templates/base.html`
3. Use `escapeHtml()` for all dynamic content rendered into the DOM
4. Add SRI `integrity` attributes for any new external CDN resources
