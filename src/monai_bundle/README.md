# MONAI Bundle Support

Generic training / fine-tuning and inference for any model packaged in the
[MONAI Bundle](https://docs.monai.io/en/stable/bundle_intro.html) format, on
SageMaker. Not tied to a single model — it drives a bundle through the bundle's
own configs via `python -m monai.bundle run`.

> ⚠️ **RESEARCH USE ONLY.** Outputs are AI predictions for research/development,
> not clinical diagnoses.

## Contents

| File | Purpose |
|---|---|
| `bundle_utils.py` | Introspect a bundle (locate `configs/`, parse `metadata.json`, find checkpoints, classify train/inference configs). |
| `datalist_gen.py` | Build a MONAI Decathlon train/validation datalist from a `dataset.json`. |
| `train.py` | SageMaker entry point: stage bundle, resolve training config, inject datalist + `config_overrides`, run `monai.bundle run`, collect checkpoint + `result.json`. |
| `infer.py` | SageMaker inference handler (`model_fn`/`input_fn`/`predict_fn`/`output_fn`) running the bundle's inference config. |
| `package.py` | Validate + tar a local bundle into `model.tar.gz` and upload to S3. |

## Two ways to run

### 1. Dashboard (`source_dir`, no custom image)
The dashboard's `/api/monai-bundle/finetune` uses SageMaker's prebuilt PyTorch
container with this directory as `source_dir` and installs MONAI at job start
from `requirements.txt`. No ECR image required.

### 2. Custom image (`Dockerfile.monai-bundle`)
For a self-contained image with MONAI baked in:

```bash
./build_and_push_monai_bundle.sh            # build + push to ECR
./build_and_push_monai_bundle.sh --build    # build only
```

Reads `AWS_ACCOUNT_ID` / region from `.env.local`. Pushes to
`<account>.dkr.ecr.<region>.amazonaws.com/sm-monai-bundle:latest`.

> Large CUDA base layers can stall on proxied/limited networks. If the push
> hangs, build & push from AWS CloudShell or an EC2 instance instead.

## Required data & bundle arrangement (SageMaker channels)

At runtime the entry point reads two SageMaker channels and writes one output.

**`model` channel** (`SM_CHANNEL_MODEL`) — the bundle, unpacked from `model.tar.gz`:

```
configs/
  metadata.json
  train.json            # (or a fine-tune config; pass via train_config)
  inference.json
models/
  model.pt              # pretrained checkpoint
scripts/                # optional bundle python
```

Stage a Model Zoo bundle to S3 with `POST /api/monai-bundle/stage`, or package a
local bundle dir with `package.py`.

**`data` channel** (`SM_CHANNEL_DATA`) — a MONAI **Decathlon** dataset:

```
dataset.json            # contains a "training" list of {image,label} pairs
imagesTr/*.nii.gz
labelsTr/*.nii.gz
```

`dataset.json` example:

```json
{
  "labels": {"0": "background", "1": "pancreas", "2": "cancer"},
  "training": [
    {"image": "./imagesTr/pancreas_001.nii.gz", "label": "./labelsTr/pancreas_001.nii.gz"}
  ]
}
```

`datalist_gen.py` splits `training` into train/validation and normalizes paths.

**Output** (`SM_MODEL_DIR`) — the fine-tuned checkpoint + `result.json`, which
SageMaker bundles into the output `model.tar.gz`.

## Config overrides

Pass bundle-specific config keys through `config_overrides` (JSON) without code
changes, e.g. `{"epochs": 5}`, or for VISTA3D-style softmax finetune
`{"label_mappings": {"default": [[1, 4], [2, 24]]}}`.

## Supported bundles

The dashboard's "MONAI Model Bundle" category is backed by the
`SUPPORTED_BUNDLES` registry in `app/services/monai_bundle_service.py`. Add an
entry there to support a new bundle.
