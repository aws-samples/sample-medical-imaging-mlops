# NV-Segment-CT (VISTA-3D) — dedicated SageMaker integration

Training / fine-tuning support for the **NVIDIA-Medtech NV-Segment-CT** bundle
([GitHub](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/main/NV-Segment-CT) ·
[Hugging Face](https://huggingface.co/nvidia/NV-Segment-CT)), which follows the
MONAI Model Zoo bundle format. **CT only** — the MR sibling model is
commercially licensed and intentionally excluded.

> ⚠️ **RESEARCH USE ONLY** — outputs are not clinical diagnoses.

## Why a dedicated image (not the shared `sm-monai-bundle`)

VISTA-3D requires **MONAI 1.4 on PyTorch 2.4** — a materially different
framework stack than the shared image (MONAI 1.3 / torch 2.9). Per the
import-monai-bundle skill's rule, a hard framework gap can't be fixed by
pip-at-startup, so this bundle gets its own image built from the matching
**PyTorch 2.4 (cu124) training DLC** with the full VISTA dependency stack baked
in (`cucim` deliberately excluded: no installable wheel in the container, and
it's optional GPU acceleration).

One image covers **both training and fine-tuning** — they are the same process:
`train.py` composes the bundle's own recipe
`['configs/train.json', 'configs/train_continual.json']` and fine-tunes from
the pretrained checkpoint delivered in the model channel.

## Layout

| Piece | Path |
|---|---|
| Bundle (configs/scripts/docs) | `NVIDIA-MedTech/NV-Segment-CT/` |
| Entry point (train/fine-tune) | `src/nvmedtech/train.py` |
| Stage to S3 | `src/nvmedtech/package.py` |
| Image | `Dockerfile.nvmedtech` → ECR `sm-nvmedtech` |
| Build script | `build_and_push_nvmedtech.sh` |

The 872 MB pretrained checkpoint is **not** kept in the repo: `package.py`
fetches it from Hugging Face (`nvidia/NV-Segment-CT`, as declared by the
bundle's own `inference.json`) into `models/model.pt` when staging, so the
`model.tar.gz` artifact is self-contained.

## Workflow

```bash
# 1. Build & push the image (on the docker build host)
bash build_and_push_nvmedtech.sh

# 2. Stage the bundle to S3 (downloads the checkpoint from HF once)
python -m src.nvmedtech.package \
    --s3_uri s3://<bucket>/<prefix>/NV-Segment-CT/model.tar.gz

# 3. Launch fine-tuning (SageMaker training job; billable)
#    model channel = the staged model.tar.gz, data channel = your dataset
#    e.g. MSD Task09_Spleen — the bundle ships a folds datalist and the
#    label mapping {"default": [[1, 3]]} (spleen label 1 -> VISTA class 3).
```

Hyperparameters (container arguments): `--train_config`
(default `train.json,train_continual.json`), `--config_overrides`
(JSON, merged into the last config — e.g. `{"epochs": 2}`),
`--generate_datalist` (default `false`; the bundle ships its own datalists).

For other datasets, override `label_mappings` to map your label indices to
VISTA-3D's global classes (`configs/label_dict.json` has the full 127-class map)
and either provide a Decathlon `dataset.json` (with `--generate_datalist true`)
or your own datalist via `data_list_file_path`.

## Not yet included

- **Inference/deploy image** — deploying a fine-tuned NV-Segment-CT needs a
  matching dedicated *inference* image (PyTorch 2.4 inference DLC + MONAI 1.4),
  which is not part of this folder yet. The shared `sm-monai-bundle-inference`
  image (torch 2.6/MONAI 1.3) will not serve this bundle.
- Dashboard wiring — this integration is CLI-first; the MONAI Bundle tab's
  generic flow targets Model Zoo bundles on the shared images.
