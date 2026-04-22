# src/classification

Training scripts and utilities for 2D medical image classification on SageMaker BYOC.

## Files

| File | Description |
|---|---|
| `train.py` | SageMaker entry point — parses hyperparameters, runs training loop, saves checkpoints |
| `model_def.py` | Model factory — returns ResNet50, VGG16, EfficientNet, DenseNet121, ViT, or custom |
| `utils.py` | Dataset scanning, label mapping, and DataLoader construction |
| `label_dict.json` | Default label map for the bundled vertebral X-ray dataset (8 classes) |
| `requirements.txt` | Python dependencies for the BYOC container |

---

## Data Format

```
data/
├── train/
│   ├── ClassName_A/   # one subfolder per class
│   │   ├── img001.jpg
│   │   └── ...
│   └── ClassName_B/
│       └── ...
└── val/               # (or test/)
    ├── ClassName_A/
    └── ClassName_B/
```

SageMaker mounts the S3 data channel at `SM_CHANNEL_DATA`; the script expects `train/` and `val/` (or `test/`) subdirectories inside it.

---

## Label Mapping

Label assignment is **dynamic** — built from the sorted folder names found in the training split. The mapping is then reused for the validation split to guarantee consistency.

### `generate_list_labels(data_folder, label_dict=None)`

| Argument | Type | Description |
|---|---|---|
| `data_folder` | `str` | Path to a split directory containing class subdirectories |
| `label_dict` | `dict \| None` | Existing `{class_name: int}` map to reuse; built from scratch if `None` |

Returns `(image_list, label_list, num_classes, label_dict)`.

- If `label_dict=None`, labels are assigned by sorted folder order and logged at `INFO`.
- If a class in the folder is absent from the provided dict, it is assigned the next available integer and logged at `WARNING`.

### `create_data_loaders(data_path, batch_size, label_dict=None)`

Convenience wrapper that applies MONAI transforms and returns `(DataLoader, num_classes, label_dict)`.

Pass the `label_dict` returned from the training loader when creating the validation loader:

```python
train_loader, num_classes, label_dict = create_data_loaders(train_path, batch_size)
val_loader,   _,           _          = create_data_loaders(val_path,   batch_size, label_dict)
```

The final `label_dict` is saved to `label_dict.json` in `SM_MODEL_DIR` alongside the model checkpoint so it can be used at inference time.

---

## Transforms

| Step | MONAI transform |
|---|---|
| Load | `LoadImage(image_only=True)` |
| Channel | `EnsureChannelFirst()` |
| Resize | `Resize(spatial_size=(256, 256, 1))` |
| Normalise | `ScaleIntensity()` |

---

## Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `densenet121` | Architecture to train |
| `--epochs` | `1` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--learning_rate` | `0.001` | Adam learning rate |
| `--val_interval` | `1` | Run validation every N epochs |
| `--task_type` | `classification` | Task identifier (passed through) |
| `--dimension` | `2D` | Input dimensionality (passed through) |

> **Note:** `num_classes` is no longer passed as a hyperparameter. The training script derives it dynamically from the class subdirectories found in the training data split via `generate_list_labels`.
