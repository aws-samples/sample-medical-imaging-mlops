"""
Generic MONAI Bundle introspection.

A MONAI Bundle is a self-describing directory following a standard layout:

    <bundle>/
        configs/
            metadata.json        # task, network, inputs/outputs, version
            inference.json        # (or inference.yaml) inference workflow
            train.json            # (or train.yaml) training workflow  [optional]
            *.json / *.yaml       # additional/overlay configs
        models/
            model.pt              # checkpoint(s)
        scripts/                  # optional bundle-specific python
        docs/

All bundles are driven by the same CLI:

    python -m monai.bundle run --config_file=configs/<name>.json

This module discovers those pieces so a single code path can drive ANY bundle
without hardcoding one model.

RESEARCH USE ONLY.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_EXTS = (".json", ".yaml", ".yml")
_CKPT_EXTS = (".pt", ".pth", ".ts", ".safetensors")


@dataclass
class BundleInfo:
    """Discovered structure of a MONAI bundle."""
    root: str
    configs_dir: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    config_files: list = field(default_factory=list)   # basenames in configs/
    checkpoints: list = field(default_factory=list)     # abs paths under models/
    train_configs: list = field(default_factory=list)   # config basenames that look like training
    inference_configs: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.configs_dir is not None and bool(self.config_files)

    @property
    def task(self) -> Optional[str]:
        # MONAI metadata commonly declares task under "task".
        return self.metadata.get("task")

    def default_inference_config(self) -> Optional[str]:
        for name in ("inference.json", "inference.yaml", "inference.yml"):
            if name in self.config_files:
                return name
        return self.inference_configs[0] if self.inference_configs else None

    def default_train_config(self) -> Optional[str]:
        for name in ("train.json", "train.yaml", "train.yml"):
            if name in self.config_files:
                return name
        return self.train_configs[0] if self.train_configs else None


def _find_configs_dir(root: str) -> Optional[str]:
    """Locate the bundle's configs/ dir (bundle may be nested one level down)."""
    direct = os.path.join(root, "configs")
    if os.path.isdir(direct):
        return direct
    # Some repos nest the bundle in a subfolder (e.g. NV-Segment-CT/configs).
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if os.path.isdir(sub) and os.path.isdir(os.path.join(sub, "configs")):
            return os.path.join(sub, "configs")
    return None


def inspect_bundle(root: str) -> BundleInfo:
    """Introspect a MONAI bundle rooted at `root`."""
    info = BundleInfo(root=root)

    configs_dir = _find_configs_dir(root)
    if configs_dir:
        info.configs_dir = configs_dir
        info.config_files = sorted(
            f for f in os.listdir(configs_dir)
            if f.endswith(_CONFIG_EXTS)
        )
        meta_path = os.path.join(configs_dir, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    info.metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not parse metadata.json: %s", e)

        for name in info.config_files:
            lname = name.lower()
            if "train" in lname or "finetune" in lname or "continual" in lname:
                info.train_configs.append(name)
            if "infer" in lname or "eval" in lname:
                info.inference_configs.append(name)

    bundle_base = os.path.dirname(configs_dir) if configs_dir else root
    models_dir = os.path.join(bundle_base, "models")
    search_root = models_dir if os.path.isdir(models_dir) else bundle_base
    for cur, _, files in os.walk(search_root):
        for fn in files:
            if fn.endswith(_CKPT_EXTS):
                info.checkpoints.append(os.path.join(cur, fn))
    info.checkpoints.sort()

    logger.info(
        "Bundle @ %s: valid=%s configs=%s train=%s infer=%s ckpts=%d task=%s",
        root, info.is_valid, info.config_files, info.train_configs,
        info.inference_configs, len(info.checkpoints), info.task,
    )
    return info


def bundle_base_dir(root: str) -> str:
    """Return the directory that contains configs/ (the dir to run monai.bundle from)."""
    configs_dir = _find_configs_dir(root)
    return os.path.dirname(configs_dir) if configs_dir else root
