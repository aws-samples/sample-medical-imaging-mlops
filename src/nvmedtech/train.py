"""
NV-Segment-CT (VISTA-3D) training / fine-tuning entry point for SageMaker BYOC.

Dedicated to the NVIDIA-Medtech NV-Segment-CT bundle
(https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/main/NV-Segment-CT),
which follows the MONAI Model Zoo bundle format. The dedicated image
(Dockerfile.nvmedtech) bakes the bundle's framework requirements
(MONAI 1.4 on the PyTorch 2.4 training DLC), so no runtime installs are needed.

Fine-tuning follows the bundle's own recipe: a composed config
    ['configs/train.json', 'configs/train_continual.json']
with `label_mappings` mapping dataset labels to VISTA-3D global class indices
(e.g. MSD Task09 spleen label 1 -> VISTA class 3 — the bundle's shipped default).

SageMaker channels:
  SM_CHANNEL_MODEL  -> the bundle (configs/ models/ scripts/), tar or unpacked
  SM_CHANNEL_DATA   -> the dataset (e.g. Task09_Spleen: imagesTr/ labelsTr/)
  SM_MODEL_DIR      -> output; contents are bundled into model.tar.gz

Hyperparameters (ContainerArguments):
  --train_config        comma-separated configs (default:
                        train.json,train_continual.json)
  --config_overrides    JSON dict merged into the LAST config, e.g.
                        '{"epochs": 2, "label_mappings": {"default": [[1, 3]]}}'
  --generate_datalist   default false — the bundle ships its own datalists
                        (e.g. msd_task09_spleen_folds.json). Set true to build a
                        Decathlon datalist from the data channel's dataset.json.

If models/model.pt is missing from the bundle, the pretrained VISTA-3D
checkpoint is fetched from Hugging Face using the repo/file the bundle's own
inference.json declares (huggingface_repo_id / huggingface_checkpoint_file).

⚠️ RESEARCH USE ONLY — not for clinical diagnosis.
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CONFIGS = "train.json,train_continual.json"


def _stage_bundle(model_channel: str, work_dir: str) -> str:
    """Copy the bundle from the model channel into a writable working dir and
    extract any model.tar.gz. Returns the bundle root (dir containing configs/)."""
    if not os.path.isdir(model_channel):
        raise FileNotFoundError(f"Model channel dir not found: {model_channel}")
    shutil.copytree(model_channel, work_dir, dirs_exist_ok=True)

    for root, _, files in os.walk(work_dir):
        for fn in files:
            if fn.endswith((".tar.gz", ".tgz", ".tar")):
                tar_path = os.path.join(root, fn)
                try:
                    with tarfile.open(tar_path) as tar:
                        # Validate members to prevent path traversal.
                        for m in tar.getmembers():
                            target = os.path.realpath(os.path.join(root, m.name))
                            if not target.startswith(os.path.realpath(root)):
                                raise ValueError(f"Unsafe path in archive: {m.name}")
                        tar.extractall(root)
                    logger.info("Extracted bundle archive %s", tar_path)
                except tarfile.TarError as e:
                    logger.warning("Could not extract %s: %s", tar_path, e)

    # Find the dir that holds configs/metadata.json.
    for root, dirs, _ in os.walk(work_dir):
        if "configs" in dirs and os.path.isfile(os.path.join(root, "configs", "metadata.json")):
            return root
    raise ValueError(f"No MONAI bundle (configs/metadata.json) found under {work_dir}")


def _ensure_checkpoint(bundle_dir: str):
    """Ensure models/model.pt exists; fetch the pretrained VISTA-3D checkpoint
    from Hugging Face if the bundle's config declares one."""
    target = os.path.join(bundle_dir, "models", "model.pt")
    if os.path.isfile(target):
        logger.info("Checkpoint present: %s", target)
        return
    hf_repo, hf_file = None, None
    for cfg_name in ("inference.json", "train.json"):
        fp = os.path.join(bundle_dir, "configs", cfg_name)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if cfg.get("huggingface_repo_id") and cfg.get("huggingface_checkpoint_file"):
            hf_repo, hf_file = cfg["huggingface_repo_id"], cfg["huggingface_checkpoint_file"]
            break
    if not (hf_repo and hf_file):
        raise FileNotFoundError(
            "models/model.pt missing and the bundle declares no Hugging Face "
            "checkpoint (huggingface_repo_id/huggingface_checkpoint_file)."
        )
    from huggingface_hub import hf_hub_download
    logger.info("Fetching pretrained checkpoint %s from %s", hf_file, hf_repo)
    got = hf_hub_download(repo_id=hf_repo, filename=hf_file)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(got, target)
    logger.info("Checkpoint staged at models/model.pt")


def _resolve_train_configs(bundle_dir: str, requested: str) -> list:
    names = [n.strip() for n in requested.split(",") if n.strip()]
    configs_dir = os.path.join(bundle_dir, "configs")
    for n in names:
        if not os.path.isfile(os.path.join(configs_dir, n)):
            available = sorted(os.listdir(configs_dir))
            raise FileNotFoundError(f"Training config '{n}' not found. Available: {available}")
    return names


def _apply_overrides(cfg_path: str, overrides: dict):
    """Merge override keys into a JSON config (later config wins in the
    monai.bundle composition, so overrides go into the last one)."""
    if not overrides:
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.update(overrides)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Applied overrides to %s: %s", os.path.basename(cfg_path), list(overrides))


def _find_checkpoints(bundle_dir: str) -> list:
    out = []
    for root, _, files in os.walk(bundle_dir):
        for fn in files:
            if fn.endswith(".pt"):
                out.append(os.path.join(root, fn))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=DEFAULT_TRAIN_CONFIGS)
    parser.add_argument("--config_overrides", type=str, default="{}")
    parser.add_argument("--generate_datalist", type=str, default="false")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--data_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data"))
    parser.add_argument("--bundle_channel", type=str,
                        default=os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model"))
    args, _ = parser.parse_known_args()
    logger.info("Args: %s", args)

    try:
        overrides = json.loads(args.config_overrides)
        if not isinstance(overrides, dict):
            raise ValueError("config_overrides must be a JSON object")
    except json.JSONDecodeError as e:
        raise ValueError(f"--config_overrides must be valid JSON: {e}")

    work_dir = "/opt/ml/bundle_work"
    bundle_dir = _stage_bundle(args.bundle_channel, work_dir)
    _ensure_checkpoint(bundle_dir)

    train_configs = _resolve_train_configs(bundle_dir, args.train_config)
    last_cfg_path = os.path.join(bundle_dir, "configs", train_configs[-1])

    dataset_dir = args.data_dir

    if str(args.generate_datalist).lower() in ("1", "true", "yes"):
        # Build a Decathlon datalist from the data channel's dataset.json for
        # datasets the bundle has no shipped datalist for.
        from datalist_gen import build_datalist
        datalist_path = os.path.join(work_dir, "datalist.json")
        datalist = build_datalist(dataset_dir, args.val_fraction, args.seed)
        with open(datalist_path, "w", encoding="utf-8") as f:
            json.dump(datalist, f, indent=2)
        overrides.setdefault("data_list_file_path", datalist_path)
        logger.info("Generated datalist: %s (train=%d val=%d)", datalist_path,
                    len(datalist["training"]), len(datalist["validation"]))

    _apply_overrides(last_cfg_path, overrides)

    pre_ckpts = set(_find_checkpoints(bundle_dir))

    config_file_arg = (
        "['" + "','".join(os.path.join("configs", c) for c in train_configs) + "']"
        if len(train_configs) > 1 else os.path.join("configs", train_configs[0])
    )
    cmd = [
        sys.executable, "-m", "monai.bundle", "run",
        "--config_file", config_file_arg,
        "--dataset_dir", dataset_dir,   # CLI overrides win over config values
    ]
    logger.info("Running: %s (cwd=%s)", " ".join(cmd), bundle_dir)
    subprocess.run(cmd, cwd=bundle_dir, check=True)

    # ── Assemble the output artifact: full bundle + fine-tuned weights ───────
    os.makedirs(args.model_dir, exist_ok=True)
    for name in os.listdir(bundle_dir):
        src = os.path.join(bundle_dir, name)
        dst = os.path.join(args.model_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    produced = [c for c in _find_checkpoints(bundle_dir) if c not in pre_ckpts]
    result = {
        "bundle": "NV-Segment-CT (VISTA-3D)",
        "train_config": train_configs,
        "config_overrides": overrides,
        "dataset_dir": dataset_dir,
        "produced_checkpoints": [os.path.relpath(c, bundle_dir) for c in produced],
        "disclaimer": "RESEARCH USE ONLY — not for clinical diagnosis.",
    }
    with open(os.path.join(args.model_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Training complete. result=%s", result)


if __name__ == "__main__":
    main()
