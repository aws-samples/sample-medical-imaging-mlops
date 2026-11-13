"""
Generic MONAI Bundle training/fine-tuning entry point for SageMaker BYOC.

Drives ANY MONAI bundle through its own training config via
`python -m monai.bundle run`, with dataset paths and arbitrary config keys
passed as pass-through overrides — no model-specific logic.

SageMaker channels:
  SM_CHANNEL_MODEL  -> the bundle (configs/ models/ scripts/)
  SM_CHANNEL_DATA   -> the dataset (Decathlon dataset.json + images/labels)
  SM_MODEL_DIR      -> output; contents are bundled into model.tar.gz

Hyperparameters:
  --train_config        config basename under configs/ (default: bundle default)
  --config_overrides    JSON dict of config keys to set, e.g.
                        '{"epochs": 5, "label_mappings": {"default": [[1,4],[2,24]]}}'
  --generate_datalist   if true, build a Decathlon datalist and inject its path

⚠️ RESEARCH USE ONLY — not for clinical diagnosis.
"""
import argparse
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - argv-list invocations of pip / `python -m monai.bundle` only (no shell)
import sys
import tarfile

from bundle_utils import inspect_bundle, bundle_base_dir
from datalist_gen import build_datalist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CONFIG_EXTS = (".json", ".yaml", ".yml")
# PEP 508 requirement without URL: name[extras] <op><version>[, ...] [; markers]
_REQ_LINE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._ -]+\])?"
    r"(\s*(===|==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+(\s*,\s*(===|==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+)*)?"
    r"(\s*;\s*[A-Za-z0-9_ .<>=!'\"()-]+)?$"
)


def _safe_extract(tar: tarfile.TarFile, dest: str) -> None:
    """Extract a tarball, refusing members that escape `dest` (absolute paths,
    `..`, links pointing outside) and device/fifo members. Uses the stdlib
    'data' extraction filter (PEP 706) when available as a second layer."""
    dest_real = os.path.realpath(dest)
    for m in tar.getmembers():
        target = os.path.realpath(os.path.join(dest_real, m.name))
        if os.path.commonpath([dest_real, target]) != dest_real:
            raise tarfile.TarError(f"Unsafe path in archive: {m.name}")
        if m.issym() or m.islnk():
            link = os.path.realpath(os.path.join(os.path.dirname(target), m.linkname))
            if os.path.commonpath([dest_real, link]) != dest_real:
                raise tarfile.TarError(f"Unsafe link in archive: {m.name} -> {m.linkname}")
        if m.isdev() or m.isfifo():
            raise tarfile.TarError(f"Unsupported member type in archive: {m.name}")
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest_real, filter="data")
    else:  # Python without PEP 706: members were fully validated above
        tar.extractall(dest_real)  # nosec B202 - every member validated above


def _stage_bundle(model_channel: str, work_dir: str) -> str:
    """Copy the bundle from the model channel into a writable working dir.

    SageMaker delivers a plain-prefix channel as-is (it does NOT auto-extract),
    so the channel may contain either an unpacked bundle (configs/, models/, ...)
    or a compressed model.tar.gz. Handle both: copy everything in, then extract
    any *.tar.gz found so inspect_bundle sees the bundle structure.
    """
    if not os.path.isdir(model_channel):
        raise FileNotFoundError(f"Model channel dir not found: {model_channel}")
    shutil.copytree(model_channel, work_dir, dirs_exist_ok=True)

    # Extract any tarballs (e.g. model.tar.gz) delivered in the channel.
    for root, _, files in os.walk(work_dir):
        for fn in files:
            if fn.endswith((".tar.gz", ".tgz", ".tar")):
                tar_path = os.path.join(root, fn)
                try:
                    with tarfile.open(tar_path) as tar:
                        _safe_extract(tar, root)
                    logger.info("Extracted bundle archive %s", tar_path)
                except tarfile.TarError as e:
                    logger.warning("Could not extract %s: %s", tar_path, e)

    return bundle_base_dir(work_dir)


def _ensure_pretrained_checkpoint(bundle_dir: str):
    """If the bundle has a checkpoint but no models/model.pt, stage one there.
    Many training configs load the pretrained weights from models/model.pt."""
    models_dir = os.path.join(bundle_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    target = os.path.join(models_dir, "model.pt")
    if os.path.isfile(target):
        return
    info = inspect_bundle(bundle_dir)
    if info.checkpoints:
        src = info.checkpoints[0]
        if os.path.abspath(src) != os.path.abspath(target):
            shutil.copy2(src, target)
            logger.info("Staged pretrained checkpoint %s -> %s", src, target)


def _resolve_train_config(bundle_dir: str, requested: str | None) -> str:
    info = inspect_bundle(bundle_dir)
    if not info.valid:
        raise ValueError(f"No valid MONAI bundle found under {bundle_dir}")
    if requested:
        if requested not in info.config_files:
            raise FileNotFoundError(
                f"Training config '{requested}' not found. Available: {info.config_files}"
            )
        return requested
    default = info.default_train_config()
    if not default:
        raise ValueError(
            f"No training config found in bundle. Available configs: {info.config_files}. "
            "Pass --train_config explicitly."
        )
    logger.info("Using default training config: %s", default)
    return default


def _apply_overrides(cfg_path: str, overrides: dict):
    """Patch a JSON config file with override keys (only JSON configs are patched;
    YAML configs receive overrides via the monai.bundle CLI instead)."""
    if not cfg_path.endswith(".json") or not overrides:
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.update(overrides)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Applied overrides to %s: %s", os.path.basename(cfg_path), list(overrides))


def _install_bundle_requirements(bundle_dir: str):
    """Install the artifact's CURATED code/requirements.txt (written by staging:
    base deps + bundle metadata packages, with framework-coupled pins like
    torch*/cucim already skipped). The inference DLC installs this automatically
    at endpoint startup, but TRAINING containers do not — e.g. wholeBrainSeg's
    UNEST network needs `timm` at training time. Per-package fallback: only a
    monai failure is fatal; optional packages are skipped with a warning."""
    reqs = os.path.join(bundle_dir, "code", "requirements.txt")
    if not os.path.isfile(reqs):
        return
    # Only plain PEP 508 "name[extras] <specifiers> ; markers" lines are honoured:
    # pip option lines (--index-url, -e, -r, -c ...) and direct URL references
    # in an artifact must not be able to redirect where packages come from.
    with open(reqs, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    lines = [ln for ln in raw if _REQ_LINE_RE.match(ln)]
    rejected = [ln for ln in raw if ln not in lines]
    if rejected:
        logger.warning("Ignoring non-package requirement lines: %s", rejected)
    if not lines:
        return
    safe_reqs = os.path.join(bundle_dir, "code", "requirements.sanitized.txt")
    with open(safe_reqs, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Installing bundle requirements from %s", reqs)
    # argv list (no shell); file contents restricted to package specifiers above.
    pip_argv = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", safe_reqs]
    result = subprocess.run(  # nosec B603 - fixed argv list, no shell  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        pip_argv,
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info("Bundle requirements installed.")
        return
    logger.warning("Bulk install failed, retrying per package:\n%s",
                   (result.stderr or "")[-2000:])
    failed = []
    for line in lines:
        pip_argv = [sys.executable, "-m", "pip", "install", "--no-cache-dir", line]
        r = subprocess.run(  # nosec B603 - argv list; `line` matched _REQ_LINE_RE  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
            pip_argv,
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            failed.append(line)
            logger.warning("Could not install '%s' (continuing)", line)
    hard = [f for f in failed if f.lower().startswith("monai")]
    if hard:
        raise RuntimeError(f"Failed to install required package(s): {hard}")
    if failed:
        logger.warning("Skipped optional/unavailable packages: %s", failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str,
                        default=os.environ.get("SM_HP_TRAIN_CONFIG") or None)
    parser.add_argument("--config_overrides", type=str,
                        default=os.environ.get("SM_HP_CONFIG_OVERRIDES", "{}"))
    parser.add_argument("--generate_datalist", type=str,
                        default=os.environ.get("SM_HP_GENERATE_DATALIST", "true"))
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    # Fall back to the fixed SageMaker container paths, which are always present
    # regardless of whether the training toolkit set SM_CHANNEL_* env vars
    # (e.g. when launched via a bare ContainerEntrypoint).
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
    _install_bundle_requirements(bundle_dir)
    _ensure_pretrained_checkpoint(bundle_dir)

    train_config = _resolve_train_config(bundle_dir, args.train_config)
    cfg_path = os.path.join(bundle_dir, "configs", train_config)

    dataset_dir = args.data_dir
    cli_overrides = []

    if str(args.generate_datalist).lower() in ("1", "true", "yes"):
        datalist_path = os.path.join(work_dir, "datalist.json")
        datalist = build_datalist(dataset_dir, args.val_fraction, args.seed)
        with open(datalist_path, "w", encoding="utf-8") as f:
            json.dump(datalist, f, indent=2)
        overrides.setdefault("data_list_file_path", datalist_path)
        overrides.setdefault("dataset_dir", dataset_dir)
        logger.info("Generated datalist: %s (train=%d val=%d)",
                    datalist_path, len(datalist["training"]), len(datalist["validation"]))

    _apply_overrides(cfg_path, overrides)

    # Detect the config's training run id — monai.bundle defaults to 'run', but
    # many bundles name their section differently (e.g. wholeBrainSeg: 'training',
    # HoVer-Net: 'train'). Pick the first executable section present.
    run_id = "run"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            _cfg = json.load(f)
        for cand in ("run", "training", "train", "run_training"):
            if cand in _cfg:
                run_id = cand
                break
    except Exception as e:  # noqa: BLE001 - YAML/unusual config; keep default
        logger.warning("Could not detect run_id in %s: %s", train_config, e)

    cmd = [
        sys.executable, "-m", "monai.bundle", "run",
        "--config_file", os.path.join("configs", train_config),
        "--dataset_dir", dataset_dir,
    ]
    if run_id != "run":
        cmd += ["--run_id", run_id]
    cmd += cli_overrides
    logger.info("Running (run_id=%s): %s (cwd=%s)", run_id, " ".join(cmd), bundle_dir)
    # argv list (no shell); train_config was validated against the bundle's own
    # config files by _resolve_train_config.
    subprocess.run(cmd, cwd=bundle_dir, check=True)  # nosec B603 - argv list, no shell  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit

    # Collect produced checkpoint(s) + result.json into SM_MODEL_DIR.
    # ── Assemble a servable bundle into SM_MODEL_DIR (Option B) ───────────────
    # SageMaker bundles SM_MODEL_DIR into the output model.tar.gz. The serving
    # handler lives in the INFERENCE IMAGE (not here), so the artifact carries
    # ONLY the bundle (configs/ + models/ with the fine-tuned weights).
    os.makedirs(args.model_dir, exist_ok=True)
    post = inspect_bundle(bundle_dir)

    # Copy the full bundle (configs/, models/, scripts/, metadata, ...).
    for name in os.listdir(bundle_dir):
        src = os.path.join(bundle_dir, name)
        dst = os.path.join(args.model_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # Promote the fine-tuned checkpoint to models/model.pt so the inference
    # config's default checkpoint path picks up the trained weights.
    produced = [c for c in post.checkpoints if os.path.basename(c) != "model.pt"]
    models_out = os.path.join(args.model_dir, "models")
    os.makedirs(models_out, exist_ok=True)
    if produced:
        shutil.copy2(produced[0], os.path.join(models_out, os.path.basename(produced[0])))
        logger.info("Fine-tuned checkpoint -> models/%s", os.path.basename(produced[0]))

    # Stage the serving handler into code/ so the deployed artifact is servable
    # by the inference DLC (which discovers the handler at /opt/ml/model/code/).
    here = os.path.dirname(os.path.abspath(__file__))
    code_out = os.path.join(args.model_dir, "code")
    os.makedirs(code_out, exist_ok=True)
    for fn in ("inference.py", "bundle_utils.py"):
        src = os.path.join(here, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(code_out, fn))
    reqs = os.path.join(here, "requirements-inference.txt")
    if os.path.exists(reqs):
        shutil.copy2(reqs, os.path.join(code_out, "requirements.txt"))

    result = {
        "train_config": train_config,
        "config_overrides": overrides,
        "dataset_dir": dataset_dir,
        "task": post.task,
        "produced_checkpoints": [os.path.basename(c) for c in produced],
        "disclaimer": "RESEARCH USE ONLY — not for clinical diagnosis.",
    }
    with open(os.path.join(args.model_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Training complete. result=%s", result)


if __name__ == "__main__":
    main()
