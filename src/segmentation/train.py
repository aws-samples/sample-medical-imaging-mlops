"""
Training script for 3D medical image segmentation on SageMaker BYOC.
Data channel: SM_CHANNEL_DATA with train/ val/ subfolders of NIfTI volumes.
Binary segmentation — out_channels is always 1.
"""
import os
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.inferers import SlidingWindowInferer

from model_def import ModelDef
from utils import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",     type=str,   default=os.environ.get("SM_MODEL_DIR", "./output"))
parser.add_argument("--model_name",    type=str,   default="segresnet")
parser.add_argument("--task_type",     type=str,   default="segmentation")
parser.add_argument("--dimension",     type=str,   default="3D")
parser.add_argument("--batch_size",    type=int,   default=2)
parser.add_argument("--epochs",        type=int,   default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--val_interval",  type=int,   default=2)
parser.add_argument("--num_classes",   type=int,   default=0,
                    help="Number of classes INCLUDING background. "
                         "0 or 1 = binary (sigmoid+Dice). >1 = multi-class "
                         "(softmax+DiceCE). If 0, derived from the label volumes.")
parser.add_argument("--roi_size",      type=str,   default="96,96,96",
                    help="Training patch size as comma-separated ints "
                         "(e.g. '96,96,96'). Patches are foreground-biased crops "
                         "at native resolution; inference slides this ROI over "
                         "the full volume. Each dim should be divisible by 16.")
parser.add_argument("--num_samples",   type=int,   default=2,
                    help="Foreground/background patches sampled per volume.")
parser.add_argument("--checkpoint_s3_uri", type=str, default="",
                    help="S3 URI of model.tar.gz to resume from (fine-tuning)")
args, _ = parser.parse_known_args()
logger.info("Args: %s", args)

# ── MLflow setup ──────────────────────────────────────────────────────────────
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
use_mlflow = False
if mlflow_uri:
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        experiment_name = f"segmentation-{args.model_name}"
        mlflow.set_experiment(experiment_name)
        use_mlflow = True
        logger.info("MLflow tracking enabled: %s  experiment: %s", mlflow_uri, experiment_name)
    except Exception as e:
        logger.warning("MLflow setup failed, continuing without it: %s", e)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=args.model_dir)

data_root  = os.environ.get("SM_CHANNEL_DATA", "./data")
train_path = os.path.join(data_root, "train")
val_path   = os.path.join(data_root, "val")
if not os.path.isdir(val_path):
    val_path = os.path.join(data_root, "valid")

logger.info("Train: %s  |  Val: %s", train_path, val_path)

# Parse ROI (training patch) size, e.g. "96,96,96" -> (96, 96, 96).
try:
    ROI_SIZE = tuple(int(x) for x in str(args.roi_size).split(",") if x.strip())
    if len(ROI_SIZE) != 3:
        raise ValueError
except Exception:
    logger.warning("Invalid --roi_size %r; falling back to 96,96,96", args.roi_size)
    ROI_SIZE = (96, 96, 96)
logger.info("Training ROI (patch) size: %s  |  num_samples/vol: %d", ROI_SIZE, args.num_samples)

train_loader = create_data_loaders(train_path, args.batch_size, training=True,
                                   roi_size=ROI_SIZE, num_samples=args.num_samples)
val_loader   = create_data_loaders(val_path,   args.batch_size, training=False)


def _derive_num_classes(folder: str) -> int:
    """Infer class count (incl. background) as max(label value)+1 over the label
    volumes under <folder>/<subject>/label.nii.gz. Scans up to 50 volumes — more
    than enough to see every class in practice — and is only used when the user
    does not pass --num_classes explicitly."""
    import glob
    import numpy as np
    import nibabel as nib
    label_paths = sorted(glob.glob(os.path.join(folder, "*", "label.nii.gz")))
    if not label_paths:
        return 1
    max_label = 0
    for p in label_paths[:50]:
        try:
            arr = np.asanyarray(nib.load(p).dataobj)
            m = int(arr.max())
            if m > max_label:
                max_label = m
        except Exception as e:  # unreadable file — skip, keep going
            logger.warning("Could not read %s while deriving num_classes: %s", p, e)
    return max_label + 1


# Resolve number of classes (INCLUDING background).
#   >1  -> multi-class: softmax + DiceCELoss(to_onehot_y) + argmax/one-hot metric
#   <=1 -> binary (unchanged legacy path): sigmoid + DiceLoss + threshold
if args.num_classes and args.num_classes > 0:
    NUM_CLASSES = args.num_classes
    logger.info("num_classes=%d (from --num_classes)", NUM_CLASSES)
else:
    NUM_CLASSES = _derive_num_classes(train_path)
    logger.info("num_classes=%d (auto-derived from label volumes)", NUM_CLASSES)

MULTICLASS = NUM_CLASSES > 1

model     = ModelDef(NUM_CLASSES, args.model_name).get_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
if MULTICLASS:
    # to_onehot_y: integer label map (B,1,H,W,D) -> one-hot internally.
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
else:
    loss_fn = DiceLoss(sigmoid=True)
logger.info("Model: %s  |  classes: %d  |  mode: %s  |  Params: %s",
            args.model_name, NUM_CLASSES, "multiclass" if MULTICLASS else "binary",
            sum(p.numel() for p in model.parameters()))

# ── Checkpoint resume (fine-tuning) ──────────────────────────────────────────
start_epoch = 1
fine_tuned_from = ""

if args.checkpoint_s3_uri:
    import io, tarfile, tempfile, boto3
    logger.info("Fine-tuning: loading checkpoint from %s", args.checkpoint_s3_uri)
    try:
        path   = args.checkpoint_s3_uri.replace("s3://", "")
        bucket = path.split("/")[0]
        key    = "/".join(path.split("/")[1:])
        s3     = boto3.client("s3")
        _fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz")
        os.close(_fd)
        s3.download_file(bucket, key, tmp_path)
        with tarfile.open(tmp_path, "r:gz") as tar:
            ckpt_name = "best_model.pth" if "best_model.pth" in tar.getnames() else "model.pth"
            f    = tar.extractfile(tar.getmember(ckpt_name))
            ckpt = torch.load(io.BytesIO(f.read()), map_location=device, weights_only=True)
        import os as _os; _os.unlink(tmp_path)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch     = ckpt.get("epoch", 0) + 1
        fine_tuned_from = args.checkpoint_s3_uri
        logger.info("Resumed from %s (epoch %d, val_dice=%.4f)",
                    ckpt_name, ckpt.get("epoch", 0), ckpt.get("val_dice", 0))
    except Exception as e:
        logger.warning("Could not load checkpoint, starting from scratch: %s", e)

if MULTICLASS:
    # pred: (C,H,W,D) logits -> softmax -> argmax -> one-hot for Dice.
    post_pred  = Compose([Activations(softmax=True),
                          AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=NUM_CLASSES)])
    # Exclude background so the dominant class doesn't inflate the score.
    dice_metric = DiceMetric(include_background=False, reduction="mean")
else:
    post_pred  = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = None
    dice_metric = DiceMetric(include_background=True, reduction="mean")

# Validation runs on FULL native volumes via sliding window over the training ROI
# (patch training => the net only ever saw ROI_SIZE inputs). roi_size here must
# match the training patch size and the inference handler's SlidingWindowInferer.
val_inferer = SlidingWindowInferer(roi_size=ROI_SIZE, sw_batch_size=1, overlap=0.25)

# ── Training loop ─────────────────────────────────────────────────────────────
best_dice = -1.0

if use_mlflow:
    mlflow.start_run()
    mlflow.log_params({
        "model_name":      args.model_name,
        "task_type":       args.task_type,
        "dimension":       args.dimension,
        "batch_size":      args.batch_size,
        "epochs":          args.epochs,
        "learning_rate":   args.learning_rate,
        "val_interval":    args.val_interval,
        "device":          str(device),
        "fine_tuned_from": fine_tuned_from or "none",
    })

for epoch in range(start_epoch, args.epochs + 1):
    model.train()
    epoch_loss, steps = 0.0, 0
    for batch in train_loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        steps      += 1
    train_loss = epoch_loss / steps
    logger.info("Epoch %d/%d  train_loss=%.4f", epoch, args.epochs, train_loss)
    writer.add_scalar("Loss/train", train_loss, epoch)
    if use_mlflow:
        mlflow.log_metric("train_loss", train_loss, step=epoch)

    if epoch % args.val_interval == 0:
        model.eval()
        from monai.data import decollate_batch
        with torch.no_grad():
            for batch in val_loader:
                images  = batch["image"].to(device)
                masks   = batch["mask"].to(device)
                logits  = val_inferer(images, model)   # sliding window over full volume
                preds   = [post_pred(p) for p in decollate_batch(logits)]
                if MULTICLASS:
                    labels = [post_label(m) for m in decollate_batch(masks)]
                else:
                    labels = decollate_batch(masks)
                dice_metric(y_pred=preds, y=labels)
        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        logger.info("Epoch %d/%d  val_dice=%.4f", epoch, args.epochs, val_dice)
        writer.add_scalar("Dice/val", val_dice, epoch)
        if use_mlflow:
            mlflow.log_metric("val_dice", val_dice, step=epoch)
        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(args.model_dir, "best_model.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice}, best_path)
            logger.info("Best model saved (val_dice=%.4f)", best_dice)
            if use_mlflow:
                mlflow.log_artifact(best_path)

final_path = os.path.join(args.model_dir, "model.pth")
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": args.epochs}, final_path)

# ── result.json ───────────────────────────────────────────────────────────────
result = {
    "model_name":       args.model_name,
    "task_type":        args.task_type,
    "dimension":        args.dimension,
    "num_classes":      NUM_CLASSES,
    "multiclass":       MULTICLASS,
    "roi_size":         list(ROI_SIZE),
    "final_epoch":      args.epochs,
    "best_val_dice":    round(best_dice, 4),
    "learning_rate":    args.learning_rate,
    "batch_size":       args.batch_size,
    "data_s3_uri":      os.environ.get("SM_CHANNEL_DATA", data_root),
    "fine_tuned_from":  fine_tuned_from or None,
}
result_path = os.path.join(args.model_dir, "result.json")
with open(result_path, "w", encoding="utf-8") as f:
    import json as _json
    _json.dump(result, f, indent=2)
logger.info("Saved result.json: %s", result)

if use_mlflow:
    mlflow.log_metric("best_val_dice", best_dice)
    mlflow.log_artifact(final_path)
    mlflow.log_artifact(result_path)
    mlflow.end_run()

writer.close()
logger.info("Training complete. Best val_dice=%.4f", best_dice)

# ── Copy inference artifacts into model_dir ───────────────────────────────────
# SageMaker bundles everything in model_dir into model.tar.gz.
# The PyTorch serving container discovers custom handlers at code/inference.py.
import shutil as _shutil

def _copy_inference_artifacts(src_dir: str, dest_dir: str, fallback_dir: str = "/opt/program") -> None:
    code_dir = os.path.join(dest_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    # inference.py and requirements-inference.txt are copied alongside train.py
    # by the Dockerfile, so they follow wherever SageMaker relocates the script.
    # If not found in src_dir, fall back to /opt/program/ (original Docker location).
    file_map = {
        "inference.py": "inference.py",
        "requirements-inference.txt": "requirements.txt",
        "model_def.py": "model_def.py",
    }
    for src_name, dest_name in file_map.items():
        src = os.path.join(src_dir, src_name)
        if not os.path.exists(src):
            src = os.path.join(fallback_dir, src_name)
        if os.path.exists(src):
            _shutil.copy2(src, os.path.join(code_dir, dest_name))
            logger.info("Copied %s → code/%s", src, dest_name)
        else:
            logger.warning("Not found, skipping: %s (checked %s and %s)", src_name, src_dir, fallback_dir)

_copy_inference_artifacts(
    src_dir=os.path.dirname(os.path.abspath(__file__)),
    dest_dir=args.model_dir,
    fallback_dir="/opt/program",
)
