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
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete

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

train_loader = create_data_loaders(train_path, args.batch_size, training=True)
val_loader   = create_data_loaders(val_path,   args.batch_size, training=False)

# Binary segmentation — always 1 output channel
NUM_CLASSES = 1
model     = ModelDef(NUM_CLASSES, args.model_name).get_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
loss_fn   = DiceLoss(sigmoid=True)
logger.info("Model: %s  |  Params: %s", args.model_name, sum(p.numel() for p in model.parameters()))

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
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.flush()
            tmp_path = tmp.name
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

post_trans  = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
dice_metric = DiceMetric(include_background=True, reduction="mean")

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
        with torch.no_grad():
            for batch in val_loader:
                images  = batch["image"].to(device)
                masks   = batch["mask"].to(device)
                outputs = post_trans(model(images))
                dice_metric(y_pred=outputs, y=masks)
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

def _copy_inference_artifacts(src_dir: str, dest_dir: str) -> None:
    code_dir = os.path.join(dest_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    for fname in ("inference.py", "model_def.py", "requirements.txt"):
        # Look in the inference subdir (Docker container layout), then
        # the relative path from source tree, then the training dir itself
        candidates = [
            os.path.join(src_dir, "inference", fname),
            os.path.join(src_dir, "..", "..", "inference", "segmentation", fname),
            os.path.join(src_dir, fname),
        ]
        for src in candidates:
            src = os.path.normpath(src)
            if os.path.exists(src):
                _shutil.copy2(src, os.path.join(code_dir, fname))
                logger.info("Copied %s to model_dir/code/ for deployment", fname)
                break
        else:
            logger.warning("Deployment artifact not found, skipping: %s", fname)

_copy_inference_artifacts(
    src_dir=os.path.dirname(os.path.abspath(__file__)),
    dest_dir=args.model_dir,
)
