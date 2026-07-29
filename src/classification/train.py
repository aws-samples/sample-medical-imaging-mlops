"""
Training script for medical image classification on SageMaker BYOC.
Accepts: --task_type, --dimension, --model_name, --batch_size, --epochs,
         --learning_rate, --val_interval, --data_s3_uri
Data channel: SM_CHANNEL_DATA with train/ val/ subfolders
--data_s3_uri is optional provenance metadata written into result.json;
it does not affect data loading (data is always read from SM_CHANNEL_DATA).
"""
import os
import json
import shutil
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model_def import ModelDef
from utils import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",     type=str,   default=os.environ.get("SM_MODEL_DIR", "./output"))
parser.add_argument("--model_name",    type=str,   default="densenet121")
parser.add_argument("--task_type",     type=str,   default="classification")
parser.add_argument("--dimension",     type=str,   default="2D")
parser.add_argument("--batch_size",    type=int,   default=16)
parser.add_argument("--epochs",        type=int,   default=1)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--val_interval",  type=int,   default=1)
parser.add_argument("--data_s3_uri",   type=str,   default="")
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
        experiment_name = f"{args.task_type}-{args.model_name}"
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
    val_path = os.path.join(data_root, "test")

logger.info("Train: %s  |  Val: %s", train_path, val_path)

train_loader, num_classes, label_dict = create_data_loaders(train_path, args.batch_size)
val_loader,   _,           _          = create_data_loaders(val_path,   args.batch_size, label_dict)

with open(os.path.join(args.model_dir, "label_dict.json"), "w", encoding="utf-8") as f:
    json.dump(label_dict, f, indent=2)
logger.info("Label dict: %s", label_dict)

model     = ModelDef(num_classes, args.model_name).get_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn   = nn.CrossEntropyLoss()
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
            # Prefer best_model.pth; fall back to model.pth
            ckpt_name = "best_model.pth" if "best_model.pth" in tar.getnames() else "model.pth"
            f    = tar.extractfile(tar.getmember(ckpt_name))
            ckpt = torch.load(io.BytesIO(f.read()), map_location=device, weights_only=True)
        import os as _os; _os.unlink(tmp_path)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch     = ckpt.get("epoch", 0) + 1
        fine_tuned_from = args.checkpoint_s3_uri
        logger.info("Resumed from %s (epoch %d, val_acc=%.4f)",
                    ckpt_name, ckpt.get("epoch", 0), ckpt.get("val_acc", 0))
    except Exception as e:
        logger.warning("Could not load checkpoint, starting from scratch: %s", e)


def copy_inference_artifacts(src_dir: str, dest_dir: str) -> None:
    """Copy inference script and requirements into model_dir so SageMaker
    bundles them into model.tar.gz — no post-training merging needed."""
    for fname in ("inference.py", "requirements.txt", "model_def.py"):
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, fname))
            logger.info("Copied %s to model_dir for deployment", fname)
        else:
            logger.warning("Deployment artifact not found, skipping: %s", src)


def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 5:
                inputs = inputs[..., 0]
            raw    = model(inputs)
            logits = raw[0] if isinstance(raw, (list, tuple)) else raw
            loss   = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds       = logits.argmax(dim=1)
            total_loss += loss.item() * inputs.size(0)
            correct    += (preds == labels).sum().item()
            total      += inputs.size(0)
    return total_loss / total, correct / total


# ── Training loop ─────────────────────────────────────────────────────────────
best_acc   = 0.0
last_train_acc = 0.0
last_val_acc   = 0.0

if use_mlflow:
    mlflow.start_run()
    mlflow.log_params({
        "model_name":       args.model_name,
        "task_type":        args.task_type,
        "dimension":        args.dimension,
        "batch_size":       args.batch_size,
        "epochs":           args.epochs,
        "learning_rate":    args.learning_rate,
        "val_interval":     args.val_interval,
        "num_classes":      num_classes,
        "device":           str(device),
        "fine_tuned_from":  fine_tuned_from or "none",
    })

for epoch in range(start_epoch, args.epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    last_train_acc = train_acc
    logger.info("Epoch %d/%d  train_loss=%.4f  train_acc=%.4f", epoch, args.epochs, train_loss, train_acc)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    if use_mlflow:
        mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc}, step=epoch)

    if epoch % args.val_interval == 0:
        val_loss, val_acc = run_epoch(val_loader, training=False)
        last_val_acc = val_acc
        logger.info("Epoch %d/%d  val_loss=%.4f  val_acc=%.4f", epoch, args.epochs, val_loss, val_acc)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        if use_mlflow:
            mlflow.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.model_dir, "best_model.pth")
            torch.save({
                "model":          model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "epoch":          epoch,
                "val_acc":        val_acc,
                "train_acc":      train_acc,
                "train_loss":     train_loss,
                "val_loss":       val_loss,
                "model_name":     args.model_name,
                "num_classes":    num_classes,
            }, best_path)
            logger.info("Best checkpoint saved (epoch=%d  val_acc=%.4f)", epoch, best_acc)
            if use_mlflow:
                mlflow.log_artifact(best_path)

final_path = os.path.join(args.model_dir, "model.pth")
torch.save({
    "model":          model.state_dict(),
    "optimizer":      optimizer.state_dict(),
    "epoch":          args.epochs,
    "model_name":     args.model_name,
    "num_classes":    num_classes,
    "learning_rate":  args.learning_rate,
    "batch_size":     args.batch_size,
}, final_path)

# Lean deployment checkpoint — weights only, no optimizer/scheduler state.
# ~50% smaller than model.pth; this is what inference.py loads.
deploy_path = os.path.join(args.model_dir, "model_deploy.pth")
torch.save(model.state_dict(), deploy_path)
logger.info("Saved deployment checkpoint (weights only): %s", deploy_path)

# ── result.json ───────────────────────────────────────────────────────────────
# Tracks training provenance; read by inference.py at container startup.
result = {
    "model_name":        args.model_name,
    "num_classes":       num_classes,
    "final_epoch":       args.epochs,
    "train_accuracy":    round(last_train_acc, 4),
    "val_accuracy":      round(last_val_acc,   4),
    "best_val_accuracy": round(best_acc, 4),
    "data_s3_uri":       args.data_s3_uri or os.environ.get("SM_CHANNEL_DATA", data_root),
    "task_type":         args.task_type,
    "dimension":         args.dimension,
    "learning_rate":     args.learning_rate,
    "batch_size":        args.batch_size,
    "fine_tuned_from":   fine_tuned_from or None,
}
result_path = os.path.join(args.model_dir, "result.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
logger.info("Saved result.json: %s", result)

copy_inference_artifacts(
    src_dir=os.path.dirname(os.path.abspath(__file__)),
    dest_dir=args.model_dir,
)

if use_mlflow:
    mlflow.log_metric("best_val_acc", best_acc)
    mlflow.log_artifact(final_path)
    mlflow.log_artifact(deploy_path)
    mlflow.log_artifact(result_path)
    mlflow.log_artifact(os.path.join(args.model_dir, "label_dict.json"))
    mlflow.end_run()

writer.close()
logger.info("Training complete. Best val_acc=%.4f", best_acc)
