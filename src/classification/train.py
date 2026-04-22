"""
Training script for medical image classification on SageMaker BYOC.
Accepts: --task_type, --dimension, --model_name, --batch_size, --epochs,
         --learning_rate, --num_classes, --val_interval
Data channel: SM_CHANNEL_DATA with train/ val/ subfolders
"""
import os
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
args, _ = parser.parse_known_args()
logger.info("Args: %s", args)

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

# Save label dict alongside the model for inference
import json
with open(os.path.join(args.model_dir, "label_dict.json"), "w") as f:
    json.dump(label_dict, f, indent=2)
logger.info("Label dict: %s", label_dict)

model     = ModelDef(num_classes, args.model_name).get_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn   = nn.CrossEntropyLoss()
logger.info("Model: %s  |  Params: %s", args.model_name, sum(p.numel() for p in model.parameters()))


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


best_acc = 0.0
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    logger.info("Epoch %d/%d  train_loss=%.4f  train_acc=%.4f", epoch, args.epochs, train_loss, train_acc)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)

    if epoch % args.val_interval == 0:
        val_loss, val_acc = run_epoch(val_loader, training=False)
        logger.info("Epoch %d/%d  val_loss=%.4f  val_acc=%.4f", epoch, args.epochs, val_loss, val_acc)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                       os.path.join(args.model_dir, "best_model.pth"))
            logger.info("Best model saved (val_acc=%.4f)", best_acc)

torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": args.epochs},
           os.path.join(args.model_dir, "model.pth"))
writer.close()
logger.info("Training complete. Best val_acc=%.4f", best_acc)
