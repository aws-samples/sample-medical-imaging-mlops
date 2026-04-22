"""
Training script for 3D medical image segmentation on SageMaker BYOC.
Accepts: --task_type, --dimension, --model_name, --batch_size, --epochs,
         --learning_rate, --num_classes, --val_interval
Data channel: SM_CHANNEL_DATA with train/ val/ subfolders of NIfTI volumes.
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
parser.add_argument("--num_classes",   type=int,   default=1)
parser.add_argument("--val_interval",  type=int,   default=2)
args = parser.parse_args()
logger.info("Args: %s", args)

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

model     = ModelDef(args.num_classes, args.model_name).get_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
loss_fn   = DiceLoss(sigmoid=True)
logger.info("Model: %s  |  Params: %s", args.model_name, sum(p.numel() for p in model.parameters()))

post_trans  = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
dice_metric = DiceMetric(include_background=True, reduction="mean")

best_dice = -1.0
for epoch in range(1, args.epochs + 1):
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
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice},
                       os.path.join(args.model_dir, "best_model.pth"))
            logger.info("Best model saved (val_dice=%.4f)", best_dice)

torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": args.epochs},
           os.path.join(args.model_dir, "model.pth"))
writer.close()
logger.info("Training complete. Best val_dice=%.4f", best_dice)
