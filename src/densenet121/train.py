"""
DenseNet121 training script for SageMaker BYOC.
Reads data from SM_CHANNEL_TRAIN / SM_CHANNEL_TEST environment variables.
Saves model checkpoint to SM_MODEL_DIR.
"""
import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_def import ModelDef
from utils import create_data_loader, load_label_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./output"))
    parser.add_argument("--model_name", type=str, default="DenseNet121")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--dimension", type=str, default="2D")
    return parser.parse_args()


def train_one_epoch(model, loader, loss_fn, optimizer, device, writer, epoch, model_name):
    model.train()
    running_loss, running_corrects, running_samples = 0.0, 0, 0

    for step, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        if model_name == "DenseNet121":
            outputs = model(inputs[:, :, :, :, 0])
        else:
            outputs = model(inputs)[0]

        loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        running_samples += len(inputs)

        writer.add_scalar("Loss/train_step", loss.item(), epoch * len(loader) + step)

    epoch_loss = running_loss / running_samples
    epoch_acc = running_corrects / running_samples
    logger.info("Train — Loss: %.4f  Acc: %.4f", epoch_loss, epoch_acc)
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)
    return epoch_loss, epoch_acc


def evaluate(model, loader, loss_fn, device, writer, epoch, model_name):
    model.eval()
    running_loss, running_corrects = 0.0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            if model_name == "DenseNet121":
                outputs = model(inputs[:, :, :, :, 0])
            else:
                outputs = model(inputs)[0]

            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(loader.dataset)
    total_acc = running_corrects / len(loader.dataset)
    logger.info("Val   — Loss: %.4f  Acc: %.4f", total_loss, total_acc)
    writer.add_scalar("Loss/val", total_loss, epoch)
    writer.add_scalar("Accuracy/val", total_acc, epoch)
    return total_loss, total_acc


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    label_dict = load_label_dict()
    data_path = os.environ.get("SM_CHANNEL_DATA", "./data")
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    logger.info("Task: %s | Dimension: %s", args.task_type, args.dimension)
    logger.info("Train path: %s | Test path: %s", train_path, test_path)

    train_loader, num_classes = create_data_loader(train_path, args.batch_size, label_dict, shuffle=True)
    test_loader, _ = create_data_loader(test_path, args.batch_size, label_dict, shuffle=False)

    model = ModelDef(num_classes=num_classes, model_name=args.model_name).get_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    writer = SummaryWriter(log_dir=os.path.join(args.model_dir, "tensorboard"))

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        logger.info("Epoch %d / %d", epoch + 1, args.epochs)
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, writer, epoch, args.model_name)

        if epoch % args.val_interval == 0:
            _, val_acc = evaluate(model, test_loader, loss_fn, device, writer, epoch, args.model_name)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(args.model_dir, "model_best.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_path)
                logger.info("Best model saved (val_acc=%.4f)", val_acc)

    # Save final checkpoint
    os.makedirs(args.model_dir, exist_ok=True)
    final_path = os.path.join(args.model_dir, "model.pth")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, final_path)
    logger.info("Final model saved to %s", final_path)

    writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
