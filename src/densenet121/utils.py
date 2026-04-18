"""
Data loading utilities for medical image classification.
Expects data organised as: <split>/<class_name>/<image_file>
A label_dict.json mapping class_name -> int must exist alongside this file.
"""
import json
import logging
import os
import random

import torch
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ScaleIntensity,
)

logger = logging.getLogger(__name__)

IMG_SIZE = (256, 256, 1)
LABEL_DICT_PATH = os.path.join(os.path.dirname(__file__), "label_dict.json")


def load_label_dict(path: str = LABEL_DICT_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def generate_list_labels(data_folder: str, label_dict: dict):
    """Return (image_paths, labels, num_classes) for all images under data_folder."""
    image_list, label_list = [], []

    for class_name, class_idx in label_dict.items():
        class_dir = os.path.join(data_folder, class_name)
        if not os.path.isdir(class_dir):
            logger.warning("Class folder not found, skipping: %s", class_dir)
            continue
        for fname in os.listdir(class_dir):
            image_list.append(os.path.join(class_dir, fname))
            label_list.append(class_idx)

    combined = list(zip(image_list, label_list))
    random.shuffle(combined)
    images, labels = zip(*combined) if combined else ([], [])
    return list(images), list(labels), len(label_dict)


class MedicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def create_data_loader(data_path: str, batch_size: int, label_dict: dict, shuffle: bool = True) -> tuple:
    """Build a DataLoader for the given data path. Returns (loader, num_classes)."""
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(spatial_size=IMG_SIZE),
        ScaleIntensity(),
    ])

    images, labels, num_classes = generate_list_labels(data_path, label_dict)
    dataset = MedicalImageDataset(images, labels, transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    logger.info("Loaded %d samples from %s (%d classes)", len(dataset), data_path, num_classes)
    return loader, num_classes
