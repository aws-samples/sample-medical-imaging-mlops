"""
Data utilities for MedNIST-style classification datasets.
Expected folder structure:
    data_folder/
        ClassName_A/  image1.jpg ...
        ClassName_B/  ...

Label mapping is built dynamically from the training data folder structure.
"""
import os
import json
import random
import logging
import torch
from monai.data import DataLoader
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity

IMG_SIZE = (256, 256, 1)
logger = logging.getLogger(__name__)


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels      = labels
        self.transforms  = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def generate_list_labels(data_folder, label_dict=None):
    """
    Scan data_folder for class subdirectories and build image/label lists.
    If label_dict is provided (from training set), reuse it for consistency.
    Otherwise build it from sorted folder names.
    Returns (image_list, label_list, num_classes, label_dict).
    """
    class_dirs = sorted([d for d in os.listdir(data_folder)
                         if os.path.isdir(os.path.join(data_folder, d))])

    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {data_folder}")

    # Build or reuse label dict
    if label_dict is None:
        label_dict = {cls: idx for idx, cls in enumerate(class_dirs)}
        logger.info("Built label dict from data: %s", label_dict)

    image_list, label_list = [], []
    for cls in class_dirs:
        label = label_dict.get(cls)
        if label is None:
            # Assign new label for unseen class
            label = len(label_dict)
            label_dict[cls] = label
            logger.warning("New class '%s' not in label_dict — assigned label %d", cls, label)
        cls_path = os.path.join(data_folder, cls)
        for fname in os.listdir(cls_path):
            if os.path.isfile(os.path.join(cls_path, fname)):
                image_list.append(os.path.join(cls_path, fname))
                label_list.append(label)

    if not image_list:
        raise ValueError(f"No image files found under {data_folder}")

    combined = list(zip(image_list, label_list))
    random.shuffle(combined)
    image_list, label_list = zip(*combined)
    return list(image_list), list(label_list), len(label_dict), label_dict


def create_data_loaders(data_path, batch_size, label_dict=None):
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(spatial_size=IMG_SIZE),
        ScaleIntensity(),
    ])
    images, labels, num_classes, label_dict = generate_list_labels(data_path, label_dict)
    dataset = MedNISTDataset(images, labels, transforms)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader, num_classes, label_dict
