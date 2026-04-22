"""
Data utilities for 3D NIfTI segmentation datasets.
Expected folder structure:
    data_folder/
        subject_001/
            img.nii.gz
            label.nii.gz
        subject_002/
            ...
"""
import os
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized,
    ScaleIntensityRanged, RandFlipd, RandShiftIntensityd,
)

SPATIAL_SIZE = (128, 128, 64)


def _build_file_list(data_folder):
    pairs = []
    for subj in sorted(os.listdir(data_folder)):
        subj_path = os.path.join(data_folder, subj)
        if not os.path.isdir(subj_path):
            continue
        pairs.append({
            "image": os.path.join(subj_path, "img.nii.gz"),
            "mask":  os.path.join(subj_path, "label.nii.gz"),
        })
    return pairs


def create_data_loaders(data_path, batch_size, training=True):
    if training:
        transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Resized(keys=["image", "mask"], spatial_size=SPATIAL_SIZE),
            ScaleIntensityRanged(keys="image", a_min=-100, a_max=500, b_min=0, b_max=1, clip=True),
            RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ])
    else:
        transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Resized(keys=["image", "mask"], spatial_size=SPATIAL_SIZE),
            ScaleIntensityRanged(keys="image", a_min=-100, a_max=500, b_min=0, b_max=1, clip=True),
        ])

    files   = _build_file_list(data_path)
    dataset = Dataset(data=files, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=4)
