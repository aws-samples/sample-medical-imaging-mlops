"""
Data utilities for 3D NIfTI segmentation datasets.
Expected folder structure:
    data_folder/
        subject_001/
            img.nii.gz
            label.nii.gz
        subject_002/
            ...

Patch-based training: instead of resizing each (large) volume down to a fixed
grid — which shrinks small structures (e.g. lung nodules) to a handful of voxels
and collapses training — we crop foreground-biased patches at NATIVE resolution
with RandCropByPosNegLabeld. SegResNet/SwinUNETR/UNet are fully convolutional, so
they train on patches and infer on the full volume via sliding window
(SlidingWindowInferer in inference.py, roi_size == training ROI).

Validation keeps the full native volume (no crop/resize) and is evaluated with a
sliding-window inferer in train.py, matching how inference runs.
"""
import os
from monai.data import DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ScaleIntensityRanged, RandCropByPosNegLabeld,
    RandFlipd, RandShiftIntensityd,
)

# Default training patch size (divisible by 16 for the 4x-downsampling encoders).
DEFAULT_ROI_SIZE = (96, 96, 96)
# How many foreground/background patches to sample per volume per epoch step.
DEFAULT_NUM_SAMPLES = 2


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


def create_data_loaders(data_path, batch_size, training=True,
                        roi_size=DEFAULT_ROI_SIZE, num_samples=DEFAULT_NUM_SAMPLES):
    """Build a DataLoader.

    Training: foreground-biased patch crops at native resolution. Each volume
    yields `num_samples` patches of `roi_size` (half centred on foreground, half
    on background via pos=neg=1), so the model actually sees the target instead
    of near-empty downsized volumes.

    Validation: the full native volume (channel-first, HU-windowed). Evaluated
    with a sliding-window inferer in the training loop.
    """
    roi_size = tuple(int(x) for x in roi_size)

    if training:
        transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            # CT HU window → [0,1] (image only).
            ScaleIntensityRanged(keys="image", a_min=-100, a_max=500,
                                 b_min=0, b_max=1, clip=True),
            EnsureTyped(keys=["image", "mask"]),
            # Foreground-biased patch sampling at native resolution.
            # pos=1, neg=1 → ~half the patches are centred on a labelled voxel,
            # which is what lets sparse targets (nodules) actually train.
            RandCropByPosNegLabeld(
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=roi_size,
                pos=1, neg=1,
                num_samples=num_samples,
                image_key="image",
                allow_smaller=True,
            ),
            RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ])
    else:
        # Validation: full native volume (no crop/resize); sliding-window infer.
        transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            ScaleIntensityRanged(keys="image", a_min=-100, a_max=500,
                                 b_min=0, b_max=1, clip=True),
            EnsureTyped(keys=["image", "mask"]),
        ])

    files   = _build_file_list(data_path)
    dataset = Dataset(data=files, transform=transforms)
    # RandCropByPosNegLabeld returns a LIST of patches per item; list_data_collate
    # flattens those into the batch. Validation uses batch_size=1 (volumes vary
    # in size, so they cannot be stacked).
    if training:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2, collate_fn=list_data_collate)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
