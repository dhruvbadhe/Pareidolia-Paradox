"""
dataset.py — PyTorch Dataset classes for The Pareidolia Paradox.

Contains:
    - LunarTrainDataset: Training dataset with canonical rotation, physics features,
                         fold-level StandardScaler, 3-channel conversion, and
                         physics-safe augmentation (hflip + brightness jitter).
    - LunarTestDataset:  Evaluation dataset with canonical rotation, physics features,
                         global StandardScaler, 3-channel conversion, no augmentation.

Critical design constraints:
    - Physics features are ALWAYS extracted on the un-augmented canonical rotated image.
    - Augmentation is applied ONLY to the CNN tensor, AFTER physics extraction.
    - NO vertical flip (inverts canonical light direction).
    - NO large random rotations (destroys azimuth normalization).
"""

import os

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config
from utils import extract_physics_features


class LunarTrainDataset(Dataset):
    """
    Training dataset for lunar surface binary classification.

    Each sample returns:
        - img_tensor: (3, 256, 256) float32 tensor, ImageNet-normalized, augmented
        - physics_tensor: (6,) float32 tensor, StandardScaler-transformed
        - label: int64 scalar (0 = crater/depth, 1 = mound/rise)

    Pipeline per sample:
        1. Load grayscale image (256x256)
        2. Canonical rotation by -sun_azimuth_angle (scipy.ndimage.rotate, mode='nearest')
        3. Extract 6 physics features on canonical image (un-augmented)
        4. StandardScaler transform on physics features (fold-level scaler)
        5. Convert grayscale → 3-channel via np.repeat
        6. ImageNet normalization
        7. Physics-safe augmentation: RandomHorizontalFlip + ColorJitter(brightness)
    """

    def __init__(self, dataframe, img_dir, scaler, is_train=True):
        """
        Args:
            dataframe: pandas DataFrame with columns ['image_id', 'sun_azimuth_angle', 'label'].
            img_dir: Path to directory containing training images.
            scaler: Fitted sklearn StandardScaler for physics features.
            is_train: If True, apply physics-safe augmentation. If False, no augmentation.
        """
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.scaler = scaler
        self.is_train = is_train

        # Base transform: always applied (ImageNet normalization)
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])

        # Augmentation: applied only during training
        if is_train:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=config.HFLIP_PROB),
                transforms.ColorJitter(brightness=config.COLOR_JITTER_BRIGHTNESS),
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        azimuth = float(row["sun_azimuth_angle"])
        label = int(row["label"])

        # 1. Load grayscale image
        img_path = os.path.join(self.img_dir, img_id)
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 2. Canonical rotation (CCW by -sun_azimuth_angle)
        rot_img = scipy.ndimage.rotate(raw_img, -azimuth, reshape=False, mode="nearest")

        # 3. Extract physics features on canonical image (BEFORE augmentation)
        raw_physics = extract_physics_features(rot_img)

        # 4. StandardScaler transform
        scaled_physics = self.scaler.transform(raw_physics.reshape(1, -1)).flatten()
        physics_tensor = torch.tensor(scaled_physics, dtype=torch.float32)

        # 5. Convert grayscale → 3-channel
        img_3ch = np.repeat(rot_img[:, :, np.newaxis], 3, axis=-1)

        # 6. Base transform (ToTensor + ImageNet normalize)
        img_tensor = self.base_transform(img_3ch)

        # 7. Physics-safe augmentation (only during training)
        if self.aug_transform is not None:
            img_tensor = self.aug_transform(img_tensor)

        return img_tensor, physics_tensor, label


class LunarTestDataset(Dataset):
    """
    Evaluation/test dataset for lunar surface binary classification.

    Each sample returns:
        - img_tensor: (3, 256, 256) float32 tensor, ImageNet-normalized, NO augmentation
        - physics_tensor: (6,) float32 tensor, global StandardScaler-transformed
        - img_id: string identifier for submission CSV

    Pipeline per sample:
        1. Load grayscale image (256x256)
        2. Canonical rotation by -sun_azimuth_angle
        3. Extract 6 physics features on canonical image
        4. Global StandardScaler transform on physics features
        5. Convert grayscale → 3-channel
        6. ImageNet normalization (no augmentation)
    """

    def __init__(self, metadata_csv, img_dir, scaler):
        """
        Args:
            metadata_csv: Path to test_metadata.csv with columns ['image_id', 'sun_azimuth_angle'].
            img_dir: Path to directory containing evaluation images.
            scaler: Fitted sklearn StandardScaler (global, fit on all training samples).
        """
        self.df = pd.read_csv(metadata_csv).reset_index(drop=True)
        self.img_dir = img_dir
        self.scaler = scaler
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        azimuth = float(row["sun_azimuth_angle"])

        # 1. Load grayscale image
        img_path = os.path.join(self.img_dir, img_id)
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 2. Canonical rotation
        rot_img = scipy.ndimage.rotate(raw_img, -azimuth, reshape=False, mode="nearest")

        # 3. Extract physics features on canonical image & scale
        raw_physics = extract_physics_features(rot_img)
        scaled_physics = self.scaler.transform(raw_physics.reshape(1, -1)).flatten()

        # 4. Prepare 3-channel tensor for CNN backbone
        img_3ch = np.repeat(rot_img[:, :, np.newaxis], 3, axis=-1)
        img_tensor = self.img_transform(img_3ch)

        return img_tensor, torch.tensor(scaled_physics, dtype=torch.float32), img_id
