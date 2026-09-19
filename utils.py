"""
utils.py — Utility functions for The Pareidolia Paradox.

Contains:
    - seed_everything: Reproducibility seeding for all RNG sources
    - extract_physics_features: 6 handcrafted physics features from canonical image
"""

import os
import random

import numpy as np
import torch

import config


def seed_everything(seed=None):
    """
    Pin all random number generators for full reproducibility.

    Seeds: Python stdlib, NumPy, PyTorch CPU/CUDA, and cuDNN determinism.

    Args:
        seed: Integer seed. Defaults to config.SEED (42).
    """
    if seed is None:
        seed = config.SEED
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_physics_features(canonical_img):
    """
    Extract 6 handcrafted physics features from a canonical rotated grayscale image.

    The image must already be rotated by -sun_azimuth_angle before calling this.
    All features are computed on the un-augmented canonical image.

    Features:
        1. Shadow Fraction (f_shadow): fraction of pixels below P_30
        2. Top-to-Bottom Median Luminance Ratio (R_median)
        3. Mean Gradient Energy (G^2)
        4. Dominant Orientation via Structure Tensor (theta_flow)
        5. Cast Shadow Sharpness (S_edge)
        6. Vertical Edge Asymmetry (A_edge)

    Args:
        canonical_img: 2D numpy array (H, W), dtype uint8 or float.
                       Must be the canonical rotated image.

    Returns:
        np.ndarray of shape (6,) with float64 features.
    """
    img = canonical_img.astype(np.float64)
    h, w = img.shape

    # ── Feature 1: Shadow Fraction ──
    # Fraction of pixels below the 30th percentile
    p30 = np.percentile(img, config.SHADOW_PERCENTILE)
    f_shadow = np.mean(img < p30)

    # ── Feature 2: Top-to-Bottom Median Luminance Ratio ──
    half_h = h // 2
    top_half = img[:half_h, :]
    bottom_half = img[half_h:, :]
    r_median = np.median(top_half) / (np.median(bottom_half) + 1e-6)

    # ── Feature 3: Mean Gradient Energy ──
    gy, gx = np.gradient(img)
    g_squared = np.mean(gx ** 2 + gy ** 2)

    # ── Feature 4: Dominant Orientation via Structure Tensor ──
    # Spatially averaged structure tensor components
    gx2_mean = np.mean(gx ** 2)
    gy2_mean = np.mean(gy ** 2)
    gxy_mean = np.mean(gx * gy)
    theta_flow = 0.5 * np.arctan2(2.0 * gxy_mean, gx2_mean - gy2_mean)

    # ── Feature 5: Cast Shadow Sharpness ──
    s_edge = np.std(np.abs(gy)) + np.std(np.abs(gx))

    # ── Feature 6: Vertical Edge Asymmetry ──
    gy_top = gy[:half_h, :]
    gy_bottom = gy[half_h:, :]
    a_edge = np.std(np.abs(gy_bottom)) / (np.std(np.abs(gy_top)) + 1e-6)

    return np.array([f_shadow, r_median, g_squared, theta_flow, s_edge, a_edge],
                    dtype=np.float64)
