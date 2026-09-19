"""
config.py — Centralized configuration for The Pareidolia Paradox.

All hyperparameters, paths, and constants from the V9 Golden Standard
solution specification. Every magic number lives here.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
TRAIN_IMAGES_DIR = os.path.join("train_images")
EVAL_IMAGES_DIR = os.path.join("eval_images")
TRAIN_METADATA_CSV = os.path.join("train_metadata.csv")
TEST_METADATA_CSV = os.path.join("test_metadata.csv")
CHECKPOINT_DIR = os.path.join("checkpoints")
SUBMISSION_CSV = os.path.join("submission.csv")
GLOBAL_SCALER_PATH = os.path.join(CHECKPOINT_DIR, "global_scaler.pkl")

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────
# Model Backbones (timm names)
# ─────────────────────────────────────────────
MODEL_NAMES = ["efficientnet_b3", "resnet50", "convnext_tiny"]
# Dynamic: D is read at runtime via backbone.num_features
# efficientnet_b3 → 1536, resnet50 → 2048, convnext_tiny → 768

# ─────────────────────────────────────────────
# Physics Features
# ─────────────────────────────────────────────
NUM_PHYSICS_FEATS = 6
SHADOW_PERCENTILE = 30  # P_30 threshold for shadow fraction

# ─────────────────────────────────────────────
# Classification Head
# ─────────────────────────────────────────────
HEAD_HIDDEN_DIM = 256
HEAD_DROPOUT = 0.3

# ─────────────────────────────────────────────
# Training — General
# ─────────────────────────────────────────────
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 7

# Class weights for CrossEntropyLoss (Class 0 = 1.75, Class 1 = 1.0)
# Compensates for 2854 vs 5000 class imbalance
CLASS_WEIGHTS = [1.75, 1.0]

# ─────────────────────────────────────────────
# Training — Phase 1 (Head Warmup)
# ─────────────────────────────────────────────
PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3
PHASE1_WEIGHT_DECAY = 1e-4

# ─────────────────────────────────────────────
# Training — Phase 2 (Full Differential Fine-Tuning)
# ─────────────────────────────────────────────
PHASE2_BACKBONE_LR = 1e-5
PHASE2_HEAD_LR = 1e-4
PHASE2_WEIGHT_DECAY = 1e-4
COSINE_T_MAX = MAX_EPOCHS - PHASE1_EPOCHS  # 35
COSINE_ETA_MIN = 1e-7

# ─────────────────────────────────────────────
# ImageNet Normalization (for 3-channel grayscale)
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# Data Augmentation (physics-safe)
# ─────────────────────────────────────────────
HFLIP_PROB = 0.5
COLOR_JITTER_BRIGHTNESS = 0.08

# ─────────────────────────────────────────────
# Inference — OOF Threshold Search
# ─────────────────────────────────────────────
THRESHOLD_LOW = 0.30
THRESHOLD_HIGH = 0.70
THRESHOLD_STEPS = 41  # np.linspace(0.30, 0.70, 41)

# ─────────────────────────────────────────────
# Image Properties
# ─────────────────────────────────────────────
IMG_SIZE = 256  # Native resolution, no resizing
NUM_CHANNELS = 3  # Grayscale repeated to 3 channels
