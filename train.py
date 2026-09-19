"""
train.py — Main training script for The Pareidolia Paradox.

Orchestrates the full 5-fold × 3-model training pipeline:
    1. Seed everything for reproducibility
    2. Load metadata, create StratifiedKFold splits
    3. For each fold × model combination (15 total):
        a. Fit StandardScaler on training fold physics features
        b. Create train/val datasets and dataloaders
        c. Run 2-phase training (warmup → fine-tuning)
        d. Collect OOF predictions for threshold calibration
    4. Optimize decision threshold on OOF probabilities
    5. Fit global StandardScaler on all training samples
    6. Save global scaler for inference

Usage:
    python train.py
"""

import os

import cv2
import joblib
import numpy as np
import pandas as pd
import scipy.ndimage
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.amp import autocast
from torch.utils.data import DataLoader

import config
from dataset import LunarTrainDataset
from model import HybridLunarClassifier
from trainer import evaluate_balanced_acc, train_model_fold
from utils import extract_physics_features, seed_everything


def compute_fold_physics_features(dataframe, img_dir):
    """
    Compute physics features for all samples in a dataframe.

    Used to fit StandardScaler on training fold data.

    Args:
        dataframe: pandas DataFrame with 'image_id' and 'sun_azimuth_angle'.
        img_dir: Path to image directory.

    Returns:
        np.ndarray of shape (N, 6) with physics features.
    """
    all_feats = []
    for idx in range(len(dataframe)):
        row = dataframe.iloc[idx]
        img_path = os.path.join(img_dir, row["image_id"])
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        azimuth = float(row["sun_azimuth_angle"])
        rot_img = scipy.ndimage.rotate(raw_img, -azimuth, reshape=False, mode="nearest")
        feats = extract_physics_features(rot_img)
        all_feats.append(feats)
    return np.array(all_feats)


def collect_oof_predictions(model, val_loader, device):
    """
    Collect OOF softmax probabilities and true labels from validation loader.

    Used for post-training threshold optimization.

    Args:
        model: Trained HybridLunarClassifier (will be set to eval).
        val_loader: Validation DataLoader yielding (images, physics_feats, labels).
        device: torch device.

    Returns:
        tuple: (probs_array of shape (N, 2), labels_array of shape (N,))
    """
    model.eval()
    all_probs, all_labels = [], []

    device_type = config.get_device_type(device)

    with torch.no_grad():
        for images, physics_feats, labels in val_loader:
            images = images.to(device)
            physics_feats = physics_feats.to(device)

            with autocast(device_type=device_type, enabled=(device_type == "cuda")):
                logits = model(images, physics_feats)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def optimize_threshold(oof_probs, oof_targets):
    """
    Search for optimal decision threshold on OOF probabilities.

    Scans np.linspace(0.30, 0.70, 41) and picks the threshold
    that maximizes balanced accuracy.

    Args:
        oof_probs: (N, 2) array of softmax probabilities.
        oof_targets: (N,) array of true labels.

    Returns:
        tuple: (best_threshold, best_balanced_accuracy)
    """
    best_threshold = 0.50
    best_bacc = 0.0

    for t in np.linspace(config.THRESHOLD_LOW, config.THRESHOLD_HIGH,
                         config.THRESHOLD_STEPS):
        preds = (oof_probs[:, 1] >= t).astype(int)
        score = balanced_accuracy_score(oof_targets, preds)
        if score > best_bacc:
            best_bacc = score
            best_threshold = t

    return best_threshold, best_bacc


def main():
    """Main training entry point."""
    # ─── Setup ───
    seed_everything()
    device = config.get_device()
    print(f"Device: {device}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ─── Load metadata ───
    train_df = pd.read_csv(config.TRAIN_METADATA_CSV)
    print(f"Training samples: {len(train_df)}")
    print(f"Class distribution: {train_df['label'].value_counts().to_dict()}")

    # ─── Stratified K-Fold ───
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS,
        shuffle=True,
        random_state=config.SEED,
    )

    # Storage for OOF predictions (across all models)
    # Shape will be (N, 2) averaged across 3 models
    full_oof_probs = np.zeros((len(train_df), 2), dtype=np.float64)
    full_oof_counts = np.zeros(len(train_df), dtype=np.int32)
    full_oof_targets = np.full(len(train_df), -1, dtype=np.int64)

    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(
        skf.split(train_df, train_df["label"])
    ):
        print(f"\n{'#'*70}")
        print(f"# FOLD {fold_idx} | Train: {len(train_indices)} | Val: {len(val_indices)}")
        print(f"{'#'*70}")

        train_fold_df = train_df.iloc[train_indices]
        val_fold_df = train_df.iloc[val_indices]

        # ─── Fit StandardScaler on training fold physics features ───
        print("  Computing physics features for training fold...")
        train_physics = compute_fold_physics_features(train_fold_df, config.TRAIN_IMAGES_DIR)
        fold_scaler = StandardScaler()
        fold_scaler.fit(train_physics)
        print(f"  StandardScaler fitted on {len(train_physics)} samples.")

        # ─── Create datasets ───
        train_dataset = LunarTrainDataset(
            train_fold_df, config.TRAIN_IMAGES_DIR, fold_scaler, is_train=True
        )
        val_dataset = LunarTrainDataset(
            val_fold_df, config.TRAIN_IMAGES_DIR, fold_scaler, is_train=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # ─── Train each backbone ───
        for model_name in config.MODEL_NAMES:
            print(f"\n  >>> Training {model_name} on Fold {fold_idx}")

            model = HybridLunarClassifier(
                model_name=model_name,
                pretrained=True,
            ).to(device)

            model, best_bacc = train_model_fold(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                fold_idx=fold_idx,
                model_name=model_name,
                device=device,
            )

            fold_results.append({
                "fold": fold_idx,
                "model": model_name,
                "best_val_bacc": best_bacc,
            })

            # ─── Collect OOF predictions ───
            oof_probs, oof_labels = collect_oof_predictions(model, val_loader, device)
            full_oof_probs[val_indices] += oof_probs
            full_oof_counts[val_indices] += 1
            full_oof_targets[val_indices] = oof_labels

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

    # ─── Average OOF probabilities across 3 models ───
    valid_mask = full_oof_counts > 0
    full_oof_probs[valid_mask] /= full_oof_counts[valid_mask, np.newaxis]

    # ─── Print fold results summary ───
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    for r in fold_results:
        print(f"  Fold {r['fold']} | {r['model']:20s} | Val BAcc: {r['best_val_bacc']:.4f}")

    mean_bacc = np.mean([r["best_val_bacc"] for r in fold_results])
    print(f"\n  Mean Val Balanced Acc (all 15 models): {mean_bacc:.4f}")

    # ─── OOF Threshold Optimization ───
    print(f"\n{'='*70}")
    print("OOF THRESHOLD OPTIMIZATION")
    print(f"{'='*70}")

    best_threshold, best_oof_bacc = optimize_threshold(
        full_oof_probs, full_oof_targets
    )
    print(f"  Optimal Threshold: {best_threshold:.4f}")
    print(f"  OOF Balanced Accuracy at τ*: {best_oof_bacc:.4f}")

    # Save threshold for predict.py
    threshold_path = os.path.join(config.CHECKPOINT_DIR, "best_threshold.npy")
    np.save(threshold_path, best_threshold)
    print(f"  Threshold saved to {threshold_path}")

    # ─── Fit Global StandardScaler on ALL training samples ───
    print(f"\n{'='*70}")
    print("FITTING GLOBAL STANDARD SCALER")
    print(f"{'='*70}")

    print("  Computing physics features for all 7,854 training samples...")
    all_train_physics = compute_fold_physics_features(train_df, config.TRAIN_IMAGES_DIR)
    global_scaler = StandardScaler()
    global_scaler.fit(all_train_physics)
    joblib.dump(global_scaler, config.GLOBAL_SCALER_PATH)
    print(f"  Global scaler saved to {config.GLOBAL_SCALER_PATH}")

    print(f"\n{'='*70}")
    print("ALL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  15 checkpoints saved in {config.CHECKPOINT_DIR}/")
    print(f"  Global scaler: {config.GLOBAL_SCALER_PATH}")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  Run predict.py to generate submission.csv")


if __name__ == "__main__":
    main()
