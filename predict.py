"""
predict.py — Inference script for The Pareidolia Paradox.

Generates submission.csv from 2,000 evaluation images using:
    - 15 trained checkpoints (5 folds × 3 models)
    - 2 TTA views (original + horizontal flip) = 30 forward passes per image
    - Global StandardScaler (fitted on all 7,854 training samples)
    - Calibrated decision threshold τ* (from OOF optimization)

Critical constraints:
    - NO vertical flip in TTA (inverts canonical light direction)
    - Physics features are NOT flipped (only CNN tensor is flipped)
    - torch.load uses weights_only=True

Usage:
    python predict.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import LunarTestDataset
from model import HybridLunarClassifier
from utils import seed_everything


def main():
    """Main inference entry point."""
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Load global scaler ───
    print(f"Loading global scaler from {config.GLOBAL_SCALER_PATH}")
    global_scaler = joblib.load(config.GLOBAL_SCALER_PATH)

    # ─── Load calibrated threshold ───
    threshold_path = os.path.join(config.CHECKPOINT_DIR, "best_threshold.npy")
    best_threshold = float(np.load(threshold_path))
    print(f"Calibrated threshold τ*: {best_threshold:.4f}")

    # ─── Create test dataset and loader ───
    test_dataset = LunarTestDataset(
        metadata_csv=config.TEST_METADATA_CSV,
        img_dir=config.EVAL_IMAGES_DIR,
        scaler=global_scaler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_dataset)}")

    # ─── Load all 15 models ───
    print(f"\nLoading 15 checkpoints (5 folds × 3 models)...")
    models = []
    for fold_idx in range(config.NUM_FOLDS):
        for model_name in config.MODEL_NAMES:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"fold_{fold_idx}_{model_name}_best.pth",
            )

            if not os.path.exists(ckpt_path):
                print(f"  ✗ MISSING: {ckpt_path}")
                continue

            model = HybridLunarClassifier(
                model_name=model_name,
                pretrained=False,  # Loading saved weights, not ImageNet
            )
            model.load_state_dict(
                torch.load(ckpt_path, weights_only=True)
            )
            model.to(device)
            model.eval()
            models.append(model)
            print(f"  ✓ Loaded fold_{fold_idx}_{model_name}")

    num_models = len(models)
    print(f"\n{num_models} models loaded. TTA views: 2 (original + hflip)")
    print(f"Total forward passes per image: {num_models * 2}")

    # ─── Inference with TTA ───
    print(f"\nRunning inference on {len(test_dataset)} images...")
    all_probs = []
    all_img_ids = []

    with torch.no_grad():
        for images, physics_feats, img_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            physics_feats = physics_feats.to(device)

            batch_probs = torch.zeros(
                (images.size(0), 2), device=device, dtype=torch.float32
            )

            # 2 TTA views: original and horizontal flip
            for hflip in [False, True]:
                x_img = torch.flip(images, dims=[3]) if hflip else images

                for model in models:
                    with autocast():
                        logits = model(x_img, physics_feats)
                        probs = torch.softmax(logits, dim=1)
                    batch_probs += probs

            # Average over (num_models × 2 TTA views) forward passes
            batch_probs /= (num_models * 2)

            all_probs.extend(batch_probs[:, 1].cpu().numpy())
            all_img_ids.extend(img_ids)

    # ─── Apply calibrated threshold ───
    all_probs = np.array(all_probs)
    final_preds = (all_probs >= best_threshold).astype(int)

    # ─── Generate submission CSV ───
    sub_df = pd.DataFrame({
        "image_id": all_img_ids,
        "label": final_preds,
    })
    sub_df.to_csv(config.SUBMISSION_CSV, index=False)

    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"  Submission: {config.SUBMISSION_CSV}")
    print(f"  Total predictions: {len(sub_df)}")
    print(f"  Threshold τ*: {best_threshold:.4f}")
    print(f"  Class distribution: {dict(zip(*np.unique(final_preds, return_counts=True)))}")
    print(f"  Mean P(class=1): {all_probs.mean():.4f}")


if __name__ == "__main__":
    main()
