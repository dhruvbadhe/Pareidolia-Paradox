"""
trainer.py — Training engine for The Pareidolia Paradox.

Contains:
    - evaluate_balanced_acc: Validation metric with AMP autocast and no_grad
    - train_one_epoch: Single epoch training loop with AMP mixed precision
    - train_model_fold: 2-phase training protocol with warmup checkpoint guarantee

Training protocol (Section 5B of V9 spec):
    Phase 1 (Epochs 1-5): Freeze backbone, train head only, lr=1e-3
    Phase 2 (Epochs 6-40): Unfreeze all, differential LR (backbone 1e-5, head 1e-4),
                           CosineAnnealingLR, early stopping patience=7

Critical contracts:
    - Warmup checkpoint is ALWAYS saved after Phase 1 (prevents FileNotFoundError)
    - Phase 2 uses a FRESH GradScaler (not carried from Phase 1)
    - Scheduler step happens AFTER evaluation but BEFORE early stopping check
    - torch.load uses weights_only=True
"""

import os

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

import config


def evaluate_balanced_acc(model, val_loader, device):
    """
    Evaluate model balanced accuracy on a validation loader.

    Uses AMP autocast and torch.no_grad() for efficiency.
    Handles both image and physics feature tensors from the dataloader.

    Args:
        model: HybridLunarClassifier in any mode (will be set to eval).
        val_loader: DataLoader yielding (images, physics_feats, labels).
        device: torch device.

    Returns:
        float: Balanced accuracy score in [0, 1].
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, physics_feats, labels in val_loader:
            images = images.to(device)
            physics_feats = physics_feats.to(device)

            with autocast():
                logits = model(images, physics_feats)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return balanced_accuracy_score(all_labels, all_preds)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """
    Train model for one epoch with AMP mixed precision.

    Args:
        model: HybridLunarClassifier.
        train_loader: DataLoader yielding (images, physics_feats, labels).
        criterion: Loss function (CrossEntropyLoss with class weights).
        optimizer: Optimizer (AdamW).
        scaler: GradScaler for AMP.
        device: torch device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, physics_feats, labels in tqdm(train_loader, desc="  Training", leave=False):
        images = images.to(device)
        physics_feats = physics_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(images, physics_feats)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


def train_model_fold(model, train_loader, val_loader, fold_idx, model_name,
                     device, max_epochs=None, patience=None):
    """
    Full 2-phase training protocol for a single fold.

    Phase 1 (Head Warmup, Epochs 1-5):
        - Freeze backbone parameters
        - Train only the classification head
        - AdamW(lr=1e-3, weight_decay=1e-4)
        - Own GradScaler

    Warmup Checkpoint Guarantee:
        - After Phase 1, save model state dict immediately
        - This prevents FileNotFoundError if Phase 2 never improves

    Phase 2 (Full Differential Fine-Tuning, Epochs 6-40):
        - Unfreeze all parameters
        - Differential LR: backbone 1e-5, head 1e-4
        - CosineAnnealingLR(T_max=35, eta_min=1e-7)
        - Fresh GradScaler (prevents scale carry-over)
        - Early stopping: patience=7 on val balanced accuracy

    Args:
        model: HybridLunarClassifier.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        fold_idx: Current fold index (0-4).
        model_name: Backbone name string for checkpoint filename.
        device: torch device.
        max_epochs: Total epochs including Phase 1. Defaults to config.MAX_EPOCHS.
        patience: Early stopping patience. Defaults to config.EARLY_STOP_PATIENCE.

    Returns:
        tuple: (model with best weights loaded, best_val_bacc)
    """
    if max_epochs is None:
        max_epochs = config.MAX_EPOCHS
    if patience is None:
        patience = config.EARLY_STOP_PATIENCE

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(config.CHECKPOINT_DIR,
                             f"fold_{fold_idx}_{model_name}_best.pth")

    # Weighted Cross-Entropy (1.75 for Class 0, 1.0 for Class 1)
    class_weights = torch.tensor(config.CLASS_WEIGHTS, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ─────────────────────────────────────────────
    # Phase 1: Head Warmup (Epochs 1–5)
    # ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Fold {fold_idx} | {model_name}] Phase 1: Head Warmup (Epochs 1-{config.PHASE1_EPOCHS})")
    print(f"{'='*60}")

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer_p1 = torch.optim.AdamW(
        model.head.parameters(),
        lr=config.PHASE1_LR,
        weight_decay=config.PHASE1_WEIGHT_DECAY,
    )
    scaler_p1 = GradScaler()

    for epoch in range(1, config.PHASE1_EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion,
                                   optimizer_p1, scaler_p1, device)
        print(f"  Epoch {epoch}/{config.PHASE1_EPOCHS} — Train Loss: {avg_loss:.4f}")

    # Phase 1 validation (diagnostic + sanity check)
    warmup_val_bacc = evaluate_balanced_acc(model, val_loader, device)

    # GUARANTEED initial checkpoint save (prevents FileNotFoundError)
    torch.save(model.state_dict(), save_path)
    best_val_bacc = warmup_val_bacc
    print(f"  ✓ Warmup checkpoint saved. Val Balanced Acc: {warmup_val_bacc:.4f}")

    # ─────────────────────────────────────────────
    # Phase 2: Full Differential Fine-Tuning (Epochs 6–40)
    # ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Fold {fold_idx} | {model_name}] Phase 2: Fine-Tuning "
          f"(Epochs {config.PHASE1_EPOCHS + 1}-{max_epochs})")
    print(f"{'='*60}")

    # Unfreeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = True

    optimizer_p2 = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config.PHASE2_BACKBONE_LR},
        {"params": model.head.parameters(), "lr": config.PHASE2_HEAD_LR},
    ], weight_decay=config.PHASE2_WEIGHT_DECAY)

    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2,
        T_max=config.COSINE_T_MAX,
        eta_min=config.COSINE_ETA_MIN,
    )

    # Fresh GradScaler for Phase 2 (prevents scale carry-over from Phase 1)
    scaler_p2 = GradScaler()
    epochs_no_improve = 0

    for epoch in range(config.PHASE1_EPOCHS + 1, max_epochs + 1):
        # Training
        avg_loss = train_one_epoch(model, train_loader, criterion,
                                   optimizer_p2, scaler_p2, device)

        # Step 1: Validation evaluation
        val_bacc = evaluate_balanced_acc(model, val_loader, device)

        # Step 2: Step scheduler (after evaluation, before early stopping)
        scheduler_p2.step()

        current_lr = optimizer_p2.param_groups[0]["lr"]
        print(f"  Epoch {epoch}/{max_epochs} — Loss: {avg_loss:.4f} | "
              f"Val BAcc: {val_bacc:.4f} | LR: {current_lr:.2e}")

        # Step 3: Checkpoint save & early stopping check
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best! Saved checkpoint.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  ✗ Early stopping at epoch {epoch}. "
                      f"Best Val Balanced Acc: {best_val_bacc:.4f}")
                break

    # Restore best weights
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"\n[Fold {fold_idx} | {model_name}] Training complete. "
          f"Best Val Balanced Acc: {best_val_bacc:.4f}")

    return model, best_val_bacc
