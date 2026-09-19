"""
model.py — Neural network architecture for The Pareidolia Paradox.

Contains:
    - HybridLunarClassifier: CNN backbone (timm) + 6 physics features → binary classification.

Architecture:
    CNN Backbone (timm, global_pool='avg') → embedding (D)
    Physics Features (6) ─────────────────────────┐
                                                   ↓
                                        Concatenate (D + 6)
                                                   ↓
                                    Linear(D+6, 256) → BN → ReLU → Dropout(0.3) → Linear(256, 2)

Supported backbones (embedding dims auto-detected via backbone.num_features):
    - efficientnet_b3  → D=1536
    - resnet50         → D=2048
    - convnext_tiny    → D=768
"""

import torch
import torch.nn as nn
import timm

import config


class HybridLunarClassifier(nn.Module):
    """
    Hybrid classifier combining a pretrained CNN backbone with handcrafted
    physics features for lunar surface binary classification.

    The backbone outputs a flat embedding via global average pooling.
    Physics features (6-dim) are concatenated to the embedding before
    the classification head.

    Args:
        model_name: timm model name (e.g., 'efficientnet_b3', 'resnet50', 'convnext_tiny').
        pretrained: Whether to load ImageNet pretrained weights.
        num_physics_feats: Number of handcrafted physics features (default: 6).
    """

    def __init__(self, model_name="convnext_tiny", pretrained=True,
                 num_physics_feats=None):
        super().__init__()

        if num_physics_feats is None:
            num_physics_feats = config.NUM_PHYSICS_FEATS

        # Backbone with explicit global_pool='avg' guarantees 2D output (B, D)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # Dynamic embedding dimension — never hardcoded
        in_features = self.backbone.num_features

        # Classification head: (D + 6) → 256 → 2
        self.head = nn.Sequential(
            nn.Linear(in_features + num_physics_feats, config.HEAD_HIDDEN_DIM),
            nn.BatchNorm1d(config.HEAD_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.HEAD_DROPOUT),
            nn.Linear(config.HEAD_HIDDEN_DIM, 2),
        )

    def forward(self, img, physics_feats):
        """
        Forward pass.

        Args:
            img: (B, 3, 256, 256) float tensor, ImageNet-normalized.
            physics_feats: (B, 6) float tensor, StandardScaler-transformed.

        Returns:
            logits: (B, 2) raw logits for CrossEntropyLoss.
        """
        cnn_feats = self.backbone(img)  # Shape: (B, D)
        combined = torch.cat([cnn_feats, physics_feats], dim=1)  # Shape: (B, D+6)
        return self.head(combined)
