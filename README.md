# The Pareidolia Paradox 🌙

Binary classification of 256×256 grayscale lunar surface images into **craters (depth)** vs **mounds (rise)**, evaluated on **Balanced Accuracy**.

## 🏗️ Architecture

**Hybrid CNN + Physics Feature Ensemble** — 3 pretrained backbones × 5-fold stratified CV = 15 models, with 2 TTA views at inference (30 forward passes per image).

```
Raw Grayscale (256×256) + sun_azimuth_angle
              │
    Canonical Rotation (−azimuth)
              │
    ┌─────────┴──────────┐
    │                    │
 6 Physics           3-ch CNN
 Features            Tensor
    │                    │
 StandardScaler     ImageNet Norm
    │              + Augmentation
    │                    │
    │              CNN Backbone
    │              (timm, pooled)
    │                    │
    └────── Concat ──────┘
              │
     Classification Head
      (256d → 2 classes)
```

### Key Design Decisions

- **Canonical rotation** by `−sun_azimuth_angle` normalizes shadow direction across all samples
- **Physics features extracted before augmentation** on the canonical image (never augmented)
- **No vertical flip** anywhere (training or TTA) — would invert the canonical light direction
- **No raw azimuth fed to model** — prevents shortcut overfitting on train/test distribution shift

### Backbones

| Model | timm Name | Embedding Dim |
|-------|-----------|---------------|
| EfficientNet-B3 | `efficientnet_b3` | 1536 |
| ResNet-50 | `resnet50` | 2048 |
| ConvNeXt-Tiny | `convnext_tiny` | 768 |

### 6 Physics Features

1. **Shadow Fraction** — fraction of pixels below P₃₀
2. **Top/Bottom Luminance Ratio** — median(top half) / median(bottom half)
3. **Mean Gradient Energy** — average squared gradient magnitude
4. **Structure Tensor Orientation** — dominant edge direction via arctan2
5. **Cast Shadow Sharpness** — std of absolute gradient in both axes
6. **Vertical Edge Asymmetry** — ratio of bottom to top vertical edge strength

## 📁 Project Structure

```
├── config.py          # All hyperparameters, paths, constants
├── utils.py           # seed_everything + physics feature extraction
├── dataset.py         # LunarTrainDataset + LunarTestDataset
├── model.py           # HybridLunarClassifier (timm backbone + physics head)
├── trainer.py         # 2-phase training engine (warmup → fine-tuning)
├── train.py           # Main training script (5-fold × 3 models)
├── predict.py         # Inference with 15-model ensemble + TTA
├── requirements.txt   # Python dependencies
├── solution_plan.md   # V9 architecture specification
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place data files in the project root:
```
├── train_images/          # 7,854 training PNGs (256×256 grayscale)
├── eval_images/           # 2,000 evaluation PNGs
├── train_metadata.csv     # image_id, sun_azimuth_angle, label
└── test_metadata.csv      # image_id, sun_azimuth_angle
```

### 3. Train

```bash
python train.py
```

This runs the full pipeline:
- 5-fold stratified CV × 3 backbones = **15 model checkpoints**
- Per-fold StandardScaler fitting for physics features
- OOF threshold optimization (τ* search over [0.30, 0.70])
- Global scaler fitting on all training samples

**Output:** `checkpoints/` directory with 15 `.pth` files, `global_scaler.pkl`, and `best_threshold.npy`.

### 4. Predict

```bash
python predict.py
```

This generates `submission.csv`:
- Loads all 15 checkpoints + global scaler + calibrated threshold
- 2 TTA views (original + horizontal flip) × 15 models = **30 forward passes per image**
- Averages softmax probabilities, applies threshold τ*

## ⚙️ Training Protocol

### Phase 1: Head Warmup (Epochs 1–5)
- Backbone frozen, only classification head trains
- AdamW, lr=1e-3, weight_decay=1e-4
- Checkpoint saved after Phase 1 (guaranteed)

### Phase 2: Differential Fine-Tuning (Epochs 6–40)
- All parameters unfrozen
- Differential LR: backbone 1e-5, head 1e-4
- CosineAnnealingLR (T_max=35, η_min=1e-7)
- Early stopping: patience=7 on val balanced accuracy
- Fresh GradScaler (not carried from Phase 1)

### Loss Function
- CrossEntropyLoss with class weights [1.75, 1.0]
- Compensates for 2,854 (Class 0) vs 5,000 (Class 1) imbalance

## 📊 Dataset

| Split | Samples | Class 0 | Class 1 | Ratio |
|-------|---------|---------|---------|-------|
| Train | 7,854 | 2,854 (36.3%) | 5,000 (63.7%) | 1:1.75 |
| Eval | 2,000 | — | — | — |

## 🔧 Key Configuration (config.py)

| Parameter | Value |
|-----------|-------|
| Image size | 256×256 (native) |
| Batch size | 32 |
| Folds | 5 |
| Max epochs | 40 |
| Early stop patience | 7 |
| Phase 1 LR | 1e-3 |
| Phase 2 backbone LR | 1e-5 |
| Phase 2 head LR | 1e-4 |
| Dropout | 0.3 |
| Hidden dim | 256 |

## 📝 License

Competition submission for Nova Hack — The Pareidolia Paradox.
