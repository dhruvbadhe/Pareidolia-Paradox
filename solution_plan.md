# The Pareidolia Paradox: Master Solution Architecture & Engineering Specification (V9 - Golden Standard)

> **Task:** Binary classification of 256×256 grayscale lunar surface images:
> - **Class 0 (Depth):** Craters, depressions, holes
> - **Class 1 (Rise):** Mounds, hills, rocks, boulders
>
> **Evaluation Metric:** Balanced Accuracy  
> **Dataset:** 7,854 training images (`train_metadata.csv`), 2,000 evaluation images (`test_metadata.csv`)  
> **Framework:** PyTorch, Torchvision, TIMM  

---

## 1. Executive Summary & Audited Fixes in V9

All remaining runtime contracts and edge cases are now locked in:

| Item | Potential Failure / Ambiguity | V9 Defect-Free Fix |
|---|---|---|
| **Warmup Checkpoint Save** | 🔴 **Critical:** If Phase 2 doesn't beat Phase 1, `save_path` is never created $\to$ `FileNotFoundError` on restore. | **Save Initial Checkpoint Immediately:** Execute `torch.save(model.state_dict(), save_path)` right after Phase 1 evaluation before entering Phase 2. |
| **`LunarTestDataset` Specification** | 🟡 **Missing Contract:** Dataset implementation was implicit. | **Concrete `LunarTestDataset` Implementation:** Explicit PyTorch Dataset showing canonical rotation, physics feature extraction, scaler application, 3-channel conversion, and returning `(image_tensor, physics_tensor, img_id)`. |
| **Explicit Threshold Calculation Code** | 🟢 **Missing Variable:** `best_threshold` used in 6C without explicit code in 6B. | **Concrete OOF Optimization Routine:** Complete Python loop over `np.linspace(0.30, 0.70, 41)` storing `best_threshold` directly prior to inference. |

---

## 2. Decoupled Pipeline Architecture (Native 256×256)

```
Raw Grayscale Image (256x256) + sun_azimuth_angle
                      │
   scipy.ndimage.rotate(img, -sun_azimuth_angle, reshape=False, mode='nearest')
                      │
        [ Full-Resolution Canonical Rotated Image (256x256) ]
                      │
         ┌────────────┴──────────────────────────┐
         │                                       │
  Extract 6 Physics                      Native 256x256:
     Features                        (No bicubic resizing)
(Strictly on Canonical)                          │
         │                             Repeat to 3 Channels:
         │                             np.repeat(..., 3, axis=-1)
StandardScaler (Fold-k / Global)                 │
         │                             ImageNet Normalization
         │                                       │
         │                            Physics-Safe Augmentation:
         │                            • RandomHorizontalFlip(p=0.5)
         │                            • ColorJitter(brightness=0.08)
         │                            *(Applied ONLY to CNN tensor)*
         │                                       │
         │                                 CNN Backbone
         │                      (timm with explicit global_pool='avg')
         │                                       │
   Physics Vector (6)                    CNN Embedding (D)
         │                                       │
         └───────────────────┬───────────────────┘
                             │
                     Concatenate (D + 6)
                             │
                     Classification Head
```

---

## 3. The 6 Handcrafted Physics Features

Computed directly on the un-augmented canonical rotated image ($I$):

1. **Shadow Fraction ($f_{\text{shadow}}$):**
   $$f_{\text{shadow}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(I_i < P_{30})$$
2. **Top-to-Bottom Median Luminance Ratio ($R_{\text{median}}$):**
   $$R_{\text{median}} = \frac{\operatorname{median}(I_{\text{top\_half}})}{\operatorname{median}(I_{\text{bottom\_half}}) + 1e-6}$$
3. **Mean Gradient Energy ($G^2$):**
   $$G^2 = \frac{1}{N} \sum (g_x^2 + g_y^2) \quad \text{via } \texttt{np.gradient}(I)$$
4. **Dominant Orientation via Structure Tensor ($\theta_{\text{flow}}$):**
   $$J = \begin{bmatrix} \langle g_x^2 \rangle & \langle g_x g_y \rangle \\ \langle g_x g_y \rangle & \langle g_y^2 \rangle \end{bmatrix}, \quad \theta_{\text{flow}} = \frac{1}{2}\operatorname{arctan2}\left(2\langle g_x g_y \rangle, \; \langle g_x^2 \rangle - \langle g_y^2 \rangle\right)$$
5. **Cast Shadow Sharpness ($S_{\text{edge}}$):**
   $$S_{\text{edge}} = \operatorname{std}(|\Delta_y I|) + \operatorname{std}(|\Delta_x I|)$$
6. **Vertical Edge Asymmetry ($A_{\text{edge}}$):**
   $$A_{\text{edge}} = \frac{\operatorname{std}(|\Delta_y I_{\text{bottom}}|)}{\operatorname{std}(|\Delta_y I_{\text{top}}|) + 1e-6}$$

---

## 4. Models & Dataset Implementation Contracts

### A. Dynamic Embedding Classifier
```python
import torch
import torch.nn as nn
import timm

class HybridLunarClassifier(nn.Module):
    def __init__(self, model_name="convnext_tiny", pretrained=True, num_physics_feats=6):
        super().__init__()
        # Explicit global_pool='avg' guarantees a 2D tensor output (B, D)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        in_features = self.backbone.num_features  # 1536 for B3, 2048 for ResNet50, 768 for ConvNeXt-Tiny
        
        self.head = nn.Sequential(
            nn.Linear(in_features + num_physics_feats, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, img, physics_feats):
        cnn_feats = self.backbone(img)  # Shape: (B, in_features)
        combined = torch.cat([cnn_feats, physics_feats], dim=1)
        return self.head(combined)
```

### B. Concrete `LunarTestDataset` Implementation
```python
import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LunarTestDataset(Dataset):
    def __init__(self, metadata_csv, img_dir, scaler):
        self.df = pd.read_csv(metadata_csv).reset_index(drop=True)
        self.img_dir = img_dir
        self.scaler = scaler
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        azimuth = float(row['sun_azimuth_angle'])
        
        # 1. Load grayscale image (256x256)
        raw_img = cv2.imread(f"{self.img_dir}/{img_id}", cv2.IMREAD_GRAYSCALE)
        
        # 2. Canonical rotation (CCW by -sun_azimuth_angle)
        rot_img = scipy.ndimage.rotate(raw_img, -azimuth, reshape=False, mode='nearest')
        
        # 3. Extract physics features on canonical image & scale
        raw_physics = extract_physics_features(rot_img)
        scaled_physics = self.scaler.transform(raw_physics.reshape(1, -1)).flatten()
        
        # 4. Prepare 3-channel tensor for CNN backbone
        img_3ch = np.repeat(rot_img[:, :, np.newaxis], 3, axis=-1)
        img_tensor = self.img_transform(img_3ch)
        
        return img_tensor, torch.tensor(scaled_physics, dtype=torch.float32), img_id
```

---

## 5. Evaluation Function & 2-Phase Training Protocol

### A. Concrete Validation Metric Function
```python
import numpy as np
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import balanced_accuracy_score

def evaluate_balanced_acc(model, val_loader, device):
    """
    Evaluates model on validation loader using AMP autocast and torch.no_grad().
    Handles both image and physics feature tensors.
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
```

### B. 2-Phase Training Protocol with Warmup Checkpoint Guarantee
```python
import os
import torch
from torch.cuda.amp import autocast, GradScaler

def train_model_fold(model, train_loader, val_loader, fold_idx, model_name, device, max_epochs=40, patience=7):
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/fold_{fold_idx}_{model_name}_best.pth"

    # P0: Weighted Cross-Entropy (1.75 for Class 0, 1.0 for Class 1)
    class_weights = torch.tensor([1.75, 1.0], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ---------------------------------------------------------
    # Phase 1: Head Warmup (Epochs 1–5)
    # ---------------------------------------------------------
    for p in model.backbone.parameters():
        p.requires_grad = False
    
    optimizer_p1 = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler_p1 = GradScaler()

    for epoch in range(1, 6):
        model.train()
        for images, physics_feats, labels in train_loader:
            images, physics_feats, labels = images.to(device), physics_feats.to(device), labels.to(device)
            optimizer_p1.zero_grad()
            with autocast():
                logits = model(images, physics_feats)
                loss = criterion(logits, labels)
            scaler_p1.scale(loss).backward()
            scaler_p1.step(optimizer_p1)
            scaler_p1.update()

    # Diagnostic check & GUARANTEED initial checkpoint save
    warmup_val_bacc = evaluate_balanced_acc(model, val_loader, device)
    torch.save(model.state_dict(), save_path)  # Guarantees save_path exists if Phase 2 triggers early stop
    best_val_bacc = warmup_val_bacc
    print(f"[Fold {fold_idx} | {model_name}] Warmup saved (Epoch 5). Val Balanced Acc: {warmup_val_bacc:.4f}")

    # ---------------------------------------------------------
    # Phase 2: Full Differential Fine-Tuning (Epochs 6–40)
    # ---------------------------------------------------------
    for p in model.backbone.parameters():
        p.requires_grad = True

    optimizer_p2 = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=max_epochs - 5, eta_min=1e-7
    )
    
    # Fresh GradScaler for Phase 2 to prevent scale carry-over
    scaler_p2 = GradScaler()
    epochs_no_improve = 0

    for epoch in range(6, max_epochs + 1):
        model.train()
        for images, physics_feats, labels in train_loader:
            images, physics_feats, labels = images.to(device), physics_feats.to(device), labels.to(device)
            optimizer_p2.zero_grad()
            with autocast():
                logits = model(images, physics_feats)
                loss = criterion(logits, labels)
            scaler_p2.scale(loss).backward()
            scaler_p2.step(optimizer_p2)
            scaler_p2.update()

        # Step 1: Validation evaluation
        val_bacc = evaluate_balanced_acc(model, val_loader, device)

        # Step 2: Step scheduler immediately after evaluation
        scheduler_p2.step()

        # Step 3: Checkpoint save & early stopping check
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Balanced Acc: {best_val_bacc:.4f}")
                break

    # Restore best weights with weights_only=True guard
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model, best_val_bacc
```

---

## 6. Complete 5-Fold Ensembling & Inference Protocol

### A. Reproducibility & Seeding
```python
def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### B. Out-of-Fold (OOF) Calibration & Concrete Threshold Optimization
```python
# full_oof_probs has shape (7854, 2) and full_oof_targets has shape (7854,)
best_threshold = 0.50
best_bacc = 0.0

for t in np.linspace(0.30, 0.70, 41):
    preds = (full_oof_probs[:, 1] >= t).astype(int)
    score = balanced_accuracy_score(full_oof_targets, preds)
    if score > best_bacc:
        best_bacc = score
        best_threshold = t

print(f"Optimal OOF Decision Threshold: {best_threshold:.4f} (Balanced Acc: {best_bacc:.4f})")
```

### C. Concrete Global Scaler & Test Inference (`test_metadata.csv`)
```python
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# 1. Fit Global Scaler on ALL 7,854 training samples post-CV
all_train_physics = []
for idx in range(len(train_df)):
    img_id = train_df.iloc[idx]['image_id']
    az = float(train_df.iloc[idx]['sun_azimuth_angle'])
    raw_img = cv2.imread(f"train_images/{img_id}", cv2.IMREAD_GRAYSCALE)
    rot_img = scipy.ndimage.rotate(raw_img, -az, reshape=False, mode='nearest')
    feats = extract_physics_features(rot_img)
    all_train_physics.append(feats)

global_scaler = StandardScaler()
global_scaler.fit(np.array(all_train_physics))
joblib.dump(global_scaler, "checkpoints/global_scaler.pkl")

# 2. Test Inference with 15 Checkpoints + 2 TTA Views (30 Passes Total)
test_dataset = LunarTestDataset("test_metadata.csv", "eval_images/", global_scaler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load all 15 models with weights_only=True
models = []
for fold in range(5):
    for m_name in ["efficientnet_b3", "resnet50", "convnext_tiny"]:
        m = HybridLunarClassifier(m_name, pretrained=False)
        m.load_state_dict(torch.load(f"checkpoints/fold_{fold}_{m_name}_best.pth", weights_only=True))
        m.to(device)
        m.eval()
        models.append(m)

# Run inference
all_probs = []
with torch.no_grad():
    for images, physics_feats, img_ids in test_loader:
        images = images.to(device)
        physics_feats = physics_feats.to(device)
        
        batch_probs = torch.zeros((len(images), 2), device=device)
        # 2 TTA views: original and horizontal flip
        for hflip in [False, True]:
            x_img = torch.flip(images, dims=[3]) if hflip else images
            for model in models:
                with autocast():
                    logits = model(x_img, physics_feats)
                    probs = torch.softmax(logits, dim=1)
                batch_probs += probs
                
        batch_probs /= (len(models) * 2)  # average over 30 forward passes
        all_probs.extend(batch_probs[:, 1].cpu().numpy())

# Apply calibrated threshold
final_preds = (np.array(all_probs) >= best_threshold).astype(int)
sub_df = pd.DataFrame({"image_id": test_df["image_id"], "label": final_preds})
sub_df.to_csv("submission.csv", index=False)
print(f"Submission generated successfully: {len(sub_df)} rows.")
```
