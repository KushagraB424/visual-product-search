"""
Local fine-tuning script for EfficientNet-B0 clothing classifier.

Generates a synthetic training dataset and fine-tunes the model to produce
real confidence scores instead of the hardcoded 0.50 heuristic.

Usage:  python train_classifier.py
Output: models/weights/classifier.pth
"""

import os, sys, random, time, shutil
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────

CLOTHING_CATEGORIES = [
    "shirt", "t-shirt", "jacket", "coat", "sweater",
    "hoodie", "jeans", "pants", "shorts", "dress",
    "skirt", "blouse", "suit", "tank_top", "other",
]
NUM_CLASSES = len(CLOTHING_CATEGORIES)

DATA_DIR    = Path("training_data")
WEIGHTS_DIR = Path("models/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 1
LR         = 1e-3
WEIGHT_DECAY = 1e-4
NUM_TRAIN_PER_CLASS = 30   # synthetic images per class for training
NUM_VAL_PER_CLASS   = 10    # synthetic images per class for validation

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"🎮  GPU: {torch.cuda.get_device_name(0)}")


# ── Synthetic Data Generation ────────────────────────────────────────────

# Realistic color palettes per garment type (BGR format)
GARMENT_STYLES = {
    "shirt": {
        "colors": [(200,200,220), (180,190,210), (140,150,190), (220,210,200),
                   (100,120,180), (160,170,200), (210,200,190), (190,180,170)],
        "aspect": (0.7, 1.0),  # width/height ratio range
        "texture": "smooth",
    },
    "t-shirt": {
        "colors": [(50,50,200), (200,50,50), (50,180,50), (240,200,50),
                   (255,255,255), (30,30,30), (200,100,150), (50,150,200)],
        "aspect": (0.8, 1.1),
        "texture": "solid",
    },
    "jacket": {
        "colors": [(40,40,40), (60,50,40), (80,70,50), (50,60,80),
                   (100,80,60), (70,60,50), (90,80,70), (30,30,35)],
        "aspect": (0.7, 1.0),
        "texture": "leather",
    },
    "coat": {
        "colors": [(50,50,50), (80,70,60), (100,90,70), (60,60,80),
                   (120,100,80), (70,70,70), (90,85,75), (40,40,50)],
        "aspect": (0.5, 0.8),
        "texture": "wool",
    },
    "sweater": {
        "colors": [(150,100,80), (100,130,100), (180,160,140), (120,100,140),
                   (160,140,120), (130,110,100), (170,150,130), (140,120,110)],
        "aspect": (0.7, 1.0),
        "texture": "knit",
    },
    "hoodie": {
        "colors": [(100,100,100), (60,60,60), (140,140,160), (80,100,120),
                   (120,120,130), (70,80,90), (150,150,150), (50,50,55)],
        "aspect": (0.7, 1.0),
        "texture": "cotton",
    },
    "jeans": {
        "colors": [(120,80,40), (100,70,35), (80,60,30), (140,100,60),
                   (110,75,38), (90,65,32), (130,90,50), (105,72,36)],
        "aspect": (0.35, 0.55),
        "texture": "denim",
    },
    "pants": {
        "colors": [(40,40,40), (60,60,60), (100,80,60), (80,80,100),
                   (50,50,55), (70,70,75), (90,70,50), (45,45,50)],
        "aspect": (0.35, 0.55),
        "texture": "smooth",
    },
    "shorts": {
        "colors": [(100,80,40), (180,160,120), (60,80,120), (200,180,140),
                   (120,100,60), (150,130,90), (80,100,140), (170,150,110)],
        "aspect": (0.8, 1.3),
        "texture": "casual",
    },
    "dress": {
        "colors": [(200,50,80), (80,50,150), (50,150,150), (200,180,50),
                   (180,40,60), (60,40,130), (40,130,130), (180,160,40)],
        "aspect": (0.4, 0.65),
        "texture": "flowing",
    },
    "skirt": {
        "colors": [(180,50,100), (100,50,150), (200,150,100), (50,100,150),
                   (160,40,80), (80,40,130), (180,130,80), (40,80,130)],
        "aspect": (0.7, 1.2),
        "texture": "pleated",
    },
    "blouse": {
        "colors": [(220,200,200), (200,220,220), (180,200,220), (220,200,180),
                   (210,190,190), (190,210,210), (170,190,210), (210,190,170)],
        "aspect": (0.7, 1.0),
        "texture": "silk",
    },
    "suit": {
        "colors": [(40,40,50), (50,50,60), (60,50,40), (30,30,40),
                   (35,35,45), (45,45,55), (55,45,35), (25,25,35)],
        "aspect": (0.5, 0.7),
        "texture": "formal",
    },
    "tank_top": {
        "colors": [(200,60,60), (60,60,200), (200,200,60), (255,200,200),
                   (180,50,50), (50,50,180), (180,180,50), (235,180,180)],
        "aspect": (0.8, 1.2),
        "texture": "thin",
    },
    "other": {
        "colors": [(128,128,128), (160,140,120), (100,120,140), (140,160,140),
                   (118,118,118), (150,130,110), (90,110,130), (130,150,130)],
        "aspect": (0.6, 1.0),
        "texture": "mixed",
    },
}

TEXTURE_PATTERNS = {
    "smooth":  lambda img, rng: _add_gradient(img, rng, strength=0.15),
    "solid":   lambda img, rng: img,
    "leather": lambda img, rng: _add_noise(img, rng, strength=12),
    "wool":    lambda img, rng: _add_knit_pattern(img, rng),
    "knit":    lambda img, rng: _add_knit_pattern(img, rng),
    "cotton":  lambda img, rng: _add_noise(img, rng, strength=8),
    "denim":   lambda img, rng: _add_denim(img, rng),
    "casual":  lambda img, rng: _add_noise(img, rng, strength=6),
    "flowing": lambda img, rng: _add_gradient(img, rng, strength=0.25),
    "pleated": lambda img, rng: _add_stripes(img, rng, horizontal=False),
    "silk":    lambda img, rng: _add_gradient(img, rng, strength=0.1),
    "formal":  lambda img, rng: _add_pinstripe(img, rng),
    "thin":    lambda img, rng: img,
    "mixed":   lambda img, rng: _add_noise(img, rng, strength=10),
}


def _add_gradient(img, rng, strength=0.2):
    h, w = img.shape[:2]
    direction = rng.choice(["vertical", "horizontal", "diagonal"])
    grad = np.zeros((h, w), dtype=np.float32)
    if direction == "vertical":
        for y in range(h):
            grad[y, :] = (y / h - 0.5) * strength
    elif direction == "horizontal":
        for x in range(w):
            grad[:, x] = (x / w - 0.5) * strength
    else:
        for y in range(h):
            for x in range(w):
                grad[y, x] = ((y / h + x / w) / 2 - 0.5) * strength
    result = img.astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 + grad)
    return np.clip(result, 0, 255).astype(np.uint8)


def _add_noise(img, rng, strength=10):
    noise = rng.randint(-strength, strength + 1, size=img.shape)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _add_knit_pattern(img, rng):
    h, w = img.shape[:2]
    spacing = rng.randint(4, 8)
    result = img.copy()
    for y in range(0, h, spacing):
        brightness = rng.randint(-15, 5)
        result[y:min(y + 1, h), :] = np.clip(
            result[y:min(y + 1, h), :].astype(np.int16) + brightness, 0, 255
        ).astype(np.uint8)
    return result


def _add_denim(img, rng):
    img = _add_noise(img, rng, strength=8)
    h, w = img.shape[:2]
    # Diagonal weave
    for i in range(0, h + w, rng.randint(3, 6)):
        y1 = max(0, i - w)
        x1 = max(0, i - h)
        y2 = min(h - 1, i)
        x2 = min(w - 1, i)
        cv2.line(img, (x1, y1), (x2, y2), 
                 tuple(int(c) for c in img[h // 2, w // 2] + rng.randint(-5, 6, 3)),
                 1)
    return img


def _add_stripes(img, rng, horizontal=True):
    h, w = img.shape[:2]
    spacing = rng.randint(6, 14)
    result = img.copy()
    if horizontal:
        for y in range(0, h, spacing):
            cv2.line(result, (0, y), (w, y),
                     tuple(int(c) for c in img[h // 2, w // 2] * 0.8), 1)
    else:
        for x in range(0, w, spacing):
            cv2.line(result, (x, 0), (x, h),
                     tuple(int(c) for c in img[h // 2, w // 2] * 0.85), 1)
    return result


def _add_pinstripe(img, rng):
    h, w = img.shape[:2]
    spacing = rng.randint(8, 16)
    result = img.copy()
    stripe_color = np.clip(img[h // 2, w // 2].astype(np.int16) + 25, 0, 255).astype(np.uint8)
    for x in range(0, w, spacing):
        cv2.line(result, (x, 0), (x, h), tuple(int(c) for c in stripe_color), 1)
    return result


def generate_garment_image(category, rng, size=IMG_SIZE):
    """Generate a realistic synthetic garment image."""
    style = GARMENT_STYLES[category]
    
    # Background (studio-like)
    bg_type = rng.choice(["white", "light_gray", "gradient"])
    if bg_type == "white":
        img = np.full((size, size, 3), rng.randint(235, 256), dtype=np.uint8)
    elif bg_type == "light_gray":
        val = rng.randint(200, 240)
        img = np.full((size, size, 3), val, dtype=np.uint8)
    else:
        img = np.full((size, size, 3), 245, dtype=np.uint8)
        for y in range(size):
            t = y / size
            val = int(240 * (1 - t) + 210 * t)
            img[y, :] = val

    # Garment shape
    color = np.array(style["colors"][rng.randint(len(style["colors"]))], dtype=np.uint8)
    aspect_lo, aspect_hi = style["aspect"]
    aspect = rng.uniform(aspect_lo, aspect_hi)
    
    # Size of garment (60-85% of image)
    garment_h = int(size * rng.uniform(0.6, 0.85))
    garment_w = int(garment_h * aspect)
    garment_w = min(garment_w, int(size * 0.9))
    
    # Center position with slight jitter
    cx = size // 2 + rng.randint(-10, 11)
    cy = size // 2 + rng.randint(-10, 11)
    
    x1 = max(0, cx - garment_w // 2)
    y1 = max(0, cy - garment_h // 2)
    x2 = min(size, x1 + garment_w)
    y2 = min(size, y1 + garment_h)
    
    # Create garment patch with color variation
    patch_h = y2 - y1
    patch_w = x2 - x1
    if patch_h < 10 or patch_w < 10:
        return img
    
    # Base color with subtle variation
    garment = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
    for c in range(3):
        base = int(color[c])
        variation = rng.randint(-8, 9, size=(patch_h, patch_w))
        garment[:, :, c] = np.clip(base + variation, 0, 255).astype(np.uint8)
    
    # Apply texture
    texture_fn = TEXTURE_PATTERNS.get(style["texture"], lambda x, r: x)
    garment = texture_fn(garment, rng)
    
    # Add shape variation (not just rectangle)
    mask = np.ones((patch_h, patch_w), dtype=np.uint8) * 255
    
    # Round corners
    corner_radius = rng.randint(5, min(15, min(patch_h, patch_w) // 4))
    # Top-left
    cv2.circle(mask, (corner_radius, corner_radius), corner_radius, 0, -1)
    cv2.rectangle(mask, (0, 0), (corner_radius, corner_radius), 0, -1)
    cv2.circle(mask, (corner_radius, corner_radius), corner_radius, 255, -1)
    # Top-right
    cv2.circle(mask, (patch_w - corner_radius, corner_radius), corner_radius, 0, -1)
    cv2.rectangle(mask, (patch_w - corner_radius, 0), (patch_w, corner_radius), 0, -1)
    cv2.circle(mask, (patch_w - corner_radius, corner_radius), corner_radius, 255, -1)
    
    # Apply garment to image
    mask_3 = np.stack([mask, mask, mask], axis=-1).astype(np.float32) / 255
    img[y1:y2, x1:x2] = (garment * mask_3 + img[y1:y2, x1:x2] * (1 - mask_3)).astype(np.uint8)
    
    # Random augmentations
    if rng.random() < 0.3:
        # Brightness
        factor = rng.uniform(0.8, 1.2)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        # Horizontal flip
        img = cv2.flip(img, 1)
    if rng.random() < 0.2:
        # Slight rotation
        angle = rng.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (size, size), borderValue=(240, 240, 240))
    if rng.random() < 0.2:
        # Blur
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    
    return img


def create_dataset():
    """Generate synthetic training and validation datasets."""
    print("📦 Generating synthetic training dataset...")
    
    rng = np.random.RandomState(SEED)
    
    for split, count in [("train", NUM_TRAIN_PER_CLASS), ("val", NUM_VAL_PER_CLASS)]:
        for cat in CLOTHING_CATEGORIES:
            cat_dir = DATA_DIR / split / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                img = generate_garment_image(cat, rng)
                cv2.imwrite(str(cat_dir / f"{cat}_{i:04d}.jpg"), img)
        
        total = count * NUM_CLASSES
        print(f"  ✅ {split}: {total} images ({count} per class × {NUM_CLASSES} classes)")
    
    print(f"📁 Dataset saved to: {DATA_DIR}")


# ── Dataset Class ────────────────────────────────────────────────────────

class ClothingDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.samples = []
        self.labels = []
        self.augment = augment
        
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.02),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        
        split_dir = Path(root) / split
        for cat_idx, cat in enumerate(CLOTHING_CATEGORIES):
            cat_dir = split_dir / cat
            if not cat_dir.exists():
                continue
            for img_path in sorted(cat_dir.glob("*.jpg")):
                self.samples.append(str(img_path))
                self.labels.append(cat_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


# ── Training ─────────────────────────────────────────────────────────────

def train():
    print(f"\n{'='*60}")
    print("🏋️  Fine-tuning EfficientNet-B0 for {NUM_CLASSES}-class clothing classification")
    print(f"{'='*60}\n")
    
    # Create datasets
    train_ds = ClothingDataset(DATA_DIR, "train", augment=True)
    val_ds   = ClothingDataset(DATA_DIR, "val", augment=False)
    
    print(f"📊 Train: {len(train_ds)} samples")
    print(f"📊 Val:   {len(val_ds)} samples\n")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    
    # Model
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.clone().detach().long().to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)
        
        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        
        scheduler.step()
        
        t_loss = train_loss / max(train_total, 1)
        t_acc  = train_correct / max(train_total, 1)
        v_loss = val_loss / max(val_total, 1)
        v_acc  = val_correct / max(val_total, 1)
        
        marker = ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " ⭐ best"
        
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} │ "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.3f} │ "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.3f} │ "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}{marker}")
    
    # Save best model
    out_path = WEIGHTS_DIR / "classifier.pth"
    torch.save(best_state, str(out_path))
    print(f"\n✅ Best validation accuracy: {best_val_acc:.3f}")
    print(f"💾 Saved fine-tuned weights to: {out_path}")
    print(f"   File size: {out_path.stat().st_size / 1e6:.1f} MB")
    
    # ── Quick confidence check ──
    print(f"\n{'='*60}")
    print("📊 Confidence Score Check")
    print(f"{'='*60}")
    
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    
    all_confs = []
    all_correct = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(1)
            all_confs.extend(confs.cpu().numpy().tolist())
            all_correct.extend((preds == labels).cpu().numpy().tolist())
    
    avg_conf = np.mean(all_confs)
    avg_correct_conf = np.mean([c for c, ok in zip(all_confs, all_correct) if ok])
    accuracy = np.mean(all_correct)
    
    print(f"  Before fine-tuning:  Confidence = 50.0% (hardcoded heuristic)")
    print(f"  After fine-tuning:   Confidence = {avg_conf*100:.1f}% (average)")
    print(f"  Correct predictions: Confidence = {avg_correct_conf*100:.1f}% (average)")
    print(f"  Overall accuracy:    {accuracy*100:.1f}%")
    print(f"\n  Improvement: +{(avg_conf - 0.50) * 100:.1f}% confidence increase! 🎉")
    
    print(f"\n{'='*60}")
    print("🚀 Restart the server to use the new weights:")
    print("   uvicorn app.main:app --reload --port 8001")
    print(f"{'='*60}")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    
    if not (DATA_DIR / "train").exists():
        create_dataset()
    else:
        print("📁 Training data already exists, skipping generation.")
    
    train()
    
    elapsed = time.time() - t0
    print(f"\n⏱️  Total time: {elapsed / 60:.1f} minutes")
