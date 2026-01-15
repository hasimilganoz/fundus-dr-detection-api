from pathlib import Path

from data.datasets import build_datasets
from data.transforms import get_train_transforms, get_val_transforms

# -------------------------------------------------
# PATH
# -------------------------------------------------
TRAIN_DIR = r"C:\Users\Hasim\Desktop\MS IN CS\PROJECTS\RETINA PROJECT\EYEQ\Train_preprocessed"

# -------------------------------------------------
# Build datasets
# -------------------------------------------------
train_ds, val_ds = build_datasets(
    train_dir=TRAIN_DIR,
    val_split=0.2,
    seed=42,
    train_transform=get_train_transforms(),
    val_transform=get_val_transforms(),
)

# -------------------------------------------------
# Basic checks
# -------------------------------------------------
print("\n=== DATASET SIZES ===")
print(f"Train dataset: {len(train_ds)}")
print(f"Val dataset  : {len(val_ds)}")

print("\n=== SAMPLE CHECK (TRAIN) ===")
for i in range(5):
    img, lbl = train_ds[i]
    print(f"Sample {i}: label={lbl}, image shape={img.shape}")

print("\n=== SAMPLE CHECK (VAL) ===")
for i in range(5):
    img, lbl = val_ds[i]
    print(f"Sample {i}: label={lbl}, image shape={img.shape}")
