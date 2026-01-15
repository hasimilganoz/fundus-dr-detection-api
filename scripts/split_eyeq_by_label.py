import os
import shutil
import yaml
import pandas as pd
from tqdm import tqdm


def split_eyeq_dataset(image_dir: str, csv_path: str):
    """
    Split EyeQ images into:
        image_dir/
          ├── 0_NO_DR/
          ├── 1_DR/
          └── original images (*.jpeg)
    """

    no_dr_dir = os.path.join(image_dir, "0_NO_DR")
    dr_dir = os.path.join(image_dir, "1_DR")

    os.makedirs(no_dr_dir, exist_ok=True)
    os.makedirs(dr_dir, exist_ok=True)

    # -----------------------------
    # Load CSV
    # -----------------------------
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["image"] = df["image"].astype(str).str.strip()

    # Keep valid labels
    df = df[df["dr_grade"].isin([0, 3, 4])].copy()
    df["label"] = df["dr_grade"].apply(lambda x: 0 if x == 0 else 1)

    copied, missing = 0, 0

    print(f"\n📂 Processing folder: {image_dir}")
    print(f"📄 Using CSV        : {csv_path}")

    # -----------------------------
    # Copy images
    # -----------------------------
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src = os.path.join(image_dir, row["image"])

        if not os.path.exists(src):
            missing += 1
            continue

        dst = (
            os.path.join(no_dr_dir, row["image"])
            if row["label"] == 0
            else os.path.join(dr_dir, row["image"])
        )

        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    # -----------------------------
    # Summary
    # -----------------------------
    print("✅ DONE")
    print(f"   Copied images : {copied}")
    print(f"   Missing files : {missing}")
    print(f"   0_NO_DR count : {len(os.listdir(no_dr_dir))}")
    print(f"   1_DR count    : {len(os.listdir(dr_dir))}")


if __name__ == "__main__":

    # -----------------------------
    # Load config
    # -----------------------------
    with open("configs/train.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]

    # -----------------------------
    # TRAIN
    # -----------------------------
    split_eyeq_dataset(
        image_dir=paths["train_images"],
        csv_path=paths["train_csv"],
    )

    # -----------------------------
    # TEST
    # -----------------------------
    split_eyeq_dataset(
        image_dir=paths["test_images"],
        csv_path=paths["test_csv"],
    )
