import os
import shutil
import pandas as pd
from tqdm import tqdm


# PATHS

EYEPACS_IMG_DIR = r"C:\Users\Hasim\Desktop\MS IN CS\PROJECTS\RETINA PROJECT\EYEPACKS\test_full\test"
EYEQ_CSV = r"C:\Users\Hasim\Desktop\MS IN CS\PROJECTS\RETINA PROJECT\EYEPACKS\Label_EyeQ_test.csv"
EYEQ_IMG_DIR = r"C:\Users\Hasim\Desktop\MS IN CS\PROJECTS\RETINA PROJECT\EYEQ\Test"

os.makedirs(EYEQ_IMG_DIR, exist_ok=True)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# LOAD CSV

print("Loading EyeQ CSV...")
df = pd.read_csv(EYEQ_CSV)

df.columns = [c.lower().strip() for c in df.columns]
df["image"] = df["image"].astype(str).str.strip()

assert "image" in df.columns, "CSV must contain 'image' column"

csv_images = set(df["image"].tolist())

print(f"Images listed in CSV: {len(csv_images)}")


# COPY IMAGES

copied = 0
missing = 0

print("Copying images (no label folders)...")
for fname in tqdm(os.listdir(EYEPACS_IMG_DIR)):
    if not fname.lower().endswith(IMG_EXTS):
        continue

    if fname not in csv_images:
        continue

    src = os.path.join(EYEPACS_IMG_DIR, fname)
    dst = os.path.join(EYEQ_IMG_DIR, fname)

    if not os.path.exists(dst):  # safety
        shutil.copy2(src, dst)
        copied += 1


# SUMMARY

missing = len(csv_images) - copied

print("\n================ SUMMARY ================")
print(f"✔ Images copied : {copied}")
print(f"⚠ Missing images (CSV but not found): {missing}")
print(f"📁 Output folder: {EYEQ_IMG_DIR}")
