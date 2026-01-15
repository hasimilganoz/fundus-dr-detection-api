import os
import pandas as pd

# PATHS

CSV_PATH = r"C:\Users\Hasim\Desktop\MS IN CS\PROJECTS\RETINA PROJECT\EYEPACKS\Label_EyeQ_test.csv"
EYEQ_DIR = r"C:\Users\Hasim\Desktop\MS IN CS\PROJECTS\RETINA PROJECT\EYEQ\Test_preprocessed"


# LOAD CSV

df = pd.read_csv(CSV_PATH)
df.columns = [c.lower().strip() for c in df.columns]

IMAGE_COL = "image"
QUALITY_COL = "quality"


# REMOVE BAD QUALITY IMAGES

to_remove = df[df[QUALITY_COL] == 2][IMAGE_COL].astype(str)

deleted = 0
skipped = 0

for fname in to_remove:
    path = os.path.join(EYEQ_DIR, fname)
    if os.path.exists(path):
        os.remove(path)
        deleted += 1
    else:
        skipped += 1


# SUMMARY

print("========== DELETE SUMMARY ==========")
print(f"Deleted images : {deleted}")
print(f"Missing files  : {skipped}")
print("✔ Quality=2 images removed.")
