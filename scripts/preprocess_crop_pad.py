import cv2
import numpy as np
import os

# PATHS

INPUT_DIR = r"C:\Users\Hasim\Desktop\MS IN CS\PROJECTS\RETINA PROJECT\EYEQ\Test"
OUTPUT_DIR = r"C:\Users\Hasim\Desktop\MS IN CS\PROJECTS\RETINA PROJECT\EYEQ\Test_preprocessed"


# HELPERS

def crop_and_pad(image, threshold=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)

    # If no foreground found return original image safely
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]

    h_c, w_c = cropped.shape[:2]
    size = max(h_c, w_c)

    # Padding to square
    top = (size - h_c) // 2
    bottom = size - h_c - top
    left = (size - w_c) // 2
    right = size - w_c - left

    squared = cv2.copyMakeBorder(
        cropped, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return squared


def process_dataset(input_dir, output_dir, extensions={".jpg", ".jpeg", ".png", ".bmp"}):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir)
             if os.path.splitext(f)[1].lower() in extensions]

    print(f"Found {len(files)} images.")

    for filename in files:
        img_path = os.path.join(input_dir, filename)

        # Load
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Cannot read image: {img_path}")
            continue

        # Crop + pad
        try:
            squared = crop_and_pad(image)
        except Exception as e:
            print(f"[ERROR] Processing failed for {filename}: {e}")
            continue

        # Save
        out_path = os.path.join(output_dir, filename)
        success = cv2.imwrite(out_path, squared)

        if success:
            print(f"Processed and saved: {out_path}")
        else:
            print(f"[ERROR] Failed to save: {out_path}")




process_dataset(INPUT_DIR, OUTPUT_DIR)










