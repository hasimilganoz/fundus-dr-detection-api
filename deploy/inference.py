import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

from models.resnet50 import build_resnet50
from data.transforms import get_val_transforms

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cpu")

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "weights" / "BEST.pt"

# -----------------------------
# Model
# -----------------------------
model = build_resnet50(num_classes=2)
model.load_state_dict(
    torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
)
model.to(DEVICE)
model.eval()

# -----------------------------
# Transform (SAME AS TEST)
# -----------------------------
transform = get_val_transforms()

# -----------------------------
# Inference
# -----------------------------
def predict(image: Image.Image):
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        dr_prob = probs[:, 1].item()
        pred_class = torch.argmax(probs, dim=1).item()

    return {
        "prediction": "DR" if pred_class == 1 else "Non-DR",
        "dr_probability": round(dr_prob, 4)
    }
