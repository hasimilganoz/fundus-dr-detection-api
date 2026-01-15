import yaml
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)

from data.transforms import get_val_transforms
from models.resnet50 import build_resnet50


# =====================================================
# PATHS
# =====================================================
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "train.yaml"


# =====================================================
# ROC CURVE FUNCTION
# =====================================================
def plot_roc_curve(y_true: List[int], y_scores: List[float], title: str):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


# =====================================================
# TEST EVALUATION
# =====================================================
def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:

    model.eval()

    preds = []
    targets_all = []
    probs_all = []

    loss_sum = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Test", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                probs = torch.softmax(outputs, dim=1)[:, 1]

            batch_size = inputs.size(0)
            loss_sum += loss.item() * batch_size
            num_samples += batch_size

            preds.extend(outputs.argmax(1).cpu().tolist())
            targets_all.extend(targets.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())

    metrics = {
        "loss": loss_sum / max(1, num_samples),
        "accuracy": accuracy_score(targets_all, preds),
        "f1": f1_score(targets_all, preds, average="macro", zero_division=0),
        "precision": precision_score(targets_all, preds, average="macro", zero_division=0),
        "recall": recall_score(targets_all, preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(targets_all, preds),
        "y_true": targets_all,
        "y_scores": probs_all,
    }

    return metrics


# =====================================================
# MAIN
# =====================================================
def main():
    # -----------------------------
    # Load config
    # -----------------------------
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Test Dataset
    # -----------------------------
    test_dir = cfg["paths"]["test_dir"]

    test_dataset = ImageFolder(
        root=test_dir,
        transform=get_val_transforms(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    print(f"📊 Test samples: {len(test_dataset)}")

    # -----------------------------
    # Model
    # -----------------------------
    model = build_resnet50(num_classes=cfg["num_classes"])
    model.to(device)

    # -----------------------------
    # Load BEST weights
    # -----------------------------
    weights_dir = Path(cfg["paths"]["output_dir"])
    best_ckpt = weights_dir / "BEST.pt"

    assert best_ckpt.exists(), f"BEST.pt not found in {weights_dir}"

    model.load_state_dict(
        torch.load(best_ckpt, map_location=device, weights_only=True)
    )
    print(f"✅ Loaded weights: {best_ckpt}")

    # -----------------------------
    # Loss
    # -----------------------------
    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # Evaluation
    # -----------------------------
    metrics = evaluate_test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\n===== TEST RESULTS =====")
    print(f"Loss      : {metrics['loss']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"F1-score  : {metrics['f1']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    # -----------------------------
    # ROC Curve
    # -----------------------------
    plot_roc_curve(
        y_true=metrics["y_true"],
        y_scores=metrics["y_scores"],
        title="Test ROC Curve (DR Classification)",
    )


if __name__ == "__main__":
    main()
