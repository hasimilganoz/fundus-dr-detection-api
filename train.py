import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.datasets import build_datasets
from data.transforms import get_train_transforms, get_val_transforms
from models.resnet50 import build_resnet50
from engine.train import train_one_epoch
from engine.eval import evaluate


# =====================================================
# PATHS
# =====================================================
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "train.yaml"


def main():
    # -------------------------------------------------
    # Load config
    # -------------------------------------------------
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Dataset (ImageFolder-based)
    # -------------------------------------------------
    train_ds, val_ds = build_datasets(
        train_dir=cfg["paths"]["train_dir"],
        val_split=cfg["val_split"],
        seed=cfg["seed"],
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms(),
    )

    # -------------------------------------------------
    # 🔥 OVERSAMPLING via WeightedRandomSampler (TRAIN ONLY)
    # -------------------------------------------------
    targets = [train_ds.dataset.targets[i] for i in train_ds.indices]
    class_counts = np.bincount(targets)

    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        sampler=sampler,          # 🔥 oversampling burada
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = build_resnet50(num_classes=cfg["num_classes"])
    model.to(device)

    # -------------------------------------------------
    # Loss / Optimizer
    # -------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = GradScaler()

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    early_stop_patience = cfg.get("early_stopping", {}).get("patience", None)
    epochs_no_improve = 0

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\n===== EPOCH {epoch}/{cfg['epochs']} =====")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print("Train:", train_metrics)
        print(
            "Val:",
            {k: v for k, v in val_metrics.items() if k != "confusion_matrix"},
        )
        print("Confusion Matrix:\n", val_metrics["confusion_matrix"])

        # ---------------------------------------------
        # Checkpointing
        # ---------------------------------------------
        ckpt_path = output_dir / f"epoch_{epoch:02d}_valF1_{val_metrics['f1']:.4f}.pt"
        torch.save(model.state_dict(), ckpt_path)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "BEST.pt")
            print(f"🔥 New BEST model saved (F1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve} epoch(s)")

        # ---------------------------------------------
        # Early stopping
        # ---------------------------------------------
        if early_stop_patience is not None and epochs_no_improve >= early_stop_patience:
            print(
                f"🛑 Early stopping triggered after "
                f"{epochs_no_improve} epochs without improvement."
            )
            break

    print("\n🎉 Training complete")
    print(f"🏆 Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
