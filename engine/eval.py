from typing import Dict, List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    """

    model.eval()

    preds: List[int] = []
    targets_all: List[int] = []
    loss_sum = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Val", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            loss_sum += loss.item() * batch_size
            num_samples += batch_size

            preds.extend(outputs.argmax(1).cpu().tolist())
            targets_all.extend(targets.cpu().tolist())

    metrics = {
        "loss": loss_sum / max(1, num_samples),
        "accuracy": accuracy_score(targets_all, preds),
        "f1": f1_score(targets_all, preds, average="macro", zero_division=0),
        "precision": precision_score(targets_all, preds, average="macro", zero_division=0),
        "recall": recall_score(targets_all, preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(targets_all, preds),
    }

    return metrics
