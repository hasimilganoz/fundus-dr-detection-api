from typing import Dict, List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device,
    scaler: GradScaler,
) -> Dict[str, float]:
    """
    Train model for ONE epoch.
    """

    model.train()

    preds: List[int] = []
    targets_all: List[int] = []
    loss_sum = 0.0
    num_samples = 0

    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = inputs.size(0)
        loss_sum += loss.item() * batch_size
        num_samples += batch_size

        preds.extend(outputs.argmax(1).detach().cpu().tolist())
        targets_all.extend(targets.detach().cpu().tolist())

    metrics = {
        "loss": loss_sum / max(1, num_samples),
        "accuracy": accuracy_score(targets_all, preds),
        "f1": f1_score(targets_all, preds, average="macro", zero_division=0),
        "precision": precision_score(targets_all, preds, average="macro", zero_division=0),
        "recall": recall_score(targets_all, preds, average="macro", zero_division=0),
    }

    return metrics
