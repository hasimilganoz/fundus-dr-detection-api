import torch
import numpy as np

def compute_class_weights_from_imagefolder(dataset, device):

    
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)

    num_classes = len(class_counts)
    total = class_counts.sum()

    weights = total / (num_classes * class_counts)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    return weights, class_counts
