import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_resnet50(num_classes: int):

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
