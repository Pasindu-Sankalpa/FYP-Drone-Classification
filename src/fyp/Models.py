import torch
import torch.nn as nn
import torchvision.models as models


class TransferResNetDetection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        for param in self._base.parameters():
            param.requires_grad = False

        self._base.fc = nn.Linear(in_features=2048, out_features=2)
        for param in self._base.fc.parameters():
            param.requires_grad = True

    def forward(self, X):
        return self._base(X)


class TransferDenseNetDetection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._base = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        for param in self._base.parameters():
            param.requires_grad = False

        self._base.classifier = nn.Linear(in_features=1920, out_features=2)
        for param in self._base.classifier.parameters():
            param.requires_grad = True

    def forward(self, X):
        return self._base(X)


class TransferResNetClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        for param in self._base.parameters():
            param.requires_grad = False

        self._base.fc = nn.Linear(in_features=2048, out_features=5)
        for param in self._base.fc.parameters():
            param.requires_grad = True

    def forward(self, X):
        return self._base(X)


class TransferDenseNetClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._base = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        for param in self._base.parameters():
            param.requires_grad = False

        self._base.classifier = nn.Linear(in_features=1920, out_features=5)
        for param in self._base.classifier.parameters():
            param.requires_grad = True

    def forward(self, X):
        return self._base(X)
