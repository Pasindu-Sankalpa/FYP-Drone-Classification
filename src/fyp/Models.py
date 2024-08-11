import torch
import torch.nn as nn
import torchvision.models as models

class TransferResNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.base = models.resnet101(weights=True)
        self.classifier_model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                              nn.ReLU(),
                                              nn.Linear(hidden_size, output_size)
                                              )
    def forward(self, X):
        return self.classifier_model(X)
    