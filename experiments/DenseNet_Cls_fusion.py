from fyp.TrainEval import Params, FusionPipeline
from fyp.Plotter import Plotter
from fyp.Dataset import load_images

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

params = Params(
    model_prefix="DenseNet",
    save_location="./DenseNet",
    data_category="fusion",
    mode="classification",
    epochs=30,
    batch_size=64,
    learning_rate=1e-3,
    save_weights=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

plotter = Plotter(params)
loaders, sizes = load_images(params, True)
pipeline = FusionPipeline(loaders, sizes, params)

class CustomFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = torch.nn.Linear(in_features=8, out_features=4, bias=True)
        self._soft_max = torch.nn.Softmax(dim=1)

    def forward(self, radar, audio):
        return self._soft_max(self._linear(torch.concat((radar, audio), dim=1)))


fusion_model = CustomFusion()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    fusion_model.parameters(),
    params.learning_rate,
    weight_decay=params.weight_decay,
)

model, losses, accuracies = pipeline.train_model(fusion_model, criterion, optimizer)

plotter.plot_learnining_curves(losses, accuracies)

actuals, predictions = pipeline.evaluate_model(dataset="test")

plotter.plot_confusion_matrix(actuals, predictions, 4)

pipeline.save_results

if params.save_weights:
    path = f"/home/gevindu/Final_work/Saved models/Compare_{params.model_name}.pth"
    torch.save(model, path)
