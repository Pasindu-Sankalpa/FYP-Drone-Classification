from fyp.Plotter import Plotter
from fyp.TrainEval import Pipeline, Params
from fyp.Dataset import load_images
from fyp.Models import TransferDenseNetClassification

import os
import torch
from torch import nn
from torch import optim

params = Params(
    model_prefix="DenseNet",
    save_location="./DenseNet",
    data_category="rangeDoppler",
    mode="classification",
    epochs=50,
    batch_size = 1024,
    learning_rate = 1e-4,
    save_weights=True,
    device = "cuda" if torch.cuda.is_available() else "cpu"
)

plotter = Plotter(params)
loaders, dataset_sizes = load_images(params)
pipeline = Pipeline(loaders, dataset_sizes, params)

model = TransferDenseNetClassification()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    params.learning_rate,
    weight_decay=params.weight_decay,
)

model, losses, accuracies = pipeline.train_model(model, criterion, optimizer)

plotter.plot_learnining_curves(losses, accuracies)

actuals, predictions = pipeline.evaluate_model(dataset="test")

plotter.plot_confusion_matrix(actuals, predictions, params.num_classes)

pipeline.save_results

if params.save_weights:
    path = f"/home/gevindu/Final_work/Saved models/Compare_{params.model_name}.pth"
    torch.save(model, path)
