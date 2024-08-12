from fyp.Plotter import Plotter
from fyp.TrainEval import Pipeline, Params
from fyp.Dataset import load_images
from fyp.Models import TransferResNetDetection

import os
import torch
from torch import nn
from torch import optim

params = Params(
    model_prefix="ResNet",
    save_location="./ResNet",
    data_category="mel",
    mode="detection",
    epochs=30,
    batch_size = 1024,
    learning_rate = 2.5e-4,
    save_weights=True,
    device = "cuda" if torch.cuda.is_available() else "cpu"
)

plotter = Plotter(params)
loaders, dataset_sizes = load_images(params)
pipeline = Pipeline(loaders, dataset_sizes, params)

model = TransferResNetDetection()
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
    if params.save_location is not None and not os.path.isdir(params.save_location):
        os.mkdir(params.save_location)
    path = (
        params.save_location + "/" + params.model_name + ".pth"
        if params.save_location is not None
        else params.model_name + ".pth"
    )
    torch.save(model, path)
