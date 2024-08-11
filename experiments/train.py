from fyp.Plotter import Plotter
from fyp.TrainEval import Pipeline, Params
from fyp.Dataset import load_images
from fyp.Models import TransferDenseNetDetection

import torch
from torch import nn
from torch import optim

device = "cuda" if torch.cuda.is_available() else "cpu"

DenseNetDet_params = Params(
    model_prefix="sample",
    data_category="mel",
    mode="detection",
    save_location="./sample",
    save_weights=True,
    epochs = 5
)

plotter = Plotter(
    save_loc=DenseNetDet_params.save_location, model_name=DenseNetDet_params.model_name
)
loaders, dataset_sizes = load_images(
    DenseNetDet_params.data_category,
    DenseNetDet_params.mode,
    DenseNetDet_params.batch_size,
    DenseNetDet_params.lengths,
)

model = TransferDenseNetDetection()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    DenseNetDet_params.learning_rate,
    weight_decay=DenseNetDet_params.weight_decay,
)

pipeline = Pipeline(loaders, dataset_sizes, device)
model, losses, accuracies = pipeline.train_model(
    model, criterion, optimizer, DenseNetDet_params.epochs
)

plotter.plot_learnining_curves(losses, accuracies)
actuals, predictions = pipeline.evaluate_model(dataset="test")
plotter.plot_confusion_matrix(actuals, predictions, DenseNetDet_params.num_classes)

if DenseNetDet_params.save_weights:
    path = (
        DenseNetDet_params.save_location + "/" + DenseNetDet_params.model_name + ".pth"
        if DenseNetDet_params.save_location is not None
        else DenseNetDet_params.model_name + ".pth"
    )
    torch.save(model, path)
