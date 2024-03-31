from Libs import *
from Model import CombinedModel
from DataSet import CombinedDataSet
from Train_n_evaluate import Train_n_evaluate_combined
from Plotter import Plotter

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, "\n")

arg_dict = {
    "model name": "Final_model_v5_comb",
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-5,
    "weight_decay": 0.4
}

train, validation, test = random_split(
    CombinedDataSet(), lengths=(0.7, 0.25, 0.05)
)
train_set = DataLoader(train, batch_size=arg_dict["batch_size"], shuffle=True)
validation_set = DataLoader(validation, batch_size=arg_dict["batch_size"], shuffle=True)
test_set = DataLoader(test, batch_size=arg_dict["batch_size"], shuffle=True)

dataset_sizes = {"train": len(train), "validation": len(validation), "test": len(test)}
loaders = {"train": train_set, "validation": validation_set, "test": test_set}
# print(dataset_sizes)

train_n_evaluate = Train_n_evaluate_combined(loaders, dataset_sizes, device)
plotter = Plotter(model_name=arg_dict["model name"])

model = CombinedModel().to(device)

criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(
    model.parameters(), lr=arg_dict["lr"], weight_decay=arg_dict["weight_decay"]
)

model, losses, det_losses, cls_losses, det_accuracies, cls_accuracies = (
    train_n_evaluate.train_model(
        model, criterion, optimizer, arg_dict["epochs"], scheduler=None
    )
)

plotter.plot_learnining_curves(det_losses, det_accuracies, figure_name="Detection learning curves")
plotter.plot_learnining_curves(cls_losses, cls_accuracies, figure_name="Classification learning curves")
det_accuracies["train"] = [0 for _ in det_accuracies["train"]]
det_accuracies["validation"] = [0 for _ in det_accuracies["validation"]]
plotter.plot_learnining_curves(losses, det_accuracies, figure_name="Learning curves")

actuals, predictions = train_n_evaluate.evaluate_model(model, dataset="test", mode=0)
plotter.plot_confusion_matrix(actuals, predictions, 2, figure_name="Detection confusion matrix")
actuals, predictions = train_n_evaluate.evaluate_model(model, dataset="test", mode=1)
plotter.plot_confusion_matrix(actuals, predictions, 4, figure_name="Classification confusion matrix")

torch.save(model, f"/home/gevindu/model_final/Saved models/{arg_dict['model name']}.pth")
print(f"\nSaved to /home/gevindu/model_final/Saved models/{arg_dict['model name']}.pth")

