from Libs import *
from Model import Model
from DataSet import DataSet
from Train_n_evaluate import Train_n_evaluate
from Plotter import Plotter

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, "\n")


arg_dict = {"model name": "Final_model",
            "epochs": 2,
            "batch_size": 32,
            "lr": 1e-5,
            "weight_decay": 0.4,
            "num_classes": 2
           }

plotter = Plotter(model_name=arg_dict['model name'])
train, validation = random_split(DataSet(), lengths=(0.85, 0.15))
train_set = DataLoader(train, batch_size=arg_dict['batch_size'], shuffle=True)
validation_set = DataLoader(validation, batch_size=arg_dict['batch_size'], shuffle=True)

dataset_sizes = {'train':len(train), 'validation':len(validation)}
loaders = {'train':train_set, 'validation':validation_set}
train_n_evaluate = Train_n_evaluate(loaders, dataset_sizes, device)

model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = arg_dict["lr"], weight_decay=arg_dict["weight_decay"])

model, losses, accuracies = train_n_evaluate.train_model(model, criterion, optimizer, arg_dict["epochs"], scheduler=None)
plotter.plot_learnining_curves(losses, accuracies)
# actuals, predictions = train_n_evaluate.evaluate_model(model, dataset="validation")
# plotter.plot_confusion_matrix(actuals, predictions, arg_dict["num_classes"])
# plotter.plot_lr(scheduler.get_lr_schedule())

# torch.save(model, f"/home/gevindu/Gevindu/Models/{arg_dict['model name']}.pth")
# print(f"\nSaved to /home/gevindu/Gevindu/Models/{arg_dict['model name']}.pth")

# print(model)