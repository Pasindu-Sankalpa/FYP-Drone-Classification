from Libs import *
from Model import ClassificationModel
from DataSet import ClassificationDataSet
from Train_n_evaluate import Train_n_evaluate_classification
from Plotter import Plotter

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, "\n")

arg_dict = {"model name": "Final_model_cls_v2",
            "epochs": 40,
            "batch_size": 32,
            "lr": 1e-5,
            "weight_decay": 0.4,
            "num_classes": 4
           }

train, validation, test = random_split(ClassificationDataSet(), lengths=(0.7, 0.25, 0.05))
train_set = DataLoader(train, batch_size=arg_dict['batch_size'], shuffle=True)
validation_set = DataLoader(validation, batch_size=arg_dict['batch_size'], shuffle=True)
test_set = DataLoader(test, batch_size=arg_dict['batch_size'], shuffle=True)

dataset_sizes = {'train':len(train), 'validation':len(validation), 'test':len(test)}
loaders = {'train':train_set, 'validation':validation_set, 'test': test_set}
# print(dataset_sizes)

train_n_evaluate = Train_n_evaluate_classification(loaders, dataset_sizes, device)
plotter = Plotter(model_name=arg_dict['model name'])

model = ClassificationModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = arg_dict["lr"], weight_decay=arg_dict["weight_decay"])

model, losses, accuracies = train_n_evaluate.train_model(model, criterion, optimizer, arg_dict["epochs"], scheduler=None)
plotter.plot_learnining_curves(losses, accuracies)
actuals, predictions = train_n_evaluate.evaluate_model(model, dataset="test")
plotter.plot_confusion_matrix(actuals, predictions, arg_dict["num_classes"])

torch.save(model, f"/home/gevindu/model_final/Saved models/{arg_dict['model name']}.pth")
print(f"\nSaved to /home/gevindu/model_final/Saved models/{arg_dict['model name']}.pth")
