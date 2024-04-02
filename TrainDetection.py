from Libs import *
from Model import DetectionModel
from DataSet import DetectionDataSet
from Train_n_evaluate import Train_n_evaluate_detection
from Plotter import Plotter

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

arg_dict = {"model name": "Final_model_det_v2",
            "epochs": 40,
            "batch_size": 32,
            "lr": 5e-5,
            "weight_decay": 0.4,
            "num_classes": 2
           }

train, validation, test = random_split(DetectionDataSet(verbose=False), lengths=(0.7, 0.25, 0.05))
train_set = DataLoader(train, batch_size=arg_dict['batch_size'], shuffle=True)
validation_set = DataLoader(validation, batch_size=arg_dict['batch_size'], shuffle=True)
test_set = DataLoader(test, batch_size=arg_dict['batch_size'], shuffle=True)

dataset_sizes = {'train':len(train), 'validation':len(validation), 'test':len(test)}
loaders = {'train':train_set, 'validation':validation_set, 'test': test_set}
# print(dataset_sizes)

train_n_evaluate = Train_n_evaluate_detection(loaders, dataset_sizes, device)
plotter = Plotter(model_name=arg_dict['model name'])

model = DetectionModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = arg_dict["lr"], weight_decay=arg_dict["weight_decay"], momentum=0.7)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(10, 20, 30), gamma=0.5, verbose=True)

model, losses, accuracies = train_n_evaluate.train_model(model, criterion, optimizer, arg_dict["epochs"], scheduler=lr_scheduler)
plotter.plot_learnining_curves(losses, accuracies)
actuals, predictions = train_n_evaluate.evaluate_model(model, dataset="test")
plotter.plot_confusion_matrix(actuals, predictions, arg_dict["num_classes"])

torch.save(model, f"/home/gevindu/model_final/Saved models/{arg_dict['model name']}.pth")
print(f"\nSaved to /home/gevindu/model_final/Saved models/{arg_dict['model name']}.pth")
