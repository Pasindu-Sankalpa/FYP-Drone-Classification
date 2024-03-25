from Libs import *
from Model import Model
from DataSet import TestDataSet
from Train_n_evaluate import Train_n_evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, "\n")

model = torch.load("/home/gevindu/model_final/Saved models/Final_model_v2.pth", map_location=device).eval()
dataset = TestDataSet()
loader = DataLoader(dataset, batch_size=128, shuffle=True)

dataset_sizes = {'test':len(dataset)}
loaders = {'test': loader}

train_n_evaluate = Train_n_evaluate(loaders, dataset_sizes, device)
train_n_evaluate.evaluate_model(model, "test")