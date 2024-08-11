import copy
import time
from tqdm import tqdm

import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score


class Pipeline:
    def __init__(self, model_name, loaders, dataset_sizes, device):
        self.model_name = model_name
        self.loaders = loaders
        self.dataset_sizes = dataset_sizes
        self.device = device

    def train_model(self, model, criterion, optimizer, epochs, scheduler):
        losses = {"train": [], "validation": []}
        accuracies = {"train": [], "validation": []}

        best_acc = 0.0
        since = time.time()
        best_model = copy.deepcopy(model.state_dict())

        for epoch in range(epochs):
            for phase in ["train", "validation"]:
                if phase == "train":
                    model.train()
                    print("Epoch: {}/{}".format(epoch + 1, epochs))
                    print("Training")
                elif phase == "validation":
                    model.eval()
                    print("Validating")

                running_loss = 0.0
                running_corrects = 0.0

                for X, label in tqdm(self.loaders[phase]):
                    X, label = (X.to(self.device), label.to(self.device))
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        output = model(X)
                        _, pred = torch.max(output, dim=1)

                        loss = criterion(output, label.long())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * label.size(0)
                    running_corrects += torch.sum(pred == label)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.float() / self.dataset_sizes[phase] * 100
                losses[phase].append(epoch_loss)
                accuracies[phase].append(epoch_acc.to("cpu"))

                print(f"Loss: {epoch_loss}, accuracy: {epoch_acc}")

                if phase == "validation" and epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model.state_dict())

            if scheduler:
                scheduler.step()
            print("\n")

        time_elapsed = time.time() - since
        hours = time_elapsed // 3600
        mins = time_elapsed // 60 - hours * 60
        secs = time_elapsed % 60
        print("Training Time: {}h {}m {}s".format(hours, mins, secs))
        print(f"Best validation accuracy: {best_acc}")

        model.load_state_dict(best_model)
        return model, losses, accuracies

    def evaluate_model(self, model, dataset):
        model.eval()
        predictions, actuals = [], []

        for X, label in tqdm(self.loaders[dataset]):
            X = X.to(self.device)

            with torch.no_grad():
                outputs = model(X)
                _, pred = torch.max(outputs, dim=1)

            pred = pred.to("cpu").numpy()
            label = label.numpy()
            predictions.append(pred.reshape(pred.shape[0], 1))
            actuals.append(label.reshape(label.shape[0], 1))

        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        acc = accuracy_score(actuals, predictions)
        f1 = f1_score(actuals, predictions, average="weighted", zero_division=0)

        print(
            "\nEvaluated on {} set\n accuracy: {}%, f1-score: {}".format(
                dataset, round(acc, 5) * 100, round(f1, 5)
            )
        )
        return actuals, predictions
