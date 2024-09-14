import os
import copy
import time
from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass

import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score


@dataclass
class Params:
    model_prefix: str
    data_category: Literal["mel", "rangeDoppler", "fusion"]
    mode: Literal["detection", "classification"]
    lengths: tuple[float] = (0.7, 0.25, 0.05)
    epochs: int = 1
    batch_size: int = 1024
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.2
    momentum: float = 0.75
    save_location: os.PathLike = None
    save_weights: bool = False
    dataset_size: int = 9831
    device: str = "cpu"

    @property
    def model_name(self):
        if self.data_category == "mel":
            cat = "audio"
        elif self.data_category == "rangeDoppler":
            cat = "radar"
        else:
            cat = "fusion"

        return f"{self.model_prefix}_{cat}_{self.mode}"

    @property
    def num_classes(self):
        if self.mode == "detection":
            return 2
        elif self.mode == "classification":
            return 5


class Pipeline:
    def __init__(self, loaders, dataset_sizes, params):
        self.loaders = loaders
        self.dataset_sizes = dataset_sizes
        self.params = params

        if params.save_location is not None and not os.path.isdir(params.save_location):
            os.mkdir(params.save_location)

    def train_model(self, model, criterion, optimizer, scheduler=None):
        losses = {"train": [], "validation": []}
        accuracies = {"train": [], "validation": []}

        best_acc = 0.0
        since = time.time()
        model = model.to(self.params.device)
        best_model = copy.deepcopy(model.state_dict())

        for epoch in range(self.params.epochs):
            for phase in ["train", "validation"]:
                if phase == "train":
                    model.train()
                    print("Epoch: {}/{}".format(epoch + 1, self.params.epochs))
                    print("Training")
                elif phase == "validation":
                    model.eval()
                    print("Validating")

                running_loss = 0.0
                running_corrects = 0.0

                for X, label in tqdm(self.loaders[phase]):
                    X, label = (X.to(self.params.device), label.to(self.params.device))
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
        self.model = model
        self.best_acc = float(best_acc)
        return model, losses, accuracies

    def evaluate_model(self, dataset):
        self.model.eval()
        predictions, actuals = [], []

        for X, label in tqdm(self.loaders[dataset]):
            X = X.to(self.params.device)

            with torch.no_grad():
                outputs = self.model(X)
                _, pred = torch.max(outputs, dim=1)

            pred = pred.to("cpu").numpy()
            label = label.numpy()
            predictions.append(pred.reshape(pred.shape[0], 1))
            actuals.append(label.reshape(label.shape[0], 1))

        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        acc = accuracy_score(actuals, predictions)
        f1 = f1_score(actuals, predictions, average="weighted", zero_division=0)

        print(
            "\nEvaluated on {} set\naccuracy: {}%, f1-score: {}\n".format(
                dataset, round(acc, 5) * 100, round(f1, 5)
            )
        )
        self.re_eval_split = dataset
        self.re_eval_acc = acc
        self.re_eval_f1 = f1
        return actuals, predictions

    @property
    def save_results(self):
        path = (
            self.params.save_location + "/" + self.params.model_name + ".txt"
            if self.params.save_location is not None
            else self.params.model_name + ".txt"
        )

        with open(path, "w") as f:
            f.write(
                f"""Model: {self.params.model_prefix}
Data: {"audio" if self.params.data_category == "mel" else "radar"}
Mode: {self.params.mode}\n
Best validation accuracy: {round(self.best_acc, 5)}% \n
Test accuracy: {round(self.re_eval_acc, 5) * 100}%
Test f1-score: {round(self.re_eval_f1, 5)}"""
            )


class FusionPipeline:
    def __init__(self, loaders, dataset_sizes, params):
        self.loaders = loaders
        self.dataset_sizes = dataset_sizes
        self.params = params

        if params.save_location is not None and not os.path.isdir(params.save_location):
            os.mkdir(params.save_location)

    def _get_fusion_dataset(self, radar: torch.tensor, audio: torch.tensor, label: torch.tensor) -> tuple[torch.tensor]:
        """Obtain the individual probabilities from pretrained models.

        Args:
            radar: radar images
            audio: audio images
            label: labels

        Returns:
            two torch tensors of shape (minibatch, 4) respectively for radar probabilitis and audio probabilitis.
        """
        radar_densenet = torch.load(
            f"/home/gevindu/Final_work/Saved models/Compare_{self.params.model_prefix}_radar_{self.params.mode}.pth",
            map_location=self.params.device,
        )
        radar_densenet.eval()

        audio_densenet = torch.load(
            f"/home/gevindu/Final_work/Saved models/Compare_{self.params.model_prefix}_audio_{self.params.mode}.pth",
            map_location=self.params.device,
        )
        audio_densenet.eval()

        sign = torch.nn.Sigmoid()

        with torch.no_grad():
            out_radar = sign(radar_densenet(radar)[:, 1:])
            out_audio = sign(audio_densenet(audio)[:, 1:])

        return out_radar, out_audio, label-1

    def train_model(self, model, criterion, optimizer, scheduler=None):
        losses = {"train": [], "validation": []}
        accuracies = {"train": [], "validation": []}

        best_acc = 0.0
        since = time.time()
        model = model.to(self.params.device)
        best_model = copy.deepcopy(model.state_dict())

        for epoch in range(self.params.epochs):
            for phase in ["train", "validation"]:
                if phase == "train":
                    model.train()
                    print("Epoch: {}/{}".format(epoch + 1, self.params.epochs))
                    print("Training")
                elif phase == "validation":
                    model.eval()
                    print("Validating")

                running_loss = 0.0
                running_corrects = 0.0

                for radar, audio, label in tqdm(self.loaders[phase]):
                    radar, audio, label = (
                        radar.to(self.params.device),
                        audio.to(self.params.device),
                        label.to(self.params.device)
                    )
                    radar, audio, label = self._get_fusion_dataset(radar, audio, label)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        output = model(radar, audio)
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
        self.model = model
        self.best_acc = float(best_acc)
        return model, losses, accuracies

    def evaluate_model(self, dataset):
        self.model.eval()
        predictions, actuals = [], []

        for radar, audio, label in tqdm(self.loaders[dataset]):
            radar, audio = (
                        radar.to(self.params.device),
                        audio.to(self.params.device)
                    )
            radar, audio, label = self._get_fusion_dataset(radar, audio, label)

            with torch.no_grad():
                outputs = self.model(radar, audio)
                _, pred = torch.max(outputs, dim=1)

            pred = pred.to("cpu").numpy()
            label = label.numpy()
            predictions.append(pred.reshape(pred.shape[0], 1))
            actuals.append(label.reshape(label.shape[0], 1))

        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        acc = accuracy_score(actuals, predictions)
        f1 = f1_score(actuals, predictions, average="weighted", zero_division=0)

        print(
            "\nEvaluated on {} set\naccuracy: {}%, f1-score: {}\n".format(
                dataset, round(acc, 5) * 100, round(f1, 5)
            )
        )
        self.re_eval_split = dataset
        self.re_eval_acc = acc
        self.re_eval_f1 = f1
        return actuals, predictions

    @property
    def save_results(self):
        path = (
            self.params.save_location + "/" + self.params.model_name + ".txt"
            if self.params.save_location is not None
            else self.params.model_name + ".txt"
        )

        with open(path, "w") as f:
            f.write(
                f"""Model: {self.params.model_prefix}
Data: {"audio" if self.params.data_category == "mel" else "radar"}
Mode: {self.params.mode}\n
Best validation accuracy: {round(self.best_acc, 5)}% \n
Test accuracy: {round(self.re_eval_acc, 5) * 100}%
Test f1-score: {round(self.re_eval_f1, 5)}"""
            )
