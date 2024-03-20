from Libs import *


class Train_n_evaluate:
    def __init__(self, loaders, dataset_sizes, device):
        self.loaders = loaders
        self.dataset_sizes = dataset_sizes
        self.device = device

    def train_model(self, model, criterion, optimizer, epochs, scheduler):
        losses = {"train": [], "validation": []}
        accuracies = {"train": [], "validation": []}
        since = time.time()
        best_model = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(epochs):
            for phase in ["train", "validation"]:
                if phase == "train":
                    model.train()
                    print("Epoch: {}/{}".format(epoch + 1, epochs))
                elif phase == "validation":
                    model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                for doppler, rcs, acoustic, labels in tqdm(self.loaders[phase]):
                    doppler, rcs, acoustic, labels = doppler.to(self.device), rcs.to(self.device), acoustic.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(doppler, rcs, acoustic)
                        _, pred = torch.max(outputs, dim=1)
                        loss = criterion(outputs, labels.long())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(pred == labels)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase] * 100
                losses[phase].append(epoch_loss)
                accuracies[phase].append(epoch_acc.to("cpu"))

                print(
                    "{} - loss: {}, accuracy: {}".format(phase, epoch_loss, epoch_acc)
                )

                if phase == "validation" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model.state_dict())

            if scheduler:
                scheduler.step()
            print("\n")

        time_elapsed = time.time() - since
        print("Training Time {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best accuracy {}".format(best_acc))

        model.load_state_dict(best_model)
        return model, losses, accuracies

    def evaluate_model(self, model, dataset):
        global device
        model.eval()
        predictions, actuals = [], []

        for inputs, labels in self.loaders[dataset]:
            with torch.no_grad():
                outputs = model(inputs.float().to(self.device))
                _, preds = torch.max(outputs, dim=1)

            preds = preds.to("cpu").numpy()
            labels = labels.numpy()
            predictions.append(preds.reshape(preds.shape[0], 1))
            actuals.append(labels.reshape(labels.shape[0], 1))

        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        acc = accuracy_score(actuals, predictions)
        f1 = f1_score(actuals, predictions, average="weighted", zero_division=0)

        print(
            "\nRe-evaluated on {} set\nAccuracy: {}%, f1-score: {}".format(
                dataset, round(acc, 5) * 100, round(f1, 5)
            )
        )
        return actuals, predictions
