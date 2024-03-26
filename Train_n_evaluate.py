from Libs import *


class Train_n_evaluate_detection:
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
                    doppler, rcs, acoustic, labels = (
                        doppler.to(self.device),
                        rcs.to(self.device),
                        acoustic.to(self.device),
                        labels.to(self.device),
                    )
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
                epoch_acc = running_corrects.float() / self.dataset_sizes[phase] * 100
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
        hours = time_elapsed // 3600
        mins = time_elapsed // 60 - hours * 60
        secs = time_elapsed % 60
        print("Training Time {}h {}m {}s".format(hours, mins, secs))
        print("Best validation accuracy {}".format(best_acc))

        model.load_state_dict(best_model)
        return model, losses, accuracies

    def evaluate_model(self, model, dataset):
        model.eval()
        predictions, actuals = [], []

        for doppler, rcs, acoustic, labels in tqdm(self.loaders[dataset]):
            doppler, rcs, acoustic = (
                doppler.to(self.device),
                rcs.to(self.device),
                acoustic.to(self.device),
            )
            with torch.no_grad():
                outputs = model(doppler, rcs, acoustic)
                _, preds = torch.max(outputs, dim=1)

            preds = preds.to("cpu").numpy()
            labels = labels.numpy()
            predictions.append(preds.reshape(preds.shape[0], 1))
            actuals.append(labels.reshape(labels.shape[0], 1))

        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        acc = accuracy_score(actuals, predictions)
        f1 = f1_score(actuals, predictions, average="weighted", zero_division=0)

        print(
            "\nEvaluated on {} set\nAccuracy: {}%, f1-score: {}".format(
                dataset, round(acc, 5) * 100, round(f1, 5)
            )
        )
        return actuals, predictions
    
class Train_n_evaluate_classification:
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
                    doppler, rcs, acoustic, labels = (
                        doppler.to(self.device),
                        rcs.to(self.device),
                        acoustic.to(self.device),
                        labels.to(self.device),
                    )
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
                epoch_acc = running_corrects.float() / self.dataset_sizes[phase] * 100
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
        hours = time_elapsed // 3600
        mins = time_elapsed // 60 - hours * 60
        secs = time_elapsed % 60
        print("Training Time {}h {}m {}s".format(hours, mins, secs))
        print("Best validation accuracy {}".format(best_acc))

        model.load_state_dict(best_model)
        return model, losses, accuracies

    def evaluate_model(self, model, dataset):
        model.eval()
        predictions, actuals = [], []

        for doppler, rcs, acoustic, labels in tqdm(self.loaders[dataset]):
            doppler, rcs, acoustic = (
                doppler.to(self.device),
                rcs.to(self.device),
                acoustic.to(self.device),
            )
            with torch.no_grad():
                outputs = model(doppler, rcs, acoustic)
                _, preds = torch.max(outputs, dim=1)

            preds = preds.to("cpu").numpy()
            labels = labels.numpy()
            predictions.append(preds.reshape(preds.shape[0], 1))
            actuals.append(labels.reshape(labels.shape[0], 1))

        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        acc = accuracy_score(actuals, predictions)
        f1 = f1_score(actuals, predictions, average="weighted", zero_division=0)

        print(
            "\nEvaluated on {} set\nAccuracy: {}%, f1-score: {}".format(
                dataset, round(acc, 5) * 100, round(f1, 5)
            )
        )
        return actuals, predictions

class Train_n_evaluate_combined:
    def __init__(self, loaders, dataset_sizes, device):
        self.loaders = loaders
        self.dataset_sizes = dataset_sizes
        self.device = device

    def train_model(
        self, model, criterion, optimizer, epochs, scheduler, alpha=1.5
    ):
        det_losses = {"train": [], "validation": []}
        det_accuracies = {"train": [], "validation": []}
        cls_losses = {"train": [], "validation": []}
        cls_accuracies = {"train": [], "validation": []}
        losses = {"train": [], "validation": []}

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

                running_det_loss = 0.0
                running_det_corrects = 0.0
                running_cls_loss = 0.0
                running_cls_corrects = 0.0
                running_loss = 0.0

                for doppler, rcs, acoustic, det_label, cls_label in tqdm(
                    self.loaders[phase]
                ):
                    doppler, rcs, acoustic, det_label, cls_label = (
                        doppler.to(self.device),
                        rcs.to(self.device),
                        acoustic.to(self.device),
                        det_label.to(self.device),
                        cls_label.to(self.device),
                    )
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(doppler, rcs, acoustic)
                        _, det = torch.max(outputs[0], dim=1)
                        _, cls = torch.max(outputs[1], dim=1)
                        det_loss = criterion(outputs[0], det_label.long())
                        cls_loss = criterion(outputs[1], cls_label.long())

                        loss = torch.mean(det_loss + alpha*det_label*cls_loss)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_det_loss += torch.mean(det_loss).item() * det_label.size(0)
                    running_det_corrects += torch.sum(det == det_label)

                    running_cls_loss += torch.mean(cls_loss).item() * cls_label.size(0)
                    running_cls_corrects += torch.sum(cls == cls_label)

                    running_loss += loss.item() * det_label.size(0)

                epoch_det_loss = running_det_loss / self.dataset_sizes[phase]
                epoch_det_acc = (
                    running_det_corrects.float() / self.dataset_sizes[phase] * 100
                )
                det_losses[phase].append(epoch_det_loss)
                det_accuracies[phase].append(epoch_det_acc.to("cpu"))

                epoch_cls_loss = running_cls_loss / self.dataset_sizes[phase]
                epoch_cls_acc = (
                    running_cls_corrects.float() / self.dataset_sizes[phase] * 100
                )
                cls_losses[phase].append(epoch_cls_loss)
                cls_accuracies[phase].append(epoch_cls_acc.to("cpu"))

                epoch_loss = running_loss / self.dataset_sizes[phase]
                losses[phase].append(epoch_loss)

                print(
                    "{} - multi task loss: {}\ndetection loss: {}, accuracy: {}\nclassification loss: {}, accuracy: {}".format(
                        phase,
                        epoch_loss,
                        epoch_det_loss,
                        epoch_det_acc,
                        epoch_cls_loss,
                        epoch_cls_acc,
                    )
                )

                if phase == "validation" and epoch_cls_acc > best_acc:
                    best_acc = epoch_cls_acc
                    best_model = copy.deepcopy(model.state_dict())

            if scheduler:
                scheduler.step()
            print("\n")

        time_elapsed = time.time() - since
        hours = time_elapsed // 3600
        mins = time_elapsed // 60 - hours * 60
        secs = time_elapsed % 60
        print("Training Time {}h {}m {}s".format(hours, mins, secs))
        print("Best validation accuracy {}".format(best_acc))

        model.load_state_dict(best_model)
        return model, losses, det_losses, cls_losses, det_accuracies, cls_accuracies


    def evaluate_model(self, model, dataset, mode):
        """
        mode = 0, detection
        mode = 1, classification
        """
        mode_name = "Detection" if not mode else "Classification" 
        model.eval()
        predictions, actuals = [], []

        for doppler, rcs, acoustic, det_label, cls_label in tqdm(self.loaders[dataset]):
            if mode: label=cls_label 
            else: label=det_label

            doppler, rcs, acoustic = (
                doppler.to(self.device),
                rcs.to(self.device),
                acoustic.to(self.device),
            )
            with torch.no_grad():
                outputs = model(doppler, rcs, acoustic)
                _, pred = torch.max(outputs[mode], dim=1)
                
            pred = pred.to("cpu").numpy()
            label = label.numpy()
            predictions.append(pred.reshape(pred.shape[0], 1))
            actuals.append(label.reshape(label.shape[0], 1))

        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        acc = accuracy_score(actuals, predictions)
        f1 = f1_score(actuals, predictions, average="weighted", zero_division=0)

        print(
            "\nEvaluated on {} set\n{} accuracy: {}%, f1-score: {}".format(
                dataset, mode_name, round(acc, 5) * 100, round(f1, 5)
            )
        )
        return actuals, predictions