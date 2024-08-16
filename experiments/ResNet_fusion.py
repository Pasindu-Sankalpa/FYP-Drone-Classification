from fyp.Dataset import load_images
from fyp.TrainEval import Params

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

params = Params(
    model_prefix="ResNet",
    save_location="./ResNet",
    data_category="fusion",
    mode="detection",
    batch_size=1024,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

loaders, sizes = load_images(params, True)

radar_densenet = torch.load(
    f"/home/gevindu/Final_work/Saved models/Compare_{params.model_prefix}_radar_detection.pth",
    map_location=params.device,
)
radar_densenet.eval()
audio_densenet = torch.load(
    f"/home/gevindu/Final_work/Saved models/Compare_{params.model_prefix}_audio_detection.pth",
    map_location=params.device,
)
audio_densenet.eval()
softmax = torch.nn.Softmax(dim=1)


def get_fusion_dataset(split: str) -> np.array:
    """Obtain the individual probabilities and stack them to a new dataset.

    Args:
        split: split to use from the loaders

    Returns:
        numpy array of shape (dataset_split_size, 3)
        order is: radar probability, audio probability, label
    """

    fusion_dataset = torch.zeros((sizes[split], 3))

    for minibatch_idx, (radar, audio, label) in enumerate(loaders[split]):
        radar, audio, label = (
            radar.to(params.device),
            audio.to(params.device),
            label.to(params.device),
        )
        with torch.no_grad():
            out_radar = softmax(radar_densenet(radar))[:, 1]
            out_audio = softmax(audio_densenet(audio))[:, 1]

        if label.shape[0] == params.batch_size:
            fusion_dataset[
                minibatch_idx * params.batch_size : (1 + minibatch_idx)
                * params.batch_size
            ] = torch.stack((out_radar, out_audio, label), dim=1)
        else:
            fusion_dataset[minibatch_idx * params.batch_size :] = torch.stack(
                (out_radar, out_audio, label), dim=1
            )

    return fusion_dataset.detach().numpy()


train_split = get_fusion_dataset("train")
test_split = get_fusion_dataset("test")

LogisticRegr = LogisticRegression()
LogisticRegr.fit(train_split[:, :1], train_split[:, 2].astype(np.int32))
pred = LogisticRegr.predict(test_split[:, :1])


acc = accuracy_score(test_split[:, 2].astype(np.int32), pred.astype(np.int32))
f1 = f1_score(
    test_split[:, 2].astype(np.int32),
    pred.astype(np.int32),
    average="weighted",
    zero_division=0,
)

results_str = f"""Model: {params.model_prefix}
Data: {params.data_category}
Mode: {params.mode}\n
Test accuracy: {round(acc, 5) * 100}%
Test f1-score: {round(f1, 5)}"""

path = (
    params.save_location + "/" + params.model_name + ".txt"
    if params.save_location is not None
    else params.model_name + ".txt"
)

with open(path, "w") as f:
    f.write(results_str)

print(results_str)
