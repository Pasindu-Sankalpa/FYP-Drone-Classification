import os, glob
from typing import Literal

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader, random_split

from .TrainEval import Params


class DroneData(Dataset):
    def __init__(
        self,
        data_category: Literal["mel", "rangeDoppler"],
        mode: Literal["detection", "classification"],
        lower_lim: int,
        upper_lim: int,
    ) -> None:
        """Initialize the dataset for drone detection and classification

        Args:
            data_category: what to load, mel spectrograms or range Doppler maps
            mode: detection or classification
            lower_lim: lower limit of indexes to load from disk
            upper_lim: upper limit of indexes to load from disk

        """
        self._data_category = data_category
        self._mode = mode
        self._data_dir = (
            f"/home/gevindu/Final_work/Airforce Data processed/{data_category}"
        )
        self._transform = transform.Compose(
            [
                transform.Resize((224, 224)),
                transform.ToTensor(),
                transform.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        if mode == "detection":
            self._datasets = {mode: range(lower_lim, upper_lim)}
        elif mode == "classification":
            self._datasets = {mode: []}
            for file_name in sorted(os.listdir(self._data_dir)):
                temp = file_name.split("_")
                idx, det, cls = int(temp[2]), int(temp[4]), int(temp[6][0])
                if (det and cls) and idx in range(lower_lim, upper_lim):
                    self._datasets[mode].append(idx)
        self._datasets[mode] = np.array(self._datasets[mode])

    def __len__(self) -> int:
        """Return the number of data points."""
        return self._datasets[self._mode].shape[0]

    def _set_file_name(self, idx: int) -> None:
        """set the file name for given index"""
        file_list = glob.glob(
            os.path.join(self._data_dir, f"{self._data_category}_data_{idx}_*.png")
        )
        if len(file_list) > 1:
            raise NameError("More than 1 matching names")

        self._file_name = file_list[0]

    @property
    def _det_label(self) -> int:
        """Returns detection label given the file name"""
        return int(self._file_name.split("_")[5])

    @property
    def _cls_label(self) -> int:
        """Returns classification label given the file name"""
        return int(self._file_name.split("_")[7][:-4])

    @property
    def _load_image(self) -> torch.tensor:
        """Load and transform the image"""
        return self._transform(Image.open(self._file_name).convert("RGB"))

    def return_item_for_fusion(self, idx) -> tuple[torch.tensor, int]:
        "same as __getiten__, return the values to user"
        self._set_file_name(self._datasets[self._mode][idx])

        if self._mode == "detection":
            return self._load_image, self._det_label
        elif self._mode == "classification":
            return self._load_image, self._cls_label

    def __getitem__(self, idx) -> tuple[torch.tensor, int]:
        self._set_file_name(self._datasets[self._mode][idx])

        if self._mode == "detection":
            return self._load_image, self._det_label
        elif self._mode == "classification":
            return self._load_image, self._cls_label


class FusionDroneData(Dataset):
    def __init__(
        self,
        mode: Literal["detection", "classification"],
        lower_lim: int,
        upper_lim: int,
    ) -> None:
        """Initialize the fusion dataset for drone detection and classification

        Args:
            mode: detection or classification
            lower_lim: lower limit of indexes to load from disk
            upper_lim: upper limit of indexes to load from disk
        """

        self._mode = mode
        self._radar_dataset = DroneData("rangeDoppler", mode, lower_lim, upper_lim)
        self._audio_dataset = DroneData("mel", mode, lower_lim, upper_lim)

        assert (
            self._radar_dataset._datasets[self._mode].shape[0]
            == self._audio_dataset._datasets[self._mode].shape[0]
        )
        assert (
            self._radar_dataset._datasets[mode] == self._audio_dataset._datasets[mode]
        ).all()

    def __len__(self) -> int:
        """Return the number of data points."""

        return self._radar_dataset._datasets[self._mode].shape[0]

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor, int]:
        img_radar, y_radar = self._radar_dataset.return_item_for_fusion(idx)
        img_audio, y_audio = self._audio_dataset.return_item_for_fusion(idx)

        assert y_radar == y_audio
        return img_radar, img_audio, y_radar


def load_images(
    params: Params, fuse: bool = False
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    """load the dataset as splits

    Args:
        params: Parameter data class object
        fuse: whether the datasets are fusion of audio and radar. defaults to false.

    Returns:
        tuple of loaders dictionary and split lengths dictionary

    """

    def _get_indexes(lengths, dataset_size):
        if not isinstance(lengths, np.ndarray):
            lengths = np.array(lengths)

        indexes = np.zeros(lengths.shape[0] + 1)
        indexes[1:] = np.cumsum(lengths)
        return (indexes * dataset_size).astype(np.int32)

    indexs = _get_indexes(params.lengths, params.dataset_size)

    if params.data_category == "fusion":
        train = FusionDroneData(params.mode, indexs[0], indexs[1])
        validation = FusionDroneData(params.mode, indexs[1], indexs[2])
        test = FusionDroneData(params.mode, indexs[2], indexs[3])
    else:
        train = DroneData(params.data_category, params.mode, indexs[0], indexs[1])
        validation = DroneData(params.data_category, params.mode, indexs[1], indexs[2])
        test = DroneData(params.data_category, params.mode, indexs[2], indexs[3])

    train_set = DataLoader(train, batch_size=params.batch_size, shuffle=True)
    validation_set = DataLoader(validation, batch_size=params.batch_size, shuffle=True)
    test_set = DataLoader(test, batch_size=params.batch_size, shuffle=True)

    loaders = {"train": train_set, "validation": validation_set, "test": test_set}
    dataset_sizes = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
    }
    return loaders, dataset_sizes
