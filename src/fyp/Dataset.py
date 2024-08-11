import os, glob
from typing import Literal

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader, random_split


class DroneData(Dataset):
    num_data_points = 9830

    def __init__(
        self,
        data_category: Literal["mel", "rangeDoppler"],
        mode: Literal["detection", "classification"],
    ) -> None:
        """Initialize the dataset for drone detection and classification

        Args:
            data_category: what to load, mel spectrograms or range Doppler maps
            mode: detection or classification

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
            self._datasets = {mode: np.arange(DroneData.num_data_points)}
        elif mode == "classification":
            self._datasets = {mode: []}
            for file_name in os.listdir(self._data_dir):
                temp = file_name.split("_")
                idx, det, cls = int(temp[2]), int(temp[4]), int(temp[6][0])
                if det and cls:
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

    def __getitem__(self, idx) -> tuple[torch.tensor, int, int]:
        self._set_file_name(self._datasets[self._mode][idx])
        
        if self._mode == "detection":
            return self._load_image, self._det_label
        elif self._mode == "classification":
            return self._load_image, self._cls_label


def load_images(
    data_category: Literal["mel", "rangeDoppler"],
    mode: Literal["detection", "classification"],
    batch_size: int = 128,
    lengths: tuple[float] = (0.7, 0.25, 0.05),
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    """load the dataset as splits

    Args:
        data_category: what to load, mel spectrograms or range Doppler maps
        mode: detection or classification
        batch_size: batch_size
        lengths: percentages for the splits, should be summed to 1

    Returns:
        tuple of loaders dictionary and split lengths dictionary

    """
    train, validation, test = random_split(
        DroneData(data_category, mode), lengths=lengths
    )

    train_set = DataLoader(train, batch_size=batch_size, shuffle=True)
    validation_set = DataLoader(validation, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test, batch_size=batch_size, shuffle=True)

    loaders = {"train": train_set, "validation": validation_set, "test": test_set}
    dataset_sizes = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
    }
    return loaders, dataset_sizes


if __name__ == "__main__":
    for img, det, cls in DataLoader(
        DroneData("mel", "classification"), batch_size=256, shuffle=True
    ):
        print(img.shape)
        print(det.shape)
        print(cls.shape)
        break
