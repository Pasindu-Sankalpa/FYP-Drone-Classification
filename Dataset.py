import os, glob
from typing import Literal

from PIL import Image

import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader


class DroneData(Dataset):
    num_data_points = 9830

    def __init__(self, data_category: Literal["mel", "rangeDoppler"]) -> None:
        """Initialize the dataset for drone detection and classification

        Args:
            data_category: what to load, mel spectrograms or range Doppler maps
        
        """
        self._data_category = data_category
        self._data_dir = (
            f"/home/gevindu/model_final/Airforce Data processed/{data_category}"
        )
        self._transform = transform.Compose([transform.Resize((224, 224)),
                                             transform.ToTensor(),
                                             transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                             ])

    def __len__(self) -> int:
        """Return the number of data points."""
        return DroneData.num_data_points

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
        return self._transform(Image.open(self._file_name).convert('RGB'))

    def __getitem__(self, idx) -> tuple[torch.tensor, int, int]:
        self._set_file_name(idx)
        return self._load_image, self._det_label, self._cls_label
        


if __name__ == "__main__":

    for img, det, cls in DataLoader(DroneData("rangeDoppler"), batch_size=256, shuffle=True):
        print(img.shape)
        print(det.shape)
        print(cls.shape)
        break