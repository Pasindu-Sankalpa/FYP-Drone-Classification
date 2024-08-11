from torch.utils.data import DataLoader
from fyp.Dataset import DroneData, load_images

def test_Dataset_mel_cls():
    for img, det, cls in DataLoader(
        DroneData("mel", "classification"), batch_size=256, shuffle=True
    ):
        size = img.shape[0]

        assert img.shape == (size, 3, 224, 224)
        assert sum(det) == size
        assert cls.shape[0] == size

def test_Dataset_rD_cls():
    for img, det, cls in DataLoader(
        DroneData("rangeDoppler", "classification"), batch_size=256, shuffle=True
    ):
        size = img.shape[0]

        assert img.shape == (size, 3, 224, 224)
        assert sum(det) == size
        assert cls.shape[0] == size

def test_Dataset_mel_det():
    for img, det, cls in DataLoader(
        DroneData("mel", "detection"), batch_size=256, shuffle=True
    ):
        size = img.shape[0]

        assert img.shape == (size, 3, 224, 224)
        assert det.shape[0] == size
        assert cls.shape[0] == size

def test_Dataset_rD_det():
    for img, det, cls in DataLoader(
        DroneData("rangeDoppler", "detection"), batch_size=256, shuffle=True
    ):
        size = img.shape[0]

        assert img.shape == (size, 3, 224, 224)
        assert det.shape[0] == size
        assert cls.shape[0] == size

def test_load_images():
    def support_loaders(loaders):
        assert set(["train", "test", "validation"]) == {"test", "train", "validation"}
        for key in loaders.keys():
            assert isinstance(loaders[key], DataLoader)

    loaders, sizes = load_images("mel", "detection")
    assert sum([sizes[key] for key in sizes.keys()]) == 9830
    support_loaders(loaders)


    loaders, sizes = load_images("mel", "classification")
    assert sum([sizes[key] for key in sizes.keys()]) == 4904
    support_loaders(loaders)