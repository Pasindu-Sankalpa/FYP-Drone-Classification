from torch.utils.data import DataLoader
from fyp.Dataset import DroneData, load_images
from fyp.TrainEval import Params


def test_Dataset_mel_cls():
    dataset = DroneData("mel", "classification", 1000, 2000)
    for img, cls in DataLoader(dataset, batch_size=256, shuffle=True):
        size = img.shape[0]

        assert img.shape == (size, 3, 224, 224)
        assert cls.shape[0] == size

    for idx in dataset._datasets["classification"]:
        assert idx in range(1000, 2000)


def test_Dataset_rD_cls():
    dataset = DroneData("rangeDoppler", "classification", 2000, 5000)
    for img, cls in DataLoader(dataset, batch_size=256, shuffle=True):
        size = img.shape[0]

        assert img.shape == (size, 3, 224, 224)
        assert cls.shape[0] == size

    for idx in dataset._datasets["classification"]:
        assert idx in range(2000, 5000)


def test_Dataset_mel_det():
    dataset = DroneData("mel", "detection", 5000, 7500)
    for img, det in DataLoader(dataset, batch_size=256, shuffle=True):
        size = img.shape[0]

        assert img.shape == (size, 3, 224, 224)
        assert det.shape[0] == size

    for idx in dataset._datasets["detection"]:
        assert idx in range(5000, 7500)


def test_Dataset_rD_det():
    dataset = DroneData("rangeDoppler", "detection", 0, 500)
    for img, det in DataLoader(dataset, batch_size=256, shuffle=True):
        size = img.shape[0]

        assert img.shape == (size, 3, 224, 224)
        assert det.shape[0] == size

    for idx in dataset._datasets["detection"]:
        assert idx in range(0, 500)


def test_reproducibility():
    dataset1 = DroneData("mel", "classification", 1000, 3500)
    dataset2 = DroneData("mel", "classification", 1000, 3500)

    assert (
        dataset1._datasets["classification"] == dataset2._datasets["classification"]
    ).all()

    dataset1 = DroneData("rangeDoppler", "detection", 1000, 3500)
    dataset2 = DroneData("rangeDoppler", "detection", 1000, 3500)

    assert (dataset1._datasets["detection"] == dataset2._datasets["detection"]).all()


def test_load_images():
    def support_loaders(loaders):
        assert set(loaders.keys()) == {"test", "train", "validation"}
        for key in loaders.keys():
            assert isinstance(loaders[key], DataLoader)

    params = Params(model_prefix="", data_category="mel", mode="detection")

    loaders, sizes = load_images(params)
    assert sum([sizes[key] for key in sizes.keys()]) == 9831
    support_loaders(loaders)

    params = Params(model_prefix="", data_category="mel", mode="classification")

    loaders, sizes = load_images(params)
    assert sum([sizes[key] for key in sizes.keys()]) == 4904
    support_loaders(loaders)
