from fyp import FusionDroneData, DroneData
from fyp.TrainEval import Params

def test_init():
    FusionDroneData("detection", 0, Params.dataset_size)
    FusionDroneData("classification", 0, Params.dataset_size)

