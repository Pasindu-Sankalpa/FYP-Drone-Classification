from Libs import *
from Model import DetectionModel, ClassificationModel, CombinedModel
from DataSet import TestDataSet
from statistics import mode

def eval_sep(drone_names):
    det = []
    cls = []
    d_type = []

    s = nn.Softmax(dim=1)

    det_model = torch.load("/home/gevindu/model_final/Saved models/Final_model_det.pth", map_location=device).eval()
    cls_model = torch.load("/home/gevindu/model_final/Saved models/Final_model_cls_v2.pth", map_location=device).eval()

    for doppler, rcs, acoustic, labels in loader:
        doppler, rcs, acoustic = (
            doppler.to(device),
            rcs.to(device),
            acoustic.to(device),
        )
        with torch.no_grad():
            det_max, det_pred = torch.max(det_model(doppler, rcs, acoustic), dim=1)

            output = cls_model(doppler, rcs, acoustic)
            type_prob = float(torch.min(torch.max(s(output), dim=1).values))
            cls_max, cls_pred = torch.max(output, dim=1)

        if torch.mode(det_pred).values: 
            det.append(1)
            cls.append(type_prob)
            d_type.append(int(torch.mode(cls_pred).values))
        else: det.append(0)

    if mode(det): 
        string="Drone detected. "
        if max(cls) < 0.6: string+="Can not identify the drone type."
        else: string+=f"Possible drone type is {drone_names[mode(d_type)]}"
    else: string="No drone detected."

    print(string)

def eval_comb(drone_names):
    model = torch.load("/home/gevindu/model_final/Saved models/Final_model_comb.pth", map_location=device).eval()

    for doppler, rcs, acoustic, labels in loader:
        doppler, rcs, acoustic = (
            doppler.to(device),
            rcs.to(device),
            acoustic.to(device),
        )
        with torch.no_grad():
            det_out, cls_out = model(doppler, rcs, acoustic)

            _, det_pred = torch.max(det_out, dim=1)
            _, cls_pred = torch.max(cls_out, dim=1)

            if not torch.mode(det_pred).values and not torch.mode(cls_pred).values: print("No drone identified.")
            if not torch.mode(det_pred).values and torch.mode(cls_pred).values: print("A drone identified, can not classified to an known type.")
            if torch.mode(det_pred).values: print(f"A drone identified, possible drone type is {drone_names[torch.mode(cls_pred).values.item()]}")
        break

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(TestDataSet(""), batch_size=16, shuffle=True)

    drone_names = {1: "DJI Matrice 300 RTK",
                    2: "DJI Phanthom-4 Pro Plus",
                    3: "Mavic 2 Enterprise Dual"}

    # eval_sep(drone_names)

    eval_comb(drone_names)