from Libs import *


class SqueezeLayer(nn.Module):
    def __init__(self, dim=1):
        super(SqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)


class UnsqueezeLayer(nn.Module):
    def __init__(self, dim=(1,)):
        super(UnsqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        for d in self.dim:
            x = torch.unsqueeze(x, dim=d)

        return x


class SEModule1d(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule1d, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d((1)),
            SqueezeLayer(dim=(2,)),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
            UnsqueezeLayer(dim=(2,)),
        )

    def forward(self, input):
        return input * self.se(input)


class SEResBlock1d(nn.Module):
    def __init__(self, in_channels, upsample=False, kernel_size=7, dropout=0.4):
        super(SEResBlock1d, self).__init__()
        if upsample:
            out_channels = in_channels * 2
            self.identity = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            out_channels = in_channels
            self.identity = nn.Identity()

        self.convs = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding="same"
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels, out_channels, kernel_size=kernel_size, padding="same"
            ),
            nn.BatchNorm1d(out_channels),
            SEModule1d(out_channels),
        )
        self.act_fn = nn.ReLU()

    def forward(self, input):
        return self.act_fn(self.convs(input) + self.identity(input))


class SEResNet1d(nn.Module):
    def __init__(self, base_channels=64, kernel_size=7, downsample=True):
        super(SEResNet1d, self).__init__()
        if downsample:
            self.seresnet = nn.Sequential(
                UnsqueezeLayer(dim=(1,)),
                nn.Conv1d(1, base_channels, kernel_size=15, padding=7, stride=2),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, padding=1, stride=2),
                SEResBlock1d(base_channels, kernel_size=kernel_size),
                SEResBlock1d(base_channels, kernel_size=kernel_size),
                SEResBlock1d(base_channels, upsample=True, kernel_size=kernel_size),
                SEResBlock1d(base_channels * 2, kernel_size=kernel_size),
                SEResBlock1d(base_channels * 2, kernel_size=kernel_size),
                nn.AdaptiveAvgPool1d(1),
                SqueezeLayer(dim=2),
            )
        else:
            self.seresnet = nn.Sequential(
                UnsqueezeLayer(dim=(1,)),
                nn.Conv1d(1, base_channels, kernel_size=1),
                SEResBlock1d(base_channels, kernel_size=kernel_size),
                SEResBlock1d(base_channels, kernel_size=kernel_size),
                SEResBlock1d(base_channels, upsample=True, kernel_size=kernel_size),
                SEResBlock1d(base_channels * 2, kernel_size=kernel_size),
                SEResBlock1d(base_channels * 2, kernel_size=kernel_size),
                nn.AdaptiveAvgPool1d(1),
                SqueezeLayer(dim=2),
            )

    def forward(self, input):
        return self.seresnet(input)


class SEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule2d, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            SqueezeLayer(dim=(2, 3)),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
            UnsqueezeLayer(dim=(2, 3)),
        )

    def forward(self, input):
        return input * self.se(input)


class SEResBlock2d(nn.Module):
    def __init__(self, in_channels, upsample=False, kernel_size=7, dropout=0.4):
        super(SEResBlock2d, self).__init__()
        if upsample:
            out_channels = in_channels * upsample
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            out_channels = in_channels
            self.identity = nn.Identity()

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding="same"
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding="same"
            ),
            nn.BatchNorm2d(out_channels),
            SEModule2d(out_channels),
        )
        self.act_fn = nn.ReLU()

    def forward(self, input):
        return self.act_fn(self.convs(input) + self.identity(input))


class SEResNet2d(nn.Module):
    def __init__(self):
        super(SEResNet2d, self).__init__()
        self.seresnet = nn.Sequential(
            UnsqueezeLayer(dim=(1,)),
            SEResBlock2d(1, upsample=4, kernel_size=3),
            SEResBlock2d(4, upsample=4, kernel_size=3),
            SEResBlock2d(16, upsample=4, kernel_size=5),
            SEResBlock2d(64, upsample=2, kernel_size=7),
            nn.AdaptiveAvgPool2d((1, 1)),
            SqueezeLayer(dim=(2, 3)),
        )

    def forward(self, input):
        return self.seresnet(input)


class Model(nn.Module):
    def __init__(
        self,
        base_channels=64,
        num_classes=2,
    ):
        super(Model, self).__init__()
        self.acoustic_encoder_1 = SEResNet1d(
            base_channels, kernel_size=7, downsample=True
        )
        self.acoustic_encoder_2 = SEResNet1d(
            base_channels, kernel_size=107, downsample=True
        )
        self.rcs_encoder = SEResNet1d(base_channels, kernel_size=7, downsample=False)
        self.doppler_encoder = SEResNet2d()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, doppler, rcs, audio):
        acoustic_encoded = torch.sum(
            torch.stack(
                [self.acoustic_encoder_1(audio), self.acoustic_encoder_2(audio)], dim=1
            ),
            dim=1,
        )
        rcs_encoded = self.rcs_encoder(rcs)
        doppler_encoded = self.doppler_encoder(doppler)
        return self.classifier(
            self.transformer_encoder(
                torch.stack((acoustic_encoded, rcs_encoded, doppler_encoded), dim=1)
            )[:, 0, :]
        )


if __name__ == "__main__":
    from DataSet import DataSet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "\n")

    model = Model(device).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # dataset = DataSet()
    # train_set = DataLoader(dataset, batch_size=8, shuffle=True)

    # for X1, X2, X3, y in train_set:
    #     print(X1.shape, X1.dtype)
    #     print(X2.shape, X2.dtype)
    #     print(X3.shape, X3.dtype)
    #     print(y.shape, y.dtype)
    #     print("")
    #     out = model(X1.to(device), X2.to(device), X3.to(device))
    #     break
