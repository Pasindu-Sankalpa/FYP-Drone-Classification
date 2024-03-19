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
    def __init__(self, in_channels, reduction=16):
        super(SEModule1d, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d((1)),
                                SqueezeLayer(dim=(2,)),
                                nn.Linear(in_channels, in_channels//reduction),
                                nn.ReLU(),
                                nn.Linear(in_channels//reduction, in_channels),
                                nn.Sigmoid(),
                                UnsqueezeLayer(dim=(2,)))

    def forward(self, input):
        return input*self.se(input)

class SEResBlock1d(nn.Module):
    def __init__(self, in_channels, upsample=False, kernel_size=7, dropout=0.4):
        super(SEResBlock1d, self).__init__()
        if upsample:
            out_channels = in_channels*2
            self.identity = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1), 
                                          nn.BatchNorm1d(out_channels))
        else:
            out_channels = in_channels
            self.identity = nn.Identity()
        
        self.convs = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same"), 
                                   nn.BatchNorm1d(out_channels), 
                                   nn.ReLU(), 
                                   nn.Dropout(dropout), 
                                   nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding="same"), 
                                   nn.BatchNorm1d(out_channels), 
                                   SEModule1d(out_channels))
        self.act_fn = nn.ReLU()

    def forward(self, input):
        return self.act_fn(self.convs(input) + self.identity(input))

class SEResNet1d(nn.Module):
    def __init__(self, base_channels=64, kernel_size=7, downsample=True):
        super(SEResNet1d, self).__init__()
        if downsample:
            self.seresnet = nn.Sequential(UnsqueezeLayer(dim=(1,)),
                                          nn.Conv1d(1, base_channels, kernel_size=15, padding=7, stride=2), 
                                          nn.BatchNorm1d(base_channels), 
                                          nn.ReLU(), 
                                          nn.MaxPool1d(kernel_size=3, padding=1, stride=2),
                                          SEResBlock1d(base_channels, kernel_size=kernel_size), 
                                          SEResBlock1d(base_channels, kernel_size=kernel_size),
                                          SEResBlock1d(base_channels, upsample=True, kernel_size=kernel_size), 
                                          SEResBlock1d(base_channels*2, kernel_size=kernel_size),
                                          SEResBlock1d(base_channels*2, kernel_size=kernel_size),
                                          nn.AdaptiveAvgPool1d(1),
                                          SqueezeLayer(dim=2))
        else:
            self.seresnet = nn.Sequential(UnsqueezeLayer(dim=(1,)),
                                          nn.Conv1d(1, base_channels, kernel_size=1), 
                                          SEResBlock1d(base_channels, kernel_size=kernel_size), 
                                          SEResBlock1d(base_channels, kernel_size=kernel_size),
                                          SEResBlock1d(base_channels, upsample=True, kernel_size=kernel_size), 
                                          SEResBlock1d(base_channels*2, kernel_size=kernel_size),
                                          SEResBlock1d(base_channels*2, kernel_size=kernel_size),
                                          nn.AdaptiveAvgPool1d(1),
                                          SqueezeLayer(dim=2))
  
    def forward(self, input):
        return self.seresnet(input)
    

class SEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEModule2d, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                SqueezeLayer(dim=(2, 3)),
                                nn.Linear(in_channels, in_channels//reduction),
                                nn.ReLU(),
                                nn.Linear(in_channels//reduction, in_channels),
                                nn.Sigmoid(),
                                UnsqueezeLayer(dim=(2, 3)))

    def forward(self, input):
        return input*self.se(input)
    
class SEResBlock2d(nn.Module):
    def __init__(self, in_channels, upsample=False, kernel_size=7, dropout=0.4):
        super(SEResBlock2d, self).__init__()
        if upsample:
            out_channels = in_channels*upsample
            self.identity = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), 
                                          nn.BatchNorm2d(out_channels))
        else:
            out_channels = in_channels
            self.identity = nn.Identity()
        
        self.convs = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same"), 
                                   nn.BatchNorm2d(out_channels), 
                                   nn.ReLU(), 
                                   nn.Dropout(dropout), 
                                   nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding="same"), 
                                   nn.BatchNorm2d(out_channels), 
                                   SEModule2d(out_channels))
        self.act_fn = nn.ReLU()

    def forward(self, input):
        return self.act_fn(self.convs(input) + self.identity(input))

class SEResNet2d(nn.Module):
    def __init__(self):
        super(SEResNet2d, self).__init__()
        self.seresnet = nn.Sequential(UnsqueezeLayer(dim=(1,)),
                                      SEResBlock2d(1, upsample=4, kernel_size=3), 
                                      SEResBlock2d(4, upsample=4, kernel_size=3),
                                      SEResBlock2d(16, upsample=4, kernel_size=5), 
                                      SEResBlock2d(64, upsample=2, kernel_size=7),
                                      nn.AdaptiveAvgPool2d(1),
                                      SqueezeLayer(dim=(2, 3)))

    def forward(self, input):
        return self.seresnet(input)

class Model(nn.Module):
    def __init__(self, batch_size=8, base_channels=64, num_classes=2):
        super(Model, self).__init__()
        self.onces = torch.ones((batch_size, 128))
        self.acoustic_encoder_1 = SEResNet1d(base_channels, kernel_size=7, downsample=True)
        self.acoustic_encoder_2 = SEResNet1d(base_channels, kernel_size=107, downsample=True)
        self.rcs_encoder = SEResNet1d(base_channels, kernel_size=7, downsample=False)
        self.doppler_encoder = SEResNet2d()
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, audio, rcs, doppler):  
        acoustic_encoded = torch.sum(torch.stack([self.acoustic_encoder_1(audio), 
                                                  self.acoustic_encoder_2(audio)
                                                  ], dim = 1), dim = 1)
        rcs_encoded = self.rcs_encoder(rcs)
        doppler_encoded = self.doppler_encoder(doppler)

        stacked_features = torch.stack((self.onces, acoustic_encoded, rcs_encoded, doppler_encoded), dim=1)

        combined_features = self.transformer_encoder(stacked_features)

        output = self.classifier(combined_features[:, 0, :])

        # print("Acoustic feature vector shape", acoustic_encoded.shape)
        # print("RCS feature vector shape", rcs_encoded.shape)
        # print("Doppler feature vector shape", doppler_encoded.shape)
        # print("Stacked feature matrix shape", stacked_features.shape)
        # print("Transformer output matrix shape", combined_features.shape)
        # print("Classification output matrix shape", output.shape)

        return output
    

if __name__ == "__main__":
    batch_size = 8
    audio = torch.rand((batch_size, 8000))
    rcs = torch.rand((batch_size, 128))
    doppler = torch.rand((batch_size, 256, 128))
    model = Model()

    out = model(audio, rcs, doppler)