from Libs import *

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                nn.Conv1d(in_channels, in_channels//reduction, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv1d(in_channels//reduction, in_channels, kernel_size=1),
                                nn.Sigmoid())

    def forward(self, input):
        return input*self.se(input)
    
class SEResBlock(nn.Module):
    def __init__(self, in_channels, downsample=False, kernel_size=7):
        super(SEResBlock, self).__init__()
        if downsample:
            out_channels = in_channels*2
            self.identity = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size = 1), 
                                          nn.BatchNorm1d(out_channels))
        else:
            out_channels = in_channels
            self.identity = nn.Identity()
            
        conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, padding = "same")
        conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size = kernel_size, padding = "same")
        
        self.convs = nn.Sequential(conv_1, 
                                   nn.BatchNorm1d(out_channels), 
                                   nn.ReLU(), 
                                   nn.Dropout(0.4), 
                                   conv_2, 
                                   nn.BatchNorm1d(out_channels), 
                                   SEModule(out_channels))
        self.act_fn = nn.ReLU()

    def forward(self, input):
        return self.act_fn(self.convs(input) + self.identity(input))
    
class SEResNet(nn.Module):
    def __init__(self, base_channels=64, kernel_size=7):
        super(SEResNet, self).__init__()
        self.seresnet = nn.Sequential(nn.Conv1d(1, base_channels, kernel_size = 15, padding = 7, stride = 2), 
                                      nn.BatchNorm1d(base_channels), 
                                      nn.ReLU(), 
                                      nn.MaxPool1d(kernel_size = 3, padding = 1, stride = 2),
                                      SEResBlock(base_channels, kernel_size = kernel_size), 
                                      SEResBlock(base_channels, kernel_size = kernel_size),
                                      SEResBlock(base_channels, downsample = True, kernel_size = kernel_size), 
                                      SEResBlock(base_channels*2, kernel_size = kernel_size),
                                      # SEResBlock(base_channels*2, downsample = True, kernel_size = kernel_size), 
                                      # SEResBlock(base_channels*4, kernel_size=kernel_size),
                                      # SEResBlock(base_channels*4, downsample = True, kernel_size=kernel_size), 
                                      # SEResBlock(base_channels*8, kernel_size=kernel_size),
                                      nn.AdaptiveAvgPool1d(1))
  
    def forward(self, input):
        return self.seresnet(input)
    
# class Model(nn.Module):
#     "Non-attention"
#     def __init__(self, base_channels = 64, num_classes = 6):
#         super(Model, self).__init__()
#         factor = 4

#         self.backbone_0 = SEResNet(base_channels, kernel_size=7)
#         self.backbone_1 = SEResNet(base_channels, kernel_size=107)
#         self.backbone_2 = SEResNet(base_channels, kernel_size=207)

#         self.classifier = nn.Linear(base_channels*factor, num_classes)

#     def forward(self, input):  
#         return self.classifier(torch.sum(torch.stack([self.backbone_0(input.unsqueeze(1)).squeeze(2), 
#                                                       self.backbone_1(input.unsqueeze(1)).squeeze(2), 
#                                                       self.backbone_2(input.unsqueeze(1)).squeeze(2)
#                                                       ], dim = 1), dim = 1))

class Model(nn.Module):
    "Attention"
    def __init__(self, base_channels = 64, num_classes = 6):
        super(Model, self).__init__()
        n_backbone = 3
        factor = 2

        self.backbone_0 = SEResNet(base_channels, kernel_size=7)
        self.backbone_1 = SEResNet(base_channels, kernel_size=157)
        self.backbone_2 = SEResNet(base_channels, kernel_size=307)

        self.attention = nn.Sequential(nn.Linear(base_channels*factor*n_backbone, base_channels*factor), 
                                          nn.BatchNorm1d(base_channels*factor), 
                                          nn.ReLU(), 
                                          nn.Dropout(0.4), 
                                          nn.Linear(base_channels*factor, n_backbone))

        self.classifier = nn.Linear(base_channels*factor, num_classes)
        
    def forward(self, input):  
        features_0 = self.backbone_0(input.unsqueeze(1)).squeeze(2)
        features_1 = self.backbone_1(input.unsqueeze(1)).squeeze(2)
        features_2 = self.backbone_2(input.unsqueeze(1)).squeeze(2)
    
        attention_scores = torch.sigmoid(self.attention(torch.cat([features_0, features_1, features_2], dim = 1)))
        merged_features = torch.sum(torch.stack([features_0, features_1, features_2], dim = 1)*attention_scores.unsqueeze(-1), dim = 1)
        return self.classifier(merged_features)
    

# class Model(nn.Module):
#     "Transformer"
#     def __init__(self, base_channels=64, num_classes=6):
#         super(Model, self).__init__()
#         factor = 2

#         self.backbone_0 = SEResNet(base_channels, kernel_size=7)
#         self.backbone_1 = SEResNet(base_channels, kernel_size=107)
#         self.backbone_2 = SEResNet(base_channels, kernel_size=207)

#         self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=base_channels*factor, nhead=8, batch_first=True), num_layers=4)
#         self.classifier = nn.Linear(base_channels*factor, num_classes)
        
#     def forward(self, input):  
#         merged_features = self.transformer_encoder(torch.stack([self.backbone_0(input.unsqueeze(1)).squeeze(2), 
#                                                                 self.backbone_1(input.unsqueeze(1)).squeeze(2), 
#                                                                 self.backbone_2(input.unsqueeze(1)).squeeze(2)], dim = 1), is_causal=False)
        
#         return self.classifier(merged_features[:, 0, :])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model().to(device)
    input = torch.randn(50, 8000).to(device)
    
    output = model(input)
    
    print("Input shape:", input.shape)
    print("Output shape:", output.shape)
    print("Total parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))