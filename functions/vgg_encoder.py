import torch
import torch.nn as nn
from torchvision.models import vgg19

class relu_1_2(nn.Module):                  # First Block
    def __init__(self):
        super(relu_1_2, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:4])

    def forward(self, x):
        return self.feature_extractor(x)

class relu_2_2(nn.Module):                 # Second Block
    def __init__(self):
        super(relu_2_2, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:9])

    def forward(self, x):
        return self.feature_extractor(x)

class relu_3_4(nn.Module):                  # Third Block
    def __init__(self):
        super(relu_3_4, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, x):
        return self.feature_extractor(x)

class relu_4_4(nn.Module):                  # Third Block
    def __init__(self):
        super(relu_4_4, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:27])

    def forward(self, x):
        return self.feature_extractor(x)

class relu_5_4(nn.Module):                  # Third Block
    def __init__(self):
        super(relu_5_4, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:36])

    def forward(self, x):
        return self.feature_extractor(x)