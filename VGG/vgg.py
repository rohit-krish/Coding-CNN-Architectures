'''
Created on Tue January  31 10:09:36 2023
@author: rohit krishna

Original Paper: https://arxiv.org/abs/1409.1556
'''

import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGGNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1000, architecture=VGG_types['VGG16']) -> None:
        super().__init__()
        self.in_chennels = in_channels
        self.architecture = architecture
        self.conv_layers = self.create_conv_layers()

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(len(x), -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self):
        layers = []
        in_channels = self.in_chennels

        for x in self.architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3,
                        stride=1, padding=1
                    ),
                    nn.BatchNorm2d(x),  # not in the original vgg paper
                    nn.ReLU()
                ]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)


if __name__ == '__main__':
    model = VGGNet()
    x = torch.rand((10, 3, 224, 224))
    print(model(x).shape)  # expected shape (10, 1000)
    print(model)
    # print(summary(model))
