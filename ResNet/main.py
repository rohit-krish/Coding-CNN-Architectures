'''
Created on Fri February  03 23:43:13 2023
@author: rohit krishna

Original Paper: https://arxiv.org/abs/1512.03385
'''

import torch
from torch import nn
from torchinfo import summary


class ResNet(nn.Module):
    def __init__(self, layers, img_channels, n_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            img_channels, 64, kernel_size=7, stride=2,
            padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, n_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(len(x), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, num_residual_blocks, out_channels, stride):
        # identity_down_sample = None
        layers = []

        # if stride != 1 or self.in_channels != (out_channels*4):
        identity_down_sample = nn.Sequential(
            nn.Conv2d(
                self.in_channels, out_channels*4,
                kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(out_channels*4)
        )
        layers.append(
            Block(self.in_channels, out_channels, identity_down_sample, stride)
        )
        self.in_channels = out_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels*self.expansion, kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # print(list(x.shape), list(identity.shape))
        if self.identity_downsample != None:
            identity = self.identity_downsample(identity)
        

        x += identity
        x = self.relu(x)
        return x


def resnet50(img_channels=3, n_classes=1000):
    return ResNet([3, 4, 6, 3], img_channels, n_classes)


def resnet10(img_channels=3, n_classes=1000):
    return ResNet([3, 4, 23, 3], img_channels, n_classes)


def resnet152(img_channels=3, n_classes=1000):
    return ResNet([3, 4, 36, 3], img_channels, n_classes)


if __name__ == '__main__':
    model = resnet50()
    x = torch.randn(2, 3, 28, 28)
    y = model(x)
    # print(model)
    # print(y.shape)
    # print(summary(model))
