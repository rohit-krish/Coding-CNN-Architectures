'''
Created on Fri February  03 10:50:32 2023
@author: rohit krishna
'''

import torch
from torch import nn


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)

        self.incpetion3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.incpetion3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.incpetion4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.incpetion4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.incpetion4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.incpetion4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.incpetion4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.incpetion5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.incpetion5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.incpetion3a(x)
        x = self.incpetion3b(x)
        x = self.maxpool(x)

        x = self.incpetion4a(x)
        
        # auxiliary softmax classifer 1
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.incpetion4b(x)
        x = self.incpetion4c(x)
        x = self.incpetion4d(x)

        # auxiliary softmax classifer 2
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.incpetion4e(x)
        x = self.maxpool(x)

        x = self.incpetion5a(x)
        x = self.incpetion5b(x)
        x = self.avgpool(x)

        x = x.reshape(len(x), -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super().__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)
        ], 1)


class InceptionAux(nn.Module):  # inception auxilary classifier
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.7)
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = x.reshape(len(x), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = GoogLeNet()
    preds = model(x)
    assert (
        preds[2].shape == (3, 1000) and print('All good :)') == None
    ), 'Shapes don\'t match :('

    print([pred.shape for pred in preds])
