import torch.nn as nn
# from torchsummary import summary


class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        self.conv_s = nn.Sequential(
            self._conv_bn(ch_in, 32, 2),
            self._dwsc(32, 64, 1),
            self._dwsc(64, 128, 2),
            self._dwsc(128, 128, 1),
            self._dwsc(128, 256, 2),
            self._dwsc(256, 256, 1),
            self._dwsc(256, 512, 2),
            self._dwsc(512, 512, 1),
            self._dwsc(512, 512, 1),
            self._dwsc(512, 512, 1),
            self._dwsc(512, 512, 1),
            self._dwsc(512, 512, 1),
            self._dwsc(512, 1024, 2),
            self._dwsc(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, n_classes)

    def _conv_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def _dwsc(self, inp, oup, stride):
        '''
        Depthwise-Seperable-Convolution

        why seperable, ie becuase, insted of just one convolution we have dw and pw convolutions
        '''

        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_s(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = MobileNetV1(ch_in=3, n_classes=1000)
    # summary(model, input_size=(3, 224, 224), device='cpu')
