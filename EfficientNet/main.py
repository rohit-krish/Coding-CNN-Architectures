'''
Created on Sat February  04 10:14:56 2023
@author: rohit krishna

Original Paper: https://arxiv.org/abs/1905.11946

Tricks: Inverted Residual block, Sqeeze and exitation, Stoichastic depth
'''
import torch
from torch import nn
from math import ceil
from torchinfo import summary

base_model = [
    # expand_ratio, channels, repeats/layers, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

# under constraint of α · β 2 · γ 2 ≈ 2.
# α = 1.2;
# β = 1.1;
# γ = 1.15,

phi_values = {
    # (phi_value, resolution, drop_rate)
    'b0': (0, 224, 0.2),
    'b1': (0.5, 240, 0.2),
    'b2': (1, 260, 0.3),
    'b3': (2, 300, 0.3),
    'b4': (3, 380, 0.4),
    'b5': (4, 456, 0.4),
    'b6': (5, 528, 0.5),
    'b7': (6, 600, 0.5)
}


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride, padding, groups=1
    ):
        super().__init__()
        '''
        if set group1 as we did by default then this is a normal conv op
        if we set it to groups=in_channels, then this is a depth wise conv op
        '''
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C X H X W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        self.se(x) returns a value for each channels, in x
        multiplying that with x,\
            results in a new x whose channels are prioarized by\
                the values returns from self.se(x)
        '''
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride, padding, expand_ratio,
        reduction=4,  # sqeeze excitation
        survival_prob=0.8  # for stoichastic depth
    ):
        super().__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduce_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3,
                stride=1, padding=1
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size,
                stride, padding, groups=hidden_dim  # depth wise conv op
            ),
            SqueezeExitation(hidden_dim, reduce_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def stoichastic_depth(self, x):
        if self.training == False:
            return x

        # values between 0 and 1
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        binary_tensor = binary_tensor < self.survival_prob

        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            # stoichastic_depth will return some entirley 0, so to make sure
            # we do not removes all the information we add inputs at the end
            return self.stoichastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, n_classes):
        super().__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(
            version
        )
        last_chnnels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(
            width_factor, depth_factor, last_chnnels
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_chnnels, n_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi  # to control the n of layers
        width_factor = beta ** phi  # to control the n of channels
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4*ceil(int(channels*width_factor)/4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels, out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size//2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(CNNBlock(
            in_channels, last_channels, kernel_size=1,
            stride=1, padding=0
        ))

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    version = 'b0'
    phi, res, drop_rate = phi_values[version]
    n_examples, n_classes, = 4, 10

    x = torch.randn((n_examples, 3, res, res)).to(device)
    model = EfficientNet(version, n_classes).to(device)

    print(model(x).shape)
    print(model)


if __name__ == '__main__':
    # test()
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    print(model)
