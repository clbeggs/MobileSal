import torch
from torch import nn


class RGBAtten(nn.Module):
    """RGB Attention module, equation (2) of paper"""

    def __init__(self, in_channel):
        super(RGBAtten, self).__init__()

        # N x 320 x 10 x 10
        # FC is replaced with Conv
        self.model = nn.Sequential(
            nn.AvgPool3d(kernel_size=1),
            nn.Conv2d(
                in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channel, out_channels=in_channel, kernel_size=1, stride=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class RGBAttenFC(nn.Module):
    """RGB Attention module, equation (2) of paper, with fully connected """
    # N x 320 x 10 x 10

    def __init__(self, in_channel, size=10):
        super(RGBAttenFC, self).__init__()

        # FC layer is used instead of Conv
        self.model = nn.Sequential(
            nn.AvgPool3d(kernel_size=1),
            nn.Linear(size, size), # 10 x 10
            nn.ReLU(),
            nn.Linear(size, size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
