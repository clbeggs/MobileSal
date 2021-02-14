import torch
from torch import nn
from .rgb_attention import RGBAtten
from .depthwise_seperable import Depthwise_Seperable_Conv


class Depthwise(nn.Module):
    def __init__(self, in_channels, M=4):
        super(Depthwise, self).__init__()
        self.input_conv = nn.Conv2d(
            kernel_size=1, in_channels=in_channels, out_channels=in_channels * M
        )

        self.depth1 = Depthwise_Seperable_Conv(in_channels * M, in_channels * M, 1)
        self.depth2 = Depthwise_Seperable_Conv(in_channels * M, in_channels * M, 2)
        self.depth3 = Depthwise_Seperable_Conv(in_channels * M, in_channels * M, 3)

        self.norm = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels * M),
            nn.ReLU(),
            nn.Conv2d(  # Squeeze channel size to input
                kernel_size=1, in_channels=in_channels * M, out_channels=in_channels
            ),
        )

        self.out_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, stride=1, kernel_size=1
        )

    def forward(self, x):
        x1 = self.input_conv(x)  # Get x1 in eq (5)

        # Computes x2 and squeezes channel num to input size, in eq (6)
        x2 = self.norm(self.depth1(x1) + self.depth2(x1) + self.depth3(x1))

        return self.out_conv(x2 + x)  # Eq. (6)


class CompactPyramidRefine(nn.Module):
    """Compact Pyramid Refinement block

    mutli-level deep features are efficiently aggregated by the CPR module

    """

    def __init__(self, in_channels, out_channels):
        super(CompactPyramidRefine, self).__init__()
        self.depth = Depthwise(in_channels=in_channels)
        self.att = RGBAtten(in_channels)

        self.out_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        return self.out_conv(self.depth(x) * self.att(x))  # Eq(7)
