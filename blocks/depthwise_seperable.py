import torch
from torch import nn


class Depthwise_Seperable_Conv(nn.Module):
    """Depthwise Seperable Convolution implementation, with variable dilation rates
    for the Compact Pyramid Refinement block.

    Structure:
        Depthwise Conv(3x3) -> Pointwise Conv(1x1)

    Output Size:
        Input Image Size: (n x n)

        (n - dilation * (2) - 1) + 1

    Refs:
        MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
        Pytorch Conv2d Docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """

    def __init__(self, in_channels, out_channels, dilation, kernel_size=3) -> None:
        super(Depthwise_Seperable_Conv, self).__init__()

        self.depth = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            dilation=dilation,
            padding=dilation,
        )

        self.point = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            dilation=dilation,
        )

    def forward(self, x):
        return self.point(self.depth(x))
