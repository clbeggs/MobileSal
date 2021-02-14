import os
import numpy as np
import torch
from torch import Tensor
from torch import nn
from typing import Callable, Any, Optional, List
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from blocks import *


torch.manual_seed(28)
np.random.seed(28)


class DepthBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthBlock, self).__init__()
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )


class DepthStreamStage(nn.Module):
    """
    InvertedResidual -> 1x1 Conv -> 3x3 Depthwise_Seperable_Conv -> 1x1 Conv


    Depthwise Seperable Conv Reference:
        https://pytorch.org/docs/1.7.1/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    """

    def __init__(self, in_channels, out_channels, M=4) -> None:
        super(DepthStreamStage, self).__init__()

        self.model = nn.Sequential(
            InvertedResidual(
                inp=in_channels, oup=int(out_channels / M), expand_ratio=M, stride=2
            ),
            InvertedResidual(
                inp=int(out_channels / M), oup=out_channels, expand_ratio=M, stride=1
            ),
        )

    def forward(self, x):
        return self.model(x)


class DepthStream(nn.Module):
    """
    The output feature maps of five stages of the depth streamare denoted as
    D1,D2,D3,D4,D5, the first four of whichhave 16,32,64,96
    D5 and C5 have the same number of output channels and the same stride.
    """

    def __init__(self):
        super(DepthStream, self).__init__()

        # out channels as per literature: [16, 32, 64, 96, 320]
        self.d1 = DepthStreamStage(in_channels=1, out_channels=16)
        self.d2 = DepthStreamStage(in_channels=16, out_channels=32)
        self.d3 = DepthStreamStage(in_channels=32, out_channels=64)
        self.d4 = DepthStreamStage(in_channels=64, out_channels=96)
        self.d5 = DepthStreamStage(in_channels=96, out_channels=320)

    def forward(self, depth_map):
        out1 = self.d1(depth_map)
        out2 = self.d2(out1)
        out3 = self.d3(out2)
        out4 = self.d4(out3)
        out5 = self.d5(out4)
        return (out1, out2, out3, out4, out5)


class RGBStream(nn.Module):
    def __init__(self, mbnet_params):
        super(RGBStream, self).__init__()
        # Mobilenetv2
        self.mobile_net = MobileNetV2(
            inverted_residual_setting=mbnet_params, last_channel=320
        )

    def forward(self, image):
        # Get all base feature maps from mobilenet
        (c1, c2, c3, c4, c5) = self.mobile_net(image)
        return (c1, c2, c3, c4, c5)


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=1, stride=1, kernel_size=1
        )
        self.activ = nn.Sigmoid()

    def forward(self, x):
        hid = self.activ(self.conv(x))
        return F.interpolate(hid, [320, 320], mode="bilinear", align_corners=False)


class MobileSal(nn.Module):
    """Implementation of MobileSal model.

    Paper: MobileSal: Extremely Efficient RGB-D Salient Object Detection
    Arxiv: https://arxiv.org/pdf/2012.13095.pdf

    forward pseudo code(self, rgb_image, depth_map):

        rgb_features, residual_connections = mobilenet_v2(rgb_image)

        depth_features = depth_stream(depth_map)

        refined = cmf(rgb_features, depth_features)

        depth_restore = idr(refined, residual_connections)

        out = rgbd_stream(residual_connections, refined)

        return out
    """

    def __init__(self, mobilenet_params):
        super(MobileSal, self).__init__()

        self.rgb_stream = RGBStream(mobilenet_params)
        self.CMF = CrossModalityFusion()
        self.depth_stream = DepthStream()

        self.idr = ImplicitDepthRestore(16, 32, 64, 96, 320)

        # The strange in_channels are from the fact that we concatenate after each CPR block
        self.cpr5 = CompactPyramidRefine(in_channels=32, out_channels=8)
        self.cpr4 = CompactPyramidRefine(in_channels=64, out_channels=16)
        self.cpr3 = CompactPyramidRefine(in_channels=128, out_channels=32)
        self.cpr2 = CompactPyramidRefine(in_channels=192, out_channels=64)
        self.cpr1 = CompactPyramidRefine(in_channels=320, out_channels=96)

        self.decode1 = Decoder(96)
        self.decode2 = Decoder(64)
        self.decode3 = Decoder(32)
        self.decode4 = Decoder(16)
        self.decode5 = Decoder(16)

    def forward(self, rgb_image, depth_map):
        (c1, c2, c3, c4, c5) = self.rgb_stream(rgb_image)

        (d1, d2, d3, d4, d5) = self.depth_stream(depth_map)

        # CMF - Eq (3) of mobile sal paper
        c5D = self.CMF(c5, d5)
        restored_depth = self.idr(c1, c2, c3, c4, c5D)

        pass1 = self.cpr1(c5D)
        pred1 = self.decode1(pass1)

        pass2 = self.cpr2(torch.cat((F.interpolate(pass1, [20, 20]), c4), 1))
        pred2 = self.decode2(pass2)

        pass3 = self.cpr3(torch.cat((F.interpolate(pass2, [40, 40]), c3), 1))
        pred3 = self.decode3(pass3)

        pass4 = self.cpr4(torch.cat((F.interpolate(pass3, [80, 80]), c2), 1))
        pred4 = self.decode4(pass4)

        pred5 = self.decode5(pass4)

        return (pred1, pred2, pred3, pred4, pred5), restored_depth


class MobileSalTranspose(nn.Module):
    """Implementation of MobileSal model.

    Paper: MobileSal: Extremely Efficient RGB-D Salient Object Detection
    Arxiv: https://arxiv.org/pdf/2012.13095.pdf

    forward pseudo code(self, rgb_image, depth_map):

        rgb_features, residual_connections = mobilenet_v2(rgb_image)

        depth_features = depth_stream(depth_map)

        refined = cmf(rgb_features, depth_features)

        depth_restore = idr(refined, residual_connections)

        out = rgbd_stream(residual_connections, refined)

        return out
    """

    def __init__(self, mobilenet_params):
        super(MobileSalTranspose, self).__init__()
        self.loss_fn = nn.BCELoss()

        self.rgb_stream = RGBStream(mobilenet_params)
        self.CMF = CrossModalityFusion()
        self.depth_stream = DepthStream()

        self.idr = ImplicitDepthRestore(16, 32, 64, 96, 320)
        self.cpr1 = CompactPyramidRefine(in_channels=16)
        self.cpr2 = CompactPyramidRefine(in_channels=32)
        self.cpr3 = CompactPyramidRefine(in_channels=64)
        self.cpr4 = CompactPyramidRefine(in_channels=96)
        self.cpr5 = CompactPyramidRefine(in_channels=320)

        self.decode1 = Decoder(320)
        self.decode2 = Decoder(96)
        self.decode3 = Decoder(64)
        self.decode4 = Decoder(32)
        self.decode5 = Decoder(16)

        # yapf: disable
        self.convT1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        self.convT2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        self.convT3 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        self.convT4 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        # yapf: enable

    def forward(self, rgb_image, depth_map):
        (c1, c2, c3, c4, c5) = self.rgb_stream(rgb_image)

        (d1, d2, d3, d4, d5) = self.depth_stream(depth_map)

        # CMF - Eq (3) of mobile sal paper
        c5D = self.CMF(c5, d5)
        restored_depth = self.idr(c1, c2, c3, c4, c5D)

        pass1, pred1 = self.decode1(self.cpr1(c5D))
        pass2, pred2 = self.decode2(self.cpr2(self.convT1(pass1) + c4))
        pass3, pred3 = self.decode3(self.cpr3(self.convT2(pass2) + c3))
        pass4, pred4 = self.decode4(self.cpr4(self.convT3(pass3) + c2))
        _, pred5 = self.decode5(self.cpr5(self.convT4(pass4) + c1))

        return (pred1, pred2, pred3, pred4, pred5), restored_depth
