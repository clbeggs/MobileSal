import torch
from torch import nn
import torch.nn.functional as F
from .mobilenet import InvertedResidual


class ImplicitDepthRestore(nn.Module):
    """Implicit Depth Restoration
    The IDR branch strengthens the less powerful features of mobile backbonenetwork.

    OMITTED DURING TESTING!
    """

    def __init__(self, chan1, chan2, chan3, chan4, chan5) -> None:

        super(ImplicitDepthRestore, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=chan1, out_channels=256, stride=2, kernel_size=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=chan2, out_channels=256, stride=2, kernel_size=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=chan3, out_channels=256, stride=2, kernel_size=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=chan4, out_channels=256, stride=2, kernel_size=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=chan5, out_channels=256, stride=2, kernel_size=1
        )

        # self.upsample = nn.functional.interpolate((256, 40, 40), mode='bilinear')
        self.concat = nn.Conv2d(
            in_channels=1280, out_channels=256, stride=1, kernel_size=1
        )
        self.irb = nn.Sequential(
            InvertedResidual(inp=256, oup=256, stride=1, expand_ratio=1),
            InvertedResidual(inp=256, oup=256, stride=1, expand_ratio=1),
            InvertedResidual(inp=256, oup=256, stride=1, expand_ratio=1),
            InvertedResidual(inp=256, oup=256, stride=1, expand_ratio=1),
            nn.Conv2d(in_channels=256, out_channels=1, stride=1, kernel_size=1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, c1, c2, c3, c4, c5D):

        c1 = F.interpolate(
            self.conv1(c1), [40, 40], mode="bilinear", align_corners=False
        )
        c2 = F.interpolate(
            self.conv2(c2), [40, 40], mode="bilinear", align_corners=False
        )
        c3 = F.interpolate(
            self.conv3(c3), [40, 40], mode="bilinear", align_corners=False
        )
        c4 = F.interpolate(
            self.conv4(c4), [40, 40], mode="bilinear", align_corners=False
        )
        c5 = F.interpolate(
            self.conv5(c5D), [40, 40], mode="bilinear", align_corners=False
        )

        concatenated = self.concat(torch.cat((c1, c2, c3, c4, c5), 1))
        inv = self.irb(concatenated)

        return F.interpolate(
            self.sig(inv), [320, 320], mode="bilinear", align_corners=False
        )
