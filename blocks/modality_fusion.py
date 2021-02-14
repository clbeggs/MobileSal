import torch
from torch import nn
from .mobilenet import InvertedResidual
from .rgb_attention import RGBAtten


class CrossModalityFusion(nn.Module):
    """Combine 5th feature maps
    which from paper have 320 channels.
    """

    def __init__(self):
        super(CrossModalityFusion, self).__init__()
        self.combine = InvertedResidual(
            inp=320, oup=320, stride=1, expand_ratio=1
            )
        self.out = InvertedResidual(inp=320, oup=320, stride=1, expand_ratio=1)
        self.att = RGBAtten(320)

    def forward(self, rgb_map, depth_map):
        """rgb_map is the 5th feature map for rgb """
        return self.out(
            self.att(rgb_map) * self.combine(rgb_map * depth_map) * depth_map
            )
