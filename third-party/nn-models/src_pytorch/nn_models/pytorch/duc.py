"""PyTorch Dense Upsampling Convolution (DUC) module.

Model implemented using torch.nn.Module subclassing.

Notes
-----
The implementation follows the original paper [1]_.

References
----------
.. [1] "Understanding Convolution for Semantic Segmentation", https://arxiv.org/abs/1702.08502

"""

import torch


class DUC(torch.nn.Module):
    """Dense Upsampling Convolution model class.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels to this block.
    up_channels : int
        Number of inner channels in this block, to upscale.
    upscale_factor : int, optional
        Upscale ratio between input and output (height and width).
    """

    def __init__(self, in_channels: int, up_channels: int, upscale_factor: int = 2) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, up_channels, kernel_size=3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(up_channels, momentum=0.1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.upscale = torch.nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upscale(x)
        return x

