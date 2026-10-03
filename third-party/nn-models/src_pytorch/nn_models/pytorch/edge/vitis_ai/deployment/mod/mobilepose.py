"""Modified MobilePose for deployment.

It allows to generate model files
with the `PixelShuffle` operations deployed to the CPU,
given they are not being correctly deployed to the DPU.

"""

from typing import Tuple

import torch

from .....keypoints_regression.mobilepose import MobilePose_Edge


class _DUC(torch.nn.Module):
    """(Modified) Dense Upsampling Convolution model class.

    With fused Conv2d and BatchNorm as in the quantized models.

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
        self.conv = torch.nn.Conv2d(in_channels, up_channels, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.upscale = torch.nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.upscale(x)
        return x


class MobilePose_Edge_DUC(MobilePose_Edge):
    """(Partial) MobilePose model class, non quantizable (post-processing) operations.

    Includes the DUCs that are not correctly implemented on the DPU by Xilinx.

    Parameters
    ----------
    keypoints : int, optional
        Number of keypoints to be produced.
    normalized_coordinates : bool, optional
        Selects normalized coordinates in range ]-1; 1[ or coordinates on the heatmap size.
    """

    def __init__(self, keypoints: int = 8, normalized_coordinates: bool = False) -> None:
        super().__init__(normalized_coordinates)
        self.duc1 = _DUC(256, 512)
        self.duc2 = _DUC(128, 256)
        self.duc3 = _DUC(64, 128)
        self.conv_heatmap = torch.nn.Conv2d(32, keypoints, kernel_size=1, bias=False)

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Runs only the DUCs and the post-processing operations.

        Parameters
        ----------
        feature_map : torch.Tensor
            backbone feature map, with batch as first dimension.

        Returns
        -------
        coords : torch.Tensor
            (Keypoint) coordinates, with batch as first dimension.
        heatmaps : torch.Tensor
            (Keypoint) heatmaps, with batch as first dimension.
        """
        x = self.duc1(feature_map)
        x = self.duc2(x)
        x = self.duc3(x)
        x = self.conv_heatmap(x)
        return self.post(x)

