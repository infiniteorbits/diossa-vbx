"""PyTorch MobilePose module.

Model implemented using torch.nn.Module subclassing.

Notes
-----
The implementation loosely follows the original paper [1]_,
to match the off-the-shelf model [2]_.
- The backbone is the same, but unmodified.
- The deconvolutions are replaced with dense upsampling convolutions.
- The heatmaps to keypoints computation is performed with
a differentiable spatial to numerical transform.

References
----------
.. [1] "MobilePose: Real-Time Pose Estimation for Unseen Objects with Weak Shape Supervision", https://arxiv.org/abs/2003.03522
.. [2] "GitHub: MobilePose", https://github.com/YuliangXiu/MobilePose

"""

from typing import Tuple, Dict

import torch
import dsntnn

from ..mobilenet import MobileNetV2Layers as MobileNetV2
from ..duc import DUC

from ..utils.loaders import load_ots_mobilepose

from .keypoints_regression import KeypointsRegression


class MobilePose_Edge(torch.nn.Module):
    """(Partial) MobilePose model class, non quantizable (post-processing) operations.

    Source code can be embedded as post-processing of the edge model.

    Parameters
    ----------
    normalized_coordinates : bool, optional
        Selects normalized coordinates in range ]-1; 1[ or coordinates on the heatmap size.
    """

    def __init__(self, normalized_coordinates: bool = False) -> None:
        super().__init__()

        self.normalized_coordinates = normalized_coordinates

    def forward(self, unnormalized_heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Runs only the post-processing operations.

        Parameters
        ----------
        unnormalized_heatmaps : torch.Tensor
            (Keypoint) unnormalized heatmaps, with batch as first dimension.

        Returns
        -------
        coords : torch.Tensor
            (Keypoint) coordinates, with batch as first dimension.
        heatmaps : torch.Tensor
            (Keypoint) heatmaps, with batch as first dimension.
        """
        return self.post(unnormalized_heatmaps)

    def post(self, unnormalized_heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalizes the heatmaps and calculates the coordinates.

        Parameters
        ----------
        unnormalized_heatmaps : torch.Tensor
            Unnormalized heatmaps.

        Returns
        -------
        coords : torch.Tensor
            (Keypoint) coordinates, with batch as first dimension.
            Range depends on `normalized_coordinates` boolean.
        heatmaps : torch.Tensor
            (Keypoint) heatmaps, with batch as first dimension.
        """
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        coords = dsntnn.dsnt(heatmaps, normalized_coordinates=self.normalized_coordinates)
        return coords, heatmaps


class MobilePose(MobilePose_Edge, KeypointsRegression):
    """MobilePose model class.

    Submodels:
    1. MobileNetV2
    2. DUC
    3. (DSNT)

    Parameters
    ----------
    channels : int, optional
        Number of channels on the image.
    keypoints : int, optional
        Number of keypoints to be produced.
    normalized_coordinates : bool, optional
        Selects normalized coordinates in range ]-1; 1[ or coordinates on the heatmap size.
        For training must be `normalized_coordinates = True`
        due to limitation in the `compute_loss` method.
    ignore_invisible_keypoints : bool, optional
        If True ignores the training loss of the invisible keypoints.
    """

    # operation to be used for tensor addition
    # (if None then uses standard operator '+')
    _op_add = None

    def __init__(self, channels: int = 3, keypoints: int = 8,
                 normalized_coordinates: bool = True,
                 ignore_invisible_keypoints: bool = True) -> None:
        super().__init__(normalized_coordinates)
        self.backbone = MobileNetV2(channels, op_add=self._op_add)
        self.conv_compress = torch.nn.Conv2d(1280, 256, kernel_size=1, bias=False)
        self.duc1 = DUC(256, 512)
        self.duc2 = DUC(128, 256)
        self.duc3 = DUC(64, 128)
        self.conv_heatmap = torch.nn.Conv2d(32, keypoints, kernel_size=1, bias=False)

        self.ignore_invisible_keypoints = True if ignore_invisible_keypoints else None

    def forward(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed by the forward pass.

        Runs only the quantizable operations, without further post-processing.

        Parameters
        ----------
        batch_imgs : torch.Tensor
            Batch of images (NCHW).

        Returns
        -------
        unnormalized_heatmaps : torch.Tensor
            (Keypoint) unnormalized_heatmaps, with batch as first dimension.
        """
        x = self.backbone(batch_imgs)
        x = self.conv_compress(x[-1])
        x = self.duc1(x)
        x = self.duc2(x)
        x = self.duc3(x)
        x = self.conv_heatmap(x)
        return x

    def compute_loss(self,
                     coords: torch.Tensor,
                     heatmaps: torch.Tensor,
                     ground_truth_keypoints: torch.Tensor,
                     ground_truth_visibility: torch.Tensor
                    ) -> Dict[str, torch.Tensor]:
        
        """Computes the inference loss related to the ground truth.

        Parameters
        ----------
        coords : torch.Tensor
            Predicted keypoints coords (N,num_keypoints,2) shape.
            Only supports normalized coordinates in range ]-1; 1[.
        heatmaps: torch.Tensor
            Predicted keypoints heatmaps (N,num_keypoints,56,56) shape.
        ground_truth_keypoints: torch.Tensor
            Ground truth keypoints coords (N,num_keypoints,2) shape
        ground_truth_visibility: torch.Tensor
            Ground truth visibility flags (N,num_keypoints) shape.
            Ignored if model was created with ignore_invisible_keypoints = True

        Returns
        -------
        losses : Dict[str, torch.Tensor]
            Dictionary with all loss components. Coordinates loss and heatmaps loss.
        """
        coordinates_loss = dsntnn.euclidean_losses(coords, ground_truth_keypoints)
        heatmaps_loss = dsntnn.js_reg_losses(heatmaps, ground_truth_keypoints, sigma_t=1.0)

        if self.ignore_invisible_keypoints is not None:
            coordinates_loss = coordinates_loss[ground_truth_visibility == 1]
            heatmaps_loss = heatmaps_loss[ground_truth_visibility == 1]

        return {'coords': coordinates_loss, 'heatmaps': heatmaps_loss}
    
    def load_ots_model(self, model_file: str, load_final_conv_layer: bool = True):
        """Loads parameters from an off-the-shelf model.

        Parameters
        ----------
        model_file : str
            Off-the-shelf model file path.
        load_final_conv_layer : bool = True
            Whether to load final conv layer or not. This layer defines the number of keypoints.
        """
        load_ots_mobilepose(self, model_file, load_final_conv_layer=load_final_conv_layer)


class MobilePose_Full(MobilePose):
    """MobilePose complete model class."""

    def forward(self, batch_imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Runs the full model, both quantizable and non-quantizable operations.

        Parameters
        ----------
        batch_imgs : torch.Tensor
            Batch of images (NCHW).

        Returns
        -------
        coords : torch.Tensor
            (Keypoint) coordinates, with batch as first dimension.
        heatmaps : torch.Tensor
            (Keypoint) heatmaps, with batch as first dimension.
        """
        x = super().forward(batch_imgs)
        return self.post(x)

