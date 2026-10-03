"""PyTorch MobilePose Vitis-AI compatibility module."""

import torch

import pytorch_nndct.nn

from nn_models.pytorch.keypoints_regression.mobilepose import MobilePose


class MobilePose_PTQ(MobilePose):
    """(Partial) MobilePose model class, for post training quantization.

    Overrides the forward path from `MobilePose` class
    to only include operations from torch.nn.
    """
    pass


class MobilePose_QAT(MobilePose_PTQ):
    """(Partial) MobilePose model class, for quantization aware training.

    Overrides the forward path from `MobilePose` class
    to only include operations from torch.nn and quantization scope.
    """

    # operation to be used for tensor addition
    _op_add = pytorch_nndct.nn.functional.Add

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.quant = pytorch_nndct.nn.QuantStub()
        self.dequant = pytorch_nndct.nn.DeQuantStub()

    def forward(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed by the forward pass.

        Parameters
        ----------
        batch_imgs : torch.Tensor
            Batch of images (NCHW).

        Returns
        -------
        unnormalized_heatmaps : torch.Tensor
            (Keypoint) unnormalized heatmaps, with batch as first dimension.
        """
        return self.dequant(super().forward(self.quant(batch_imgs)))

