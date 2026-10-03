"""PyTorch FCOS Vitis-AI compatibility module."""

from typing import Tuple

import torch

import pytorch_nndct.nn

from nn_models.pytorch.object_detection.fcos import FCOS


class FCOS_PTQ(FCOS):
    """(Partial) FCOS model class, for post training quantization.

    Overrides the norm_layer of `FCOS` class to use BatchNorm
    instead of GroupNorm, which is not supported by Vitis-AI
    """
    # Vitis-AI quantizer does not support GroupNorm, only BatchNorm
    norm_layer = torch.nn.BatchNorm2d


class FCOS_QAT(FCOS_PTQ):
    """(Partial) FCOS model class, for quantization aware training.

    Overrides the forward path from `FCOS` class
    to only include operations from torch.nn and quantization scope.
    """

    # operation to be used for tensor addition
    _op_add = pytorch_nndct.nn.functional.Add

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.quant = pytorch_nndct.nn.QuantStub()
        self.dequant = pytorch_nndct.nn.DeQuantStub()

    def forward(self, batch_imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Parameters
        ----------
        batch_imgs : torch.Tensor
            Batch of images (NCHW).

        Returns
        -------
        reg_bboxes : tuple of torch.Tensor
            Regressor bounding boxes.
        cls_logits : tuple of torch.Tensor
            Classifier score logits.
        centerness : tuple of torch.Tensor
            Distance to the center of the object.
        """
        return self.dequant(super().forward(self.quant(batch_imgs)))

