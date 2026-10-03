"""PyTorch Faster R-CNN Vitis-AI compatibility module."""

from typing import Tuple

import torch

import pytorch_nndct.nn

from nn_models.pytorch.object_detection.faster_rcnn import FasterRCNN


class FasterRCNN_PTQ(FasterRCNN):
    """(Partial) Faster R-CNN model class, for post training quantization.
    
    Overrides the forward path from `FasterRCNN` class
    to only include operations from torch.nn.
    """

    def forward(self, batch_imgs: torch.Tensor) \
            -> Tuple[
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                ]:
        """Defines the computation performed by the forward pass.

        Parameters
        ----------
        batch_imgs : torch.Tensor
            Batch of images (NCHW).

        Returns
        -------
        pyramid_feature_maps : tuple of torch.Tensor
            FPN pyramid feature maps.
        rpn_transformers : tuple of torch.Tensor
            RPN proposal transformers.
        rpn_objectnesses : tuple of torch.Tensor
            RPN proposal objectnesses.
        """
        feature_maps = self.resnet(batch_imgs)
        pyramid_feature_maps = self.fpn(feature_maps)

        rpn_transformers, rpn_objectnesses = self.rpn(pyramid_feature_maps)

        return pyramid_feature_maps[:-1], rpn_transformers, rpn_objectnesses


class FasterRCNN_QAT(FasterRCNN_PTQ):
    """(Partial) Faster R-CNN model class, for quantization aware training.
    
    Overrides the forward path from `FasterRCNN` class
    to only include operations from torch.nn and quantization scope.
    """

    # operation to be used for tensor addition
    _op_add = pytorch_nndct.nn.functional.Add

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.quant = pytorch_nndct.nn.QuantStub()
        self.dequant = pytorch_nndct.nn.DeQuantStub()

    def forward(self, batch_imgs: torch.Tensor) \
            -> Tuple[
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                ]:
        """Defines the computation performed by the forward pass.

        Parameters
        ----------
        batch_imgs : torch.Tensor
            Batch of images (NCHW).

        Returns
        -------
        pyramid_feature_maps : tuple of torch.Tensor
            FPN pyramid feature maps.
        rpn_transformers : tuple of torch.Tensor
            RPN proposal transformers.
        rpn_objectnesses : tuple of torch.Tensor
            RPN proposal objectnesses.
        """
        return self.dequant(super().forward(self.quant(batch_imgs)))

