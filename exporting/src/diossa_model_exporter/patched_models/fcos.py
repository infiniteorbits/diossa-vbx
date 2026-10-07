import torch

from nn_models.pytorch.object_detection.fcos import FCOS

class FCOS_PTQ(FCOS):
    """(Partial) FCOS model class, for post training quantization.

    Overrides the norm_layer of `FCOS` class to use BatchNorm
    instead of GroupNorm, which is not supported by Vitis-AI
    """
    # Vitis-AI/MicroChip quantizer does not support GroupNorm, only BatchNorm
    norm_layer = torch.nn.BatchNorm2d
