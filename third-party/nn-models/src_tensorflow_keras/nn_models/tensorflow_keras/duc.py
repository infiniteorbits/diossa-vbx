"""Keras (TensorFlow 2) Dense Upsampling Convolution (DUC) module.

Model implemented using Functional API.

Notes
-----
The implementation follows the original paper [1]_.

References
----------
.. [1] "Understanding Convolution for Semantic Segmentation", https://arxiv.org/abs/1702.08502

"""

from typing import Optional

from tensorflow import keras

from .nn import pixelshuffle
from .utils._utils import _opname


def duc(tensor, filters: int, upscale_factor: int = 2, name: Optional[str] = None):
    """Dense Upsampling Convolution model constructor.

    Parameters
    ----------
    tensor : tensor
        Input tensor.
    filters : int
        Number of inner channels in this block, to upscale.
    upscale_factor : int, optional
        Upscale ratio between input and output (height and width).
    name : str, optional
        Layer name prefix.

    Returns
    -------
    tensor
        Tensor to be fed to the next layer.
    """
    x = keras.layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False,
                            name=_opname("{}_conv", name))(tensor)
    x = keras.layers.BatchNormalization(name=_opname("{}_bn", name))(x)
    x = keras.layers.ReLU(name=_opname("{}_relu", name))(x)
    # PixelShuffle using Functional API
    # x = pixelshuffle.pixelshuffle(x, upscale_factor, _opname("{}_upscale", name))
    # PixelShuffle using layer subclassing
    x = pixelshuffle.PixelShuffle(upscale_factor, name=_opname("{}_upscale", name))(x)
    return x

