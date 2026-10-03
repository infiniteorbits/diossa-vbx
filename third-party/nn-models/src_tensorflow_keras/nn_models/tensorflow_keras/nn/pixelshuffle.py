"""Keras (TensorFlow 2) PixelShuffle module.

Layer implemented using Functional API and `Layer` subclassing.

Notes
-----
The implementation reproduces `torch.nn.PixelShuffle` layer operation,
which itself follows the original paper [1]_.
TensorFlow includes a `tf.nn.depth_to_space` layer that is related,
but the channel order is different from PyTorch.

References
----------
.. [1] "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network", https://arxiv.org/abs/1609.05158

"""

from typing import Optional

from tensorflow import keras

from ..utils._utils import _opname


def pixelshuffle(tensor, upscale_factor: int, name: Optional[str] = None):
    """PixelShuffle layer constructor, using Functional API.

    Parameters
    ----------
    tensor : tensor
        Input tensor.
    upscale_factor : int
        Upscale ratio between input and output (height and width).
    name : str, optional
        Layer name prefix.

    Returns
    -------
    tensor
        Tensor to be fed to the next layer.
    """
    (n, h, w, d) = keras.backend.int_shape(tensor)
    r = upscale_factor
    c = d // r**2
    # Keras Reshape and Permute layers don't take the batchsize dim
    x = keras.layers.Reshape((h, w, c, r, r,), name=_opname("{}_split", name))(tensor)
    x = keras.layers.Permute((1, 4, 2, 5, 3,), name=_opname("{}_permute", name))(x)
    x = keras.layers.Reshape((h*r, w*r, c,), name=_opname("{}_merge", name))(x)
    return x


class PixelShuffle(keras.layers.Layer):
    """PixelShuffle layer class, using `Layer` subclassing.

    Parameters
    ----------
    upscale_factor : int
        Upscale ratio between input and output (height and width).
    kwargs
        Keyword arguments for `keras.layers.Layer` super class.
    """

    def __init__(self, upscale_factor: int, **kwargs):
        super().__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def build(self, input_shape):
        super().build(input_shape)
        (n, h, w, d) = input_shape
        r = self.upscale_factor
        c = d // r**2
        self.split = keras.layers.Reshape((h, w, c, r, r,), name=_opname("{}_split", self.name))
        self.permute = keras.layers.Permute((1, 4, 2, 5, 3,), name=_opname("{}_permute", self.name))
        self.merge = keras.layers.Reshape((h*r, w*r, c,), name=_opname("{}_merge", self.name))

    def call(self, inputs):
        x = self.split(inputs)
        x = self.permute(x)
        x = self.merge(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"upscale_factor": self.upscale_factor})
        return config

