"""Keras (TensorFlow 2) Feature Pyramid Network (FPN) module.

Model implemented using Functional API.

Notes
-----
The implementation follows the original paper [1]_.

References
----------
.. [1] "Feature Pyramid Networks for Object Detection", https://arxiv.org/abs/1612.03144

"""

from typing import Tuple, Optional, List
from enum import Enum

from tensorflow import keras

from .utils._utils import _opname


class FPNlayers(Enum):
    P2toP6MaxPool = 1
    P3toP6P7Conv = 2


class FPNstrides:
    P2toP6MaxPool = (4, 8, 16, 32, 64)
    P3toP6P7Conv = (8, 16, 32, 64, 128)


def building_block(top, lat, filters: int = 256, name: Optional[str] = None) -> Tuple:
    """Building block for FPNs.

    Parameters
    ----------
    top : tensor
        Tensor from the upper layer.
    lat : tensor
        Tensor from the lateral layer (previous model).
    filters : int, optional
        Number of channels of this block.
    name : str, optional
        Layer name prefix.

    Returns
    -------
    bot : tensor
        Internal output from the current layer to be fed to the next layer.
    out : tensor
        External output from the current model layer to the next model.
    """
    lat = keras.layers.Conv2D(filters, kernel_size=1, strides=1, name=_opname("{}_conv_in", name))(lat)
    if top is not None:
        top = keras.layers.UpSampling2D(size=2, interpolation="nearest", name=_opname("{}_upsample", name))(top)
        bot = keras.layers.Add(name=_opname("{}_add", name))([top, lat])
    else:
        bot = lat
    out = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", name=_opname("{}_conv_out", name))(bot)

    return (bot, out)

def fpn(inputs: Tuple, layers: FPNlayers = FPNlayers.P2toP6MaxPool, filters: int = 256) -> List:
    """Feature Pyramid Network model function.

    Parameters
    ----------
    inputs : tuple of tensor
        Inputs to the model, from the previous model layers.
    layers : enum, optional
        Selects the FPN architecture, by default it is the one in the paper.
        Possible options are defined in the `FPNlayers` enumeration.
    filters : int, optional
        Number of channels of each building block.

    Returns
    -------
    list of tensor
        Outputs of all the model layers.
    """
    (c2, c3, c4, c5) = inputs

    (x, p5) = building_block(None, c5, filters, "pyramid5")
    (x, p4) = building_block(x, c4, filters, "pyramid4")
    (x, p3) = building_block(x, c3, filters, "pyramid3")

    # FPN with P2 and top P6 MaxPool layer
    if layers is FPNlayers.P2toP6MaxPool:
        (x, p2) = building_block(x, c2, filters, "pyramid2")

        p6 = keras.layers.MaxPool2D(pool_size=1, strides=2, name="pyramid6_maxpool")(p5)
        return [p2, p3, p4, p5, p6]
    # FPN with no P2 and top P6 and P7 Conv layers
    elif layers is FPNlayers.P3toP6P7Conv:
        p6 = keras.layers.Conv2D(filters, kernel_size=3, strides=2, padding="same", name="pyramid6_conv")(p5)
        # NOTE workaround for "Standalone activation `relu` is not supported":
        # move ReLU to pyramid6_conv using activation=keras.activations.relu,
        # comment the next line (standalone ReLU) and pass p6 directly to pyramid7_conv.
        x = keras.layers.ReLU(name="pyramid6_relu")(p6)
        p7 = keras.layers.Conv2D(filters, kernel_size=3, strides=2, padding="same", name="pyramid7_conv")(x)
        return [p3, p4, p5, p6, p7]
    else:
        enum = {e.name:e.value for e in FPNlayers}
        raise ValueError(f"Expecting one of {enum}, but got {layers}...")

