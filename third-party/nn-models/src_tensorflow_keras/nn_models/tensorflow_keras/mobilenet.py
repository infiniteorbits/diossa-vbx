"""Keras (TensorFlow 2) MobileNet module.

Model implemented using Functional API.

Notes
-----
The implementation follows the original paper [1]_.

References
----------
.. [1] "MobileNetV2: Inverted Residuals and Linear Bottlenecks", https://arxiv.org/abs/1801.04381

"""

from typing import Optional, Tuple

from tensorflow import keras

from .utils._utils import _opname, _make_divisible


def inverted_residual(tensor, filters: int, stride: int,
                      expansion_ratio: int, name: Optional[str] = None):
    """Inverted residual building block for MobileNets.

    Parameters
    ----------
    tensor : tensor
        Output tensor from the previous layer.
    filters : int
        Number of output channels from this block.
    stride : int
        Stride of the depthwise convolution.
    expansion_ratio : int
        Ratio between the input channels and the inner channels.
    name : str, optional
        Layer name prefix.

    Returns
    -------
    tensor
        Tensor to be fed to the next layer.

    Raises
    ------
    ValueError
        If `stride` arg is not in (1, 2).
    """

    if stride not in (1, 2):
        raise ValueError(f"stride should be 1 or 2 instead of {stride}")

    in_channels = keras.backend.int_shape(tensor)[-1]
    hidden_dim = int(round(in_channels * expansion_ratio))

    if expansion_ratio != 1:
        # pw (pointwise)
        x = keras.layers.Conv2D(hidden_dim, kernel_size=1, use_bias=False,
                                name=_opname("{}_pw_conv", name))(tensor)
        x = keras.layers.BatchNormalization(momentum=0.999, name=_opname("{}_pw_bn", name))(x)
        x = keras.layers.ReLU(6.0, name=_opname("{}_pw_relu", name))(x)
    else:
        x = tensor

    # NOTE uncomment the next line to apply PyTorch's padding
    # x = keras.layers.ZeroPadding2D(padding=1)(x)

    # dw (depthwise)
    x = keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding="same",
                                     use_bias=False, name=_opname("{}_dw_conv", name))(x)
    x = keras.layers.BatchNormalization(momentum=0.999, name=_opname("{}_dw_bn", name))(x)
    x = keras.layers.ReLU(6.0, name=_opname("{}_dw_relu", name))(x)
    # pw-linear (pointwise)
    x = keras.layers.Conv2D(filters, kernel_size=1, use_bias=False,
                            name=_opname("{}_pwl_conv", name))(x)
    x = keras.layers.BatchNormalization(momentum=0.999, name=_opname("{}_pwl_bn", name))(x)

    if stride == 1 and in_channels == filters:
        x = keras.layers.Add(name=_opname("{}_add", name))([x, tensor])

    return x


def mobilenet_v2(tensor, num_classes: int = 1000, width_multiplier: float = 1.0,
                output_layers: bool = False, dropout: Optional[float] = None):
    """MobileNet (version 2) model function.

    Parameters
    ----------
    tensor : tensor
        Output tensor from the previous layer.
    num_classes : int, optional
        Number of classes to be produced.
    width_multiplier : float, optional
        Multiplier applied to the number of channels of each layer.
    output_layers : bool, optional
        If `True` the model output is the feature maps (all layers outputs),
        otherwise the output is the features (last layer).
    dropout : float, optional
        If a value is passed then
        a dropout layer is added with the given dropout value.

    Returns
    -------
    tensor or list of tensors
        Output(s) of the model.
    """

    inverted_residual_setting = [
        # t, c, n, s
        [None, 32, 1, 2],
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
        [None, 1280, 1, 1]
        ]

    # first layer
    layer = 0
    (t, c, n, s) = inverted_residual_setting[layer]
    filters = _make_divisible(c * width_multiplier)
    x = keras.layers.Conv2D(filters, kernel_size=3, strides=s,
                            padding="same", use_bias=False,
                            name=f"layer{layer}_conv")(tensor)
    x = keras.layers.BatchNormalization(momentum=0.999, name=f"layer{layer}_bn")(x)
    x = keras.layers.ReLU(6.0, name=f"layer{layer}_relu")(x)
    outputs = [x]
    # inverted residual blocks
    for layer in range(1, len(inverted_residual_setting) - 1):
        (t, c, n, s) = inverted_residual_setting[layer]
        filters = _make_divisible(c * width_multiplier)
        for stack in range(n):
            stride = s if stack == 0 else 1
            x = inverted_residual(x, filters, stride, expansion_ratio=t, name=f"layer{layer}_{stack+1}")
        outputs.append(x)
    # last layer
    layer = len(inverted_residual_setting) - 1
    (t, c, n, s) = inverted_residual_setting[layer]
    filters = _make_divisible(c * max(1.0, width_multiplier))
    x = keras.layers.Conv2D(filters, kernel_size=1, strides=s, use_bias=False,
                            name=f"layer{layer}_conv")(x)
    x = keras.layers.BatchNormalization(momentum=0.999, name=f"layer{layer}_bn")(x)
    x = keras.layers.ReLU(6.0, name=f"layer{layer}_relu")(x)
    outputs.append(x)

    if not output_layers:
        x = keras.layers.GlobalAveragePooling2D(name="avgpool")(x)
        if dropout is not None:
            x = keras.layers.Dropout(dropout, name="drop")(x)
        x = keras.layers.Dense(num_classes, activation="softmax", name="fc")(x)

        return x
    else:
        return outputs


class MobileNetV2(keras.Model):
    """MobileNet (version 2) model class.

    Parameters
    ----------
    img_size : tuple of int, optional
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    width_multiplier : float, optional
        Multiplier applied to the number of channels of each layer.
    dropout : float, optional
        If a value is passed then
        a dropout layer is added with the given dropout value.
    """

    def __init__(self, img_size: Tuple[int, int] = (224, 224), channels: int = 3,
                 num_classes: int = 1000, width_multiplier: float = 1.0,
                 dropout: float = 0.2) -> None:
        inp = keras.Input(shape=(*reversed(img_size), channels), name="img")
        out = mobilenet_v2(inp, num_classes, width_multiplier, dropout=dropout)
        super().__init__(inputs=inp, outputs=out, name="MobileNetV2")

