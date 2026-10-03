"""Keras (TensorFlow 2) ResNet module.

Model implemented using Functional API.

Notes
-----
The implementation follows the original paper [1]_ (ResNet v1.0),
with the modification described in [2]_ (ResNet v1.5).

References
----------
.. [1] "Deep residual learning for image recognition", https://arxiv.org/abs/1512.03385
.. [2] https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch

"""

from typing import Tuple, Optional, Callable

from tensorflow import keras

from .utils._utils import _opname


def basic_block(tensor, filters: int, downsample: bool, name: Optional[str] = None):
    """Basic building block for small ResNets.

    Parameters
    ----------
    tensor : tensor
        Output tensor from the previous layer.
    filters : int
        Number of input channels of this block.
    downsample : bool
        If true enables downsampling.
    name : str, optional
        Layer name prefix.

    Returns
    -------
    tensor
        Tensor to be fed to the next layer.
    """
    x = keras.layers.Conv2D(filters, kernel_size=3, strides=2 if downsample else 1, padding="same",
                            use_bias=False, name=_opname("{}_conv1", name))(tensor)
    x = keras.layers.BatchNormalization(epsilon=1.001e-5, name=_opname("{}_bn1", name))(x)
    x = keras.layers.ReLU(name=_opname("{}_relu1", name))(x)

    x = keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="same",
                            use_bias=False, name=_opname("{}_conv2", name))(x)
    x = keras.layers.BatchNormalization(epsilon=1.001e-5, name=_opname("{}_bn2", name))(x)

    if downsample:
        shortcut = keras.layers.Conv2D(filters, kernel_size=1, strides=2,
                                       use_bias=False, name=_opname("{}_conv", name))(tensor)
        shortcut = keras.layers.BatchNormalization(epsilon=1.001e-5,
                                                   name=_opname("{}_bn", name))(shortcut)
    else:
        shortcut = tensor

    x = keras.layers.Add(name=_opname("{}_add", name))([x, shortcut])
    x = keras.layers.ReLU(name=_opname("{}_relu", name))(x)

    return x

def bottleneck(tensor, filters: int, downsample: bool, name: Optional[str] = None):
    """Bottleneck building block for big ResNets.

    Parameters
    ----------
    tensor : tensor
        Output tensor from the previous layer.
    filters : int
        Number of input channels of this block.
    downsample : bool
        If true enables downsampling.
    name : str, optional
        Layer name prefix.

    Returns
    -------
    tensor
        Tensor to be fed to the next layer.
    """
    x = keras.layers.Conv2D(filters, kernel_size=1, strides=1,
                            use_bias=False, name=_opname("{}_conv1", name))(tensor)
    x = keras.layers.BatchNormalization(epsilon=1.001e-5, name=_opname("{}_bn1", name))(x)
    x = keras.layers.ReLU(name=_opname("{}_relu1", name))(x)

    x = keras.layers.Conv2D(filters, kernel_size=3, strides=2 if downsample else 1, padding="same",
                            use_bias=False, name=_opname("{}_conv2", name))(x)
    x = keras.layers.BatchNormalization(epsilon=1.001e-5, name=_opname("{}_bn2", name))(x)
    x = keras.layers.ReLU(name=_opname("{}_relu2", name))(x)

    x = keras.layers.Conv2D(filters*4, kernel_size=1, strides=1,
                            use_bias=False, name=_opname("{}_conv3", name))(x)
    x = keras.layers.BatchNormalization(epsilon=1.001e-5, name=_opname("{}_bn3", name))(x)

    # for the bottleneck a convolution is also needed on the shortcut
    # in case the input shape does not match the output shape
    if downsample or filters*4 != tensor.shape[-1]:
        shortcut = keras.layers.Conv2D(filters*4, 1, 2 if downsample else 1,
                                       use_bias=False, name=_opname("{}_conv", name))(tensor)
        shortcut = keras.layers.BatchNormalization(epsilon=1.001e-5,
                                                   name=_opname("{}_bn", name))(shortcut)
    else:
        shortcut = tensor

    x = keras.layers.Add(name=_opname("{}_add", name))([x, shortcut])
    x = keras.layers.ReLU(name=_opname("{}_relu", name))(x)

    return x

def resnet(block: Callable,
           stacks: Tuple[int, int, int, int],
           tensor,
           num_classes: int = 1000,
           output_layers: bool = False):
    """ResNet generic model function.

    Parameters
    ----------
    block : {basic_block, bottleneck}
        Building block function.
    stacks : tuple of int
        Number of stacks per layer.
    tensor : tensor
        Input tensor.
    num_classes : int, optional
        Number of classes to be produced.
    output_layers : bool, optional
        If `True` the model output is the feature maps (all layers outputs),
        otherwise the output is the features (last layer).

    Returns
    -------
    tensor or list of tensors
        Output(s) of the model.

    Raises
    ------
    ValueError
        If `block` is not one of the two supported options.
    TypeError
        If `output_layers` arg is not a bool.
    """
    if block is not basic_block and block is not bottleneck:
        raise ValueError(f"Expected basic_block or bottleneck, but got {block}")

    if not isinstance(output_layers, bool):
        raise TypeError(f"Expected a bool, but got: {type(output_layers)}")

    # filters -> tuple with the shape at the input of each layer
    filters = (64, 64, 128, 256, 512)

    # layer 0
    layer = 0
    x = keras.layers.Conv2D(filters[layer], kernel_size=7, strides=2, padding="same",
                            use_bias=False, name="conv1")(tensor)
    x = keras.layers.BatchNormalization(epsilon=1.001e-5, name="bn1")(x)
    x = keras.layers.ReLU(name="relu1")(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="maxpool")(x)

    outputs = []
    # layers 1 to 4
    for layer in range(1, 5):
        # the first stack includes downsampling for layers 2 to 4
        x = block(x, filters[layer], downsample=layer>1, name=f"conv{layer+1}_1")
        # remaining stacks do not include downsampling
        for stack in range(1, stacks[layer-1]):
            x = block(x, filters[layer], downsample=False, name=f"conv{layer+1}_{stack+1}")
        outputs.append(x)

    if not output_layers:
        x = keras.layers.GlobalAveragePooling2D(name="avgpool")(x)
        x = keras.layers.Dense(num_classes, activation="softmax", name="fc")(x)

        return x
    else:
        return outputs

def resnet18(*args, **kwargs):

    return resnet(basic_block, (2, 2, 2, 2), *args, **kwargs)

def resnet34(*args, **kwargs):

    return resnet(basic_block, (3, 4, 6, 3), *args, **kwargs)

def resnet50(*args, **kwargs):

    return resnet(bottleneck, (3, 4, 6, 3), *args, **kwargs)

def resnet101(*args, **kwargs):

    return resnet(bottleneck, (3, 4, 23, 3), *args, **kwargs)

def resnet152(*args, **kwargs):

    return resnet(bottleneck, (3, 8, 36, 3), *args, **kwargs)


class ResNet18(keras.Model):
    """ResNet18 model class.

    Parameters
    ----------
    img_size : tuple of int, optional
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    """

    def __init__(self,
                 img_size: Tuple[int, int] = (224, 224),
                 channels: int = 3,
                 num_classes: int = 1000) -> None:
        inp = keras.Input(shape=(*reversed(img_size), channels), name="img")
        out = resnet18(inp, num_classes)
        super().__init__(inputs=inp, outputs=out, name="ResNet18")


class ResNet34(keras.Model):
    """ResNet34 model class.

    Parameters
    ----------
    img_size : tuple of int, optional
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    """

    def __init__(self,
                 img_size: Tuple[int, int] = (224, 224),
                 channels: int = 3,
                 num_classes: int = 1000) -> None:
        inp = keras.Input(shape=(*reversed(img_size), channels), name="img")
        out = resnet34(inp, num_classes)
        super().__init__(inputs=inp, outputs=out, name="ResNet34")


class ResNet50(keras.Model):
    """ResNet50 model class.

    Parameters
    ----------
    img_size : tuple of int, optional
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    """

    def __init__(self,
                 img_size: Tuple[int, int] = (224, 224),
                 channels: int = 3,
                 num_classes: int = 1000) -> None:
        inp = keras.Input(shape=(*reversed(img_size), channels), name="img")
        out = resnet50(inp, num_classes)
        super().__init__(inputs=inp, outputs=out, name="ResNet50")


class ResNet101(keras.Model):
    """ResNet101 model class.

    Parameters
    ----------
    img_size : tuple of int, optional
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    """

    def __init__(self,
                 img_size: Tuple[int, int] = (224, 224),
                 channels: int = 3,
                 num_classes: int = 1000) -> None:
        inp = keras.Input(shape=(*reversed(img_size), channels), name="img")
        out = resnet101(inp, num_classes)
        super().__init__(inputs=inp, outputs=out, name="ResNet101")


class ResNet152(keras.Model):
    """ResNet152 model class.

    Parameters
    ----------
    img_size : tuple of int, optional
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    """

    def __init__(self,
                 img_size: Tuple[int, int] = (224, 224),
                 channels: int = 3,
                 num_classes: int = 1000) -> None:
        inp = keras.Input(shape=(*reversed(img_size), channels), name="img")
        out = resnet152(inp, num_classes)
        super().__init__(inputs=inp, outputs=out, name="ResNet152")

