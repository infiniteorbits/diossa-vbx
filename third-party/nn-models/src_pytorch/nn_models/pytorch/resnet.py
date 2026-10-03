"""PyTorch ResNet module.

Model implemented using torch.nn.Module subclassing and Sequential containers.

Notes
-----
The implementation follows the original paper [1]_ (ResNet v1.0),
with the modification described in [2]_ (ResNet v1.5).

References
----------
.. [1] "Deep residual learning for image recognition", https://arxiv.org/abs/1512.03385
.. [2] https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch

"""

from typing import Tuple, Optional, Type, Union, Callable
from collections import OrderedDict

import torch


class BasicBlock(torch.nn.Module):
    """Basic building block for small ResNets.

    Parameters
    ----------
    in_channels : int
        Number of input channels to this block.
    out_channels : int
        Number of output channels from this block.
    downsample : bool
        If true enables downsampling.
    op_add : callable, optional
        Shortcut (residual) addition operation.
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool, op_add: Optional[Callable] = None) -> None:
        super().__init__()
        # torchvision model uses FrozenBatchNorm2d from torchvision.ops.misc instead of BatchNorm2d
        # FrozenBatchNorm2d = BatchNorm2d where the batch statistics and the affine parameters are fixed
        # More info: https://github.com/facebookresearch/maskrcnn-benchmark/issues/267

        # NOTE Using BatchNorm2d for now, TBC if we need to change
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu2 = torch.nn.ReLU(inplace=True)

        if downsample:
            self.shortcut = torch.nn.Sequential(OrderedDict([
                ("conv", torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)),
                ("bn", torch.nn.BatchNorm2d(out_channels))
                ]))
        else:
            self.shortcut = torch.nn.Identity()

        self._add = None if op_add is None else op_add()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)

        if self._add is None:
            x += shortcut
        else:
            x = self._add(x, shortcut)
        x = self.relu2(x)

        return x


class Bottleneck(torch.nn.Module):
    """Bottleneck building block for big ResNets.

    Parameters
    ----------
    in_channels : int
        Number of input channels to this block.
    out_channels : int
        Number of output channels from this block.
    downsample : bool
        If true enables downsampling.
    op_add : callable, optional
        Shortcut (residual) addition operation.
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool, op_add: Optional[Callable] = None) -> None:
        super().__init__()
        # ResNet 1.0 downsamples on conv1
        # ResNet 1.5 downsamples on conv2
        # Implemented ResNet 1.5 as in torchvision

        # torchvision model uses FrozenBatchNorm2d from torchvision.ops.misc instead of BatchNorm2d
        # FrozenBatchNorm2d = BatchNorm2d where the batch statistics and the affine parameters are fixed
        # More info: https://github.com/facebookresearch/maskrcnn-benchmark/issues/267

        # NOTE Using BatchNorm2d for now, TBC if we need to change
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels//4)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels//4)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        self.relu3 = torch.nn.ReLU(inplace=True)

        # for the bottleneck a convolution is also needed on the shortcut in case the input shape does not match the output shape
        if downsample or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(OrderedDict([
                ("conv", torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, bias=False)),
                ("bn", torch.nn.BatchNorm2d(out_channels))
                ]))
        else:
            self.shortcut = torch.nn.Identity()

        self._add = None if op_add is None else op_add()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self._add is None:
            x += shortcut
        else:
            x = self._add(x, shortcut)
        x = self.relu3(x)

        return x


class ResNet(torch.nn.Module):
    """ResNet generic model class.

    Parameters
    ----------
    block : {BasicBlock, Bottleneck}
        Building block class.
    stacks : tuple of int
        Number of stacks per layer.
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    op_add : callable, optional
        Shortcut (residual) addition operation.

    Raises
    ------
    ValueError
        If `block` is not one of the two supported options.
    """

    # include average pooling and fully connected layers
    _include_top = True

    def __init__(self,
            block: Type[Union[BasicBlock, Bottleneck]],
            stacks: Tuple[int, int, int, int],
            channels: int = 3,
            num_classes: int = 1000,
            op_add: Optional[Callable] = None) -> None:
        super().__init__()
        # filters -> tuple with the shape at the output of each layer
        if block is BasicBlock:
            self.filters = (64, 64, 128, 256, 512)
        elif block is Bottleneck:
            self.filters = (64, 256, 512, 1024, 2048)
        else:
            raise ValueError(f"Expected BasicBlock or Bottleneck, but got {block}")

        # layer 0
        layer = 0
        self.layer0 = torch.nn.Sequential(OrderedDict([
            ("conv", torch.nn.Conv2d(channels, self.filters[layer], kernel_size=7, stride=2, padding=3, bias=False)),
            ("bn", torch.nn.BatchNorm2d(self.filters[layer])),
            ("relu", torch.nn.ReLU(inplace=True)),
            ("maxpool", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))
        # layers 1 to 4
        for layer in range(1, 5):
            setattr(self, f"layer{layer}", torch.nn.Sequential())
            # the first stack needs to be adapted to the filters of the previous layer
            getattr(self, f"layer{layer}").add_module(f"conv{layer+1}_1",
                    block(self.filters[layer-1], self.filters[layer], downsample=True if layer>1 else False, op_add=op_add))
            # remaining stacks have the number of filters of their layer
            for stack in range(1, stacks[layer-1]):
                getattr(self, f"layer{layer}").add_module(f"conv{layer+1}_{stack+1}",
                        block(self.filters[layer], self.filters[layer], downsample=False, op_add=op_add))

        if self._include_top:
            self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.flat = torch.nn.Flatten()
            self.fc = torch.nn.Linear(self.filters[4], num_classes)
            self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.sm(x)

        return x


class ResNetLayers(ResNet):
    """ResNet model class modified to output all the feature maps.

    The model returns the output of each residual block.
    It does not include the final pooling and fully connected layers.
    To be used as backbone (can be paired with a FPN).

    Parameters
    ----------
    block : {BasicBlock, Bottleneck}
        Building block class.
    stacks : tuple of int
        Number of stacks per layer.
    channels : int, optional
        Number of channels on the image.
    op_add : callable, optional
        Shortcut (residual) addition operation.
    device : torch.device, optional
        Device to load the model into.

    Raises
    ------
    ValueError
        If `block` is not one of the two supported options.
    TypeError
        If `device` arg is passed and not a torch.device.
    """

    # do not include average pooling and fully connected layers
    _include_top = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.layer0(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return (out1, out2, out3, out4)


class ResNet18(ResNet):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(BasicBlock, (2, 2, 2, 2), *args, **kwargs)


class ResNet34(ResNet):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(BasicBlock, (3, 4, 6, 3), *args, **kwargs)


class ResNet50(ResNet):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(Bottleneck, (3, 4, 6, 3), *args, **kwargs)


class ResNet101(ResNet):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(Bottleneck, (3, 4, 23, 3), *args, **kwargs)


class ResNet152(ResNet):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(Bottleneck, (3, 8, 36, 3), *args, **kwargs)


class ResNet18Layers(ResNetLayers):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(BasicBlock, (2, 2, 2, 2), *args, **kwargs)


class ResNet34Layers(ResNetLayers):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(BasicBlock, (3, 4, 6, 3), *args, **kwargs)


class ResNet50Layers(ResNetLayers):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(Bottleneck, (3, 4, 6, 3), *args, **kwargs)


class ResNet101Layers(ResNetLayers):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(Bottleneck, (3, 4, 23, 3), *args, **kwargs)


class ResNet152Layers(ResNetLayers):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(Bottleneck, (3, 8, 36, 3), *args, **kwargs)

