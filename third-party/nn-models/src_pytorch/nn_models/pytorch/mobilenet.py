"""PyTorch MobileNet module.

Model implemented using torch.nn.Module subclassing and Sequential containers.

Notes
-----
The implementation follows the original paper [1]_.

References
----------
.. [1] "MobileNetV2: Inverted Residuals and Linear Bottlenecks", https://arxiv.org/abs/1801.04381

"""
from typing import Optional, Tuple, Callable
from collections import OrderedDict

import torch

from .utils._utils import _make_divisible


class InvertedResidual(torch.nn.Module):
    """Inverted residual building block for MobileNets.

    Parameters
    ----------
    in_channels : int
        Number of input channels to this block.
    out_channels : int
        Number of output channels from this block.
    stride : int
        Stride of the depthwise convolution.
    expansion_ratio : int
        Ratio between the input channels and the inner channels.
    op_add : callable, optional
        Shortcut (residual) addition operation.

    Raises
    ------
    ValueError
        If `stride` arg is not in (1, 2).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expansion_ratio: int,
                 op_add: Optional[Callable] = None
                ) -> None:
        super().__init__()

        if stride not in (1, 2):
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(in_channels * expansion_ratio))

        if expansion_ratio != 1:
            # pw (pointwise)
            self.pw_conv = torch.nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.pw_bn = torch.nn.BatchNorm2d(hidden_dim)
            self.pw_relu = torch.nn.ReLU6(inplace=True)
        else:
            self.pw_conv = None
        # dw (depthwise)
        self.dw_conv = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.dw_bn = torch.nn.BatchNorm2d(hidden_dim)
        self.dw_relu = torch.nn.ReLU6(inplace=True)
        # pw-linear (pointwise)
        self.pwl_conv = torch.nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.pwl_bn = torch.nn.BatchNorm2d(out_channels)

        self._add = None if op_add is None else op_add()

        if stride == 1 and in_channels == out_channels:
            self._shortcut = True
        else:
            self._shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        if self.pw_conv is not None:
            x = self.pw_conv(x)
            x = self.pw_bn(x)
            x = self.pw_relu(x)
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_relu(x)
        x = self.pwl_conv(x)
        x = self.pwl_bn(x)

        if self._shortcut is not None:
            if self._add is None:
                return x + shortcut
            else:
                return self._add(x, shortcut)
        else:
            return x


class MobileNetV2(torch.nn.Module):
    """MobileNet (version 2) model class.

    Parameters
    ----------
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    width_multiplier : float, optional
        Multiplier applied to the number of channels of each layer.
    op_add : callable, optional
        Shortcut (residual) addition operation.

    Attributes
    ----------
    dropout : float, default=0.2
        Probability of an element to be zeroed.
    """

    dropout: float = 0.2
    # include average pooling and fully connected layers
    _include_top = True

    def __init__(self,
                 channels: int = 3,
                 num_classes: int = 1000,
                 width_multiplier: float = 1.0,
                 op_add: Optional[Callable] = None) -> None:
        super().__init__()

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
        in_channels = _make_divisible(c * width_multiplier)
        self.layer0 = torch.nn.Sequential(OrderedDict([
            ("conv", torch.nn.Conv2d(channels, in_channels, kernel_size=3, stride=s, padding=1, bias=False)),
            ("bn", torch.nn.BatchNorm2d(in_channels)),
            ("relu", torch.nn.ReLU6(inplace=True))
            ]))
        # inverted residual blocks
        for layer in range(1, len(inverted_residual_setting) - 1):
            (t, c, n, s) = inverted_residual_setting[layer]
            out_channels = _make_divisible(c * width_multiplier)
            stacks = []
            for stack in range(n):
                stride = s if stack == 0 else 1
                stacks.append(InvertedResidual(in_channels, out_channels, stride, expansion_ratio=t, op_add=op_add))
                in_channels = out_channels
            setattr(self, f"layer{layer}", torch.nn.Sequential(*stacks))
        # last layer
        layer = len(inverted_residual_setting) - 1
        (t, c, n, s) = inverted_residual_setting[layer]
        self.last_channels = _make_divisible(c * max(1.0, width_multiplier))
        setattr(self, f"layer{layer}", torch.nn.Sequential(OrderedDict([
                ("conv", torch.nn.Conv2d(in_channels, self.last_channels, kernel_size=1, stride=s, bias=False)),
                ("bn", torch.nn.BatchNorm2d(self.last_channels)),
                ("relu", torch.nn.ReLU6(inplace=True))
                ]))
            )

        if self._include_top:
            self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.flat = torch.nn.Flatten()
            self.drop = torch.nn.Dropout(p=self.dropout)
            self.fc = torch.nn.Linear(self.last_channels, num_classes)
            self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.layer8(x)

        x = self.avgpool(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.fc(x)
        x = self.sm(x)

        return x


class MobileNetV2Layers(MobileNetV2):
    """MobileNet (version 2) model class modified to output all the feature maps.

    The model returns the output of each inverted residual block.
    It does not include the final pooling and fully connected layers.
    To be used as backbone (can be paired with a FPN).

    Parameters
    ----------
    channels : int, optional
        Number of channels on the image.
    width_multiplier : float, optional
        Multiplier applied to the number of channels of each layer.
    op_add : callable, optional
        Shortcut (residual) addition operation.
    device : torch.device, optional
        Device to load the model into.

    Attributes
    ----------
    dropout : float, default=0.2
        Probability of an element to be zeroed.

    Raises
    ------
    TypeError
        If `device` arg is passed and not a torch.device.
    """

    # do not include average pooling and fully connected layers
    _include_top = False

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.layer0(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)

        out8 = self.layer8(out7)

        return (out1, out2, out3, out4, out5, out6, out7, out8)

