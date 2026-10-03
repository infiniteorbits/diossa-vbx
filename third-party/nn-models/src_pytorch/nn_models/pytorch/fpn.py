"""PyTorch Feature Pyramid Network (FPN) module.

Model implemented using torch.nn.Module subclassing.

Notes
-----
The implementation follows the original paper [1]_.

References
----------
.. [1] "Feature Pyramid Networks for Object Detection", https://arxiv.org/abs/1612.03144

"""

import torch
from typing import Tuple, Optional, Callable


class Interpolate(torch.nn.Module):
    """Package interpolate operation in a torch.nn.Module"""
    def forward(self, x: torch.Tensor):
        return torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")


class BuildingBlock(torch.nn.Module):
    """Building block for FPNs.

    Parameters
    ----------
    in_channels : int
        Number of input channels to this block.
    out_channels : int
        Number of output channels from this block.
    top : bool, optional
        To select if there is a top branch (True) or not (False).
    op_add : callable, optional
        Shortcut (residual) addition operation.
    """

    def __init__(self, in_channels: int, out_channels: int = 256, top: bool = True,
                 op_add: Optional[Callable] = None) -> None:
        super().__init__()
        self.conv_in = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        if top:
            self.upscale = Interpolate()
        else:
            self.upscale = None
        self.conv_out = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self._add = None if op_add is None else op_add()

    def forward(self, top: Optional[torch.Tensor], lat: torch.Tensor) \
                -> Tuple[torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Parameters
        ----------
        top : torch.Tensor, optional
            Tensor from the upper layer.
        lat : torch.Tensor
            Tensor from the lateral layer (previous model).

        Returns
        -------
        bot : tensor
            Internal output from the current layer to be fed to the next layer.
        out : tensor
            External output from the current model layer to the next model.
        """
        lat = self.conv_in(lat)
        if self.upscale is not None and top is not None:
            top = self.upscale(top)
            if self._add is None:
                bot = top + lat
            else:
                bot = self._add(top, lat)
        else:
            bot = lat
        out = self.conv_out(bot)
        return (bot, out)


class FPN(torch.nn.Module):
    """Feature Pyramid Network generic model class.

    Parameters
    ----------
    filters : tuple of int
        Number of channels of each feature map.
    op_add : callable, optional
        Shortcut (residual) addition operation.

    Attributes
    ----------
    channels : int
        Number of channels for all pyramid levels.
    strides : tuple of int
        Stride of each layer output with respect to the input image.
    """

    channels: int = 256
    strides: Tuple[int, int, int] = (8, 16, 32)

    _levels = 3 # pyramid levels

    def __init__(self, filters: Tuple[int, int, int, int, int],
                 op_add: Optional[Callable] = None) -> None:
        super().__init__()

        for layer in range(self._levels, 0, -1):
            setattr(self,
                    f"layer{layer}",
                    BuildingBlock(filters[layer - self._levels + 4], self.channels, op_add=op_add))

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) \
                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Parameters
        ----------
        inputs : tuple of torch.Tensor
            Inputs to the model, from the previous model layers.

        Returns
        -------
        tuple of torch.Tensor
            Outputs of all the model layers.
        """
        (c2, c3, c4, c5) = inputs

        (x, p5) = self.layer3(None, c5)
        (x, p4) = self.layer2(x, c4)
        (x, p3) = self.layer1(x, c3)

        return (p3, p4, p5)


class FPN_P2toP6MaxPool(FPN):
    """Feature Pyramid Network model class, with P2 and top P6 MaxPool layer.

    Parameters
    ----------
    filters : tuple of int
        Number of channels of each feature map.
    op_add : callable, optional
        Shortcut (residual) addition operation.

    Attributes
    ----------
    channels : int
        Number of channels for all pyramid levels.
    strides : tuple of int
        Stride of each layer output with respect to the input image.
    """

    strides: Tuple[int, int, int, int, int] = (4, 8, 16, 32, 64)

    _levels = 4 # pyramid levels

    def __init__(self, filters: Tuple[int, int, int, int, int],
                 op_add: Optional[Callable] = None) -> None:
        super().__init__(filters, op_add)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) \
                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Parameters
        ----------
        inputs : tuple of torch.Tensor
            Inputs to the model, from the previous model layers.

        Returns
        -------
        tuple of torch.Tensor
            Outputs of all the model layers.
        """
        (c2, c3, c4, c5) = inputs

        (x, p5) = self.layer4(None, c5)
        (x, p4) = self.layer3(x, c4)
        (x, p3) = self.layer2(x, c3)
        (x, p2) = self.layer1(x, c2)

        p6 = self.maxpool(p5)

        return (p2, p3, p4, p5, p6)


class FPN_P3toP6P7Conv(FPN):
    """Feature Pyramid Network model class, with no P2 and top P6 and P7 Conv layers.

    Parameters
    ----------
    filters : tuple of int
        Number of channels of each feature map.
    op_add : callable, optional
        Shortcut (residual) addition operation.

    Attributes
    ----------
    channels : int
        Number of channels for all pyramid levels.
    strides : tuple of int
        Stride of each layer output with respect to the input image.
    """

    strides: Tuple[int, int, int, int, int] = (8, 16, 32, 64, 128)

    def __init__(self, filters: Tuple[int, int, int, int, int],
                 op_add: Optional[Callable] = None) -> None:
        super().__init__(filters, op_add)

        self.conv_p6 = torch.nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv_p7 = torch.nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) \
                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Parameters
        ----------
        inputs : tuple of torch.Tensor
            Inputs to the model, from the previous model layers.

        Returns
        -------
        tuple of torch.Tensor
            Outputs of all the model layers.
        """
        (c2, c3, c4, c5) = inputs

        (x, p5) = self.layer3(None, c5)
        (x, p4) = self.layer2(x, c4)
        (x, p3) = self.layer1(x, c3)

        p6 = self.conv_p6(p5)
        # NOTE instead of coding p6 as input to p7,
        # the same self.conv_p6(p5) is fed to p7 calculation,
        # which makes the model use a shared layer.
        # This makes the compiled model more robust,
        # because the compiler does not need
        # to duplicate the convolutions, which does not always work,
        # while having at worst just minimal speed impact on the float model.
        p7 = self.conv_p7(self.relu(self.conv_p6(p5)))

        return (p3, p4, p5, p6, p7)

