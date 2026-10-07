"""Base class for remappers.

A remapper is a class that remaps the state dictionary of 
a PyTorch model to match the naming convention of the base model,
it is necessary during the loading of the model since some 
models were QAT-trained under Vitis-AI which changes the graph
"""

import torch

class BaseRemapper:
    """Base class for remappers."""

    def __call__(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remap the state dictionary."""
        raise NotImplementedError("Subclasses must implement this method")