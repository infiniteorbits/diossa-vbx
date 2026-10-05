from typing import override

import torch

from diossa_model_exporter.remappers import BaseRemapper

class MobilePoseRemapper(BaseRemapper):
    """Remapper for MobilePose model."""

    @override
    def __call__(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remap the state dictionary."""
        remapped_state_dict = {}
        for key, val in state_dict.items():
            # Skip Vitis-AI fake-quantizer metadata
            if "quantizer" in key:
                continue
            # Strip leading 'model.' if present
            if key.startswith("model."):
                key = key[6:]
            # Remap nested BN keys to match base model naming
            key = key.replace("_conv.bn.", "_bn.")  # e.g., dw_conv.bn. -> dw_bn., pwl_conv.bn. -> pwl_bn.
            key = key.replace(".conv.bn.", ".bn.")  # e.g., layer0.conv.bn. -> layer0.bn.

            remapped_state_dict[key] = val
        return remapped_state_dict
