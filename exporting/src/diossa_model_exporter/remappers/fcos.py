from typing import override
import torch
import re

from diossa_model_exporter.remappers import BaseRemapper

class FCOSRemapper(BaseRemapper):
    """Remapper for FCOS model."""

    @override
    def __call__(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Remap the state dictionary."""
        remapped_state_dict = {}
        for key, val in state_dict.items():
            # Skip Vitis-AI fake-quantizer metadata
            if "quantizer" in key:
                continue

            new_key = key

            # Strip leading 'model.' if present
            if new_key.startswith("model."):
                new_key = new_key[6:]

            # Remap nested BN keys to match base model naming
            # 1. Stem layer (layer0)
            new_key = re.sub(r"^resnet\.layer0\.conv\.bn\.", "resnet.layer0.bn.", new_key)

            # 2. ResNet block inner conv BN (conv1.bn -> bn1, conv2.bn -> bn2, conv3.bn -> bn3)
            new_key = re.sub(r"\.conv([1-3])\.bn\.", r".bn\1.", new_key)

            # 3. ResNet shortcut BN
            new_key = re.sub(r"\.shortcut\.conv\.bn\.", ".shortcut.bn.", new_key)

            # 4. Subnet Norm layers (conv.bn -> gn)
            new_key = re.sub(r"^(cls_subnet|reg_subnet)\.layer(\d+)\.conv\.bn\.", r"\1.layer\2.gn.", new_key)
            remapped_state_dict[new_key] = val
        return remapped_state_dict
