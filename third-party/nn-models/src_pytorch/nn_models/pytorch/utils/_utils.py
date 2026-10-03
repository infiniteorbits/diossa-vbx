from __future__ import annotations
from typing import Optional, List


def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class RegisterableModel:
    """
    Register a class automatically by inheriting
    """
    __registered_models = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        class_name = cls.__name__
        cls.__registered_models[class_name] = cls
        

    @classmethod
    def get_model_by_name(cls, name: str) -> RegisterableModel:
        """Gets a model class by name

        Args:
            name (str): model class name to retrieve 

        Returns:
            RegisterableModel: retrieved model class
        """
        return RegisterableModel.__registered_models[name]

    @classmethod
    def get_all_model_names(cls) -> List[str]:
        """Gets all registered model classes

        Returns:
            List[str]: All registered model classes
        """
        return list(RegisterableModel.__registered_models.keys())