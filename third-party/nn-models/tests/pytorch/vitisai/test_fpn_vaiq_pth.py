import pytest

from . import unit_tests_vaiq_pth as unit_tests
from ..base.test_fpn_pth import img_size, model, name


class TestVitisai(unit_tests.TestVitisai):
    """Tests Vitis-AI quantization."""

    @pytest.mark.skip(reason="[QAT] not implemented for the backbone.")
    def test_quantize_qat_model(self, model, inputs, device, name, debug):
        super().test_quantize_qat_model(model, inputs, device, f"PT_{name}_QAT", debug)

