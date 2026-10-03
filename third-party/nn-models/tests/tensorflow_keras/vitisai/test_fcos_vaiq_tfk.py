import pytest

from . import unit_tests_vaiq_tfk as unit_tests
from ..base.unit_tests_tfk import run_float_inference, export_onnx
from ..base.test_fcos_tfk import img_size, num_classes, full_model, quant_model, noquant_model, name, verify_interfaces


class TestVitisai(unit_tests.TestVitisai):
    """Tests Vitis-AI quantization."""

    def test_run_float_inference_hybrid(self, quant_model, noquant_model, inputs, name, batchsize, img_size, channels):
        """Runs inference with both halves of the model sequentially."""

        intermediate_outputs = run_float_inference(quant_model, inputs, f"{name}_quant")
        outputs = run_float_inference(noquant_model, intermediate_outputs, f"{name}_noquant")
        verify_interfaces(batchsize, img_size, channels, noquant_model, inputs, outputs)

    @pytest.mark.xfail(reason="[ONNX] partial model currently fails.")
    def test_export_onnx_partial(self, quant_model, noquant_model, inputs, name):
        """Exports to ONNX only the second half of the model."""

        # Instead of creating dummy inputs manually,
        # the outputs of the first half of the model are used.
        outputs = run_float_inference(quant_model, inputs, f"{name}_partial")
        flat_outputs = [tensor for sequence in outputs for tensor in sequence]
        export_onnx(noquant_model, flat_outputs, f"TF_{name}_partial")

    def test_inspect_model(self, quant_model, debug):
        unit_tests.inspect_model(quant_model, debug=debug)

    def test_quantize_ptq_model(self, quant_model, inputs, name, debug):
        unit_tests.quantize_ptq_model(quant_model, inputs, f"TF_{name}_PTQ", debug=debug)

    def test_quantize_qat_model(self, quant_model, inputs, name, debug):
        unit_tests.quantize_qat_model(quant_model, inputs, f"TF_{name}_QAT", debug=debug)

