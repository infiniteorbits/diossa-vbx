from . import unit_tests_vaiq_tfk as unit_tests
from ..base.unit_tests_tfk import run_float_inference, export_onnx
from ..base.test_mobilepose_tfk import keypoints, full_model, quant_model, noquant_model, name, verify_interfaces
from nn_models.tensorflow_keras.nn.pixelshuffle import PixelShuffle


class TestVitisai(unit_tests.TestVitisai):
    """Tests Vitis-AI quantization."""

    def test_run_float_inference_hybrid(self, quant_model, noquant_model, inputs, name, batchsize, img_size, channels, keypoints):
        """Runs inference with both halves of the model sequentially."""

        intermediate_outputs = run_float_inference(quant_model, inputs, f"{name}_quant")
        outputs = run_float_inference(noquant_model, intermediate_outputs, f"{name}_noquant")
        verify_interfaces(batchsize, img_size, channels, keypoints, inputs, outputs)

    def test_export_onnx_partial(self, quant_model, noquant_model, inputs, name):
        """Exports to ONNX only the second half of the model."""

        # Instead of creating dummy inputs manually,
        # the outputs of the first half of the model are used.
        outputs = run_float_inference(quant_model, inputs, f"{name}_partial")
        export_onnx(noquant_model, (outputs,), f"TF_{name}_partial")

    def test_inspect_model(self, quant_model, debug):
        unit_tests.inspect_model(quant_model, {"PixelShuffle": PixelShuffle}, debug)

    def test_quantize_ptq_model(self, quant_model, inputs, name, debug):
        unit_tests.quantize_ptq_model(quant_model, inputs, f"TF_{name}_PTQ", {"PixelShuffle": PixelShuffle}, debug)

    def test_quantize_qat_model(self, quant_model, inputs, name, debug):
        unit_tests.quantize_qat_model(quant_model, inputs, f"TF_{name}_QAT", {"PixelShuffle": PixelShuffle}, debug)

