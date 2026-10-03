import pytest
import torch

from . import unit_tests_vaiq_pth as unit_tests
from ..base.unit_tests_pth import run_float_inference, export_onnx
from ..base.test_fasterrcnn_pth import img_size, num_classes, full_model, edge_model, name, verify_interfaces
from nn_models.pytorch.edge.vitis_ai.faster_rcnn import FasterRCNN_PTQ, FasterRCNN_QAT


@pytest.fixture(scope="class")
def ptq_model(img_size, channels, num_classes, device):
    mod =  FasterRCNN_PTQ(img_size, channels, num_classes)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="class")
def qat_model(img_size, channels, num_classes, device):
    mod =  FasterRCNN_QAT(img_size, channels, num_classes)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestVitisai(unit_tests.TestVitisai):
    """Tests Vitis-AI quantization."""

    def test_run_float_inference_hybrid(self, ptq_model, edge_model, inputs, name, batchsize, img_size, channels):
        """Runs inference with both halves of the model sequentially."""

        intermediate_outputs = run_float_inference(ptq_model, inputs, f"{name}_quant")
        outputs = run_float_inference(edge_model, intermediate_outputs, f"{name}_noquant")
        verify_interfaces(batchsize, img_size, channels, edge_model, inputs, outputs)

    def test_export_onnx_partial(self, ptq_model, edge_model, inputs, name):
        """Exports to ONNX only the second half of the model."""

        # Instead of creating dummy inputs manually,
        # the outputs of the first half of the model are used.
        outputs = run_float_inference(ptq_model, inputs, f"{name}_partial")
        export_onnx(edge_model, outputs, f"PT_{name}_partial")

    def test_inspect_model(self, full_model, inputs, device, debug):
        unit_tests.inspect_model(full_model, inputs, device, debug)

    def test_quantize_ptq_model(self, ptq_model, inputs, device, name, debug):
        unit_tests.quantize_ptq_model(ptq_model, inputs, device, f"PT_{name}_PTQ", debug)

    def test_quantize_qat_model(self, qat_model, inputs, device, name, debug):
        unit_tests.quantize_qat_model(qat_model, inputs, device, f"PT_{name}_QAT", None, True, debug)

