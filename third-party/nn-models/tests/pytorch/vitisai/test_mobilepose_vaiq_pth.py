import pytest
import torch

from . import unit_tests_vaiq_pth as unit_tests
from ..base.unit_tests_pth import run_float_inference, export_onnx
from ..base.test_mobilepose_pth import keypoints, full_model, edge_model, name, ground_truth, verify_interfaces, compute_train_step_loss
from nn_models.pytorch.edge.vitis_ai.mobilepose import MobilePose_PTQ, MobilePose_QAT


@pytest.fixture(scope="class")
def ptq_model(channels, keypoints, device):
    mod = MobilePose_PTQ(channels, keypoints)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="class")
def qat_model(channels, keypoints, device):
    mod = MobilePose_QAT(channels, keypoints)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestVitisai(unit_tests.TestVitisai):
    """Tests Vitis-AI quantization."""

    def test_run_float_inference_hybrid(self, ptq_model, edge_model, inputs, name, batchsize, img_size, channels, keypoints):
        """Runs inference with both halves of the model sequentially."""

        intermediate_outputs = run_float_inference(ptq_model, inputs, f"{name}_quant")
        outputs = run_float_inference(edge_model, (intermediate_outputs, ), f"{name}_noquant")
        verify_interfaces(batchsize, img_size, channels, keypoints, inputs, outputs)

    def test_export_onnx_partial(self, ptq_model, edge_model, inputs, name):
        """Exports to ONNX only the second half of the model."""

        # Instead of creating dummy inputs manually,
        # the outputs of the first half of the model are used.
        outputs = run_float_inference(ptq_model, inputs, f"{name}_partial")
        export_onnx(edge_model, (outputs, ), f"PT_{name}_partial")


    def test_inspect_model(self, ptq_model, inputs, device, debug):
        unit_tests.inspect_model(ptq_model, inputs, device, debug)

    def test_quantize_ptq_model(self, ptq_model, inputs, device, name, debug):
        unit_tests.quantize_ptq_model(ptq_model, inputs, device, f"PT_{name}_PTQ", debug)
    
    def test_quantize_qat_model(self, qat_model, inputs, ground_truth, device, name, debug):
        unit_tests.quantize_qat_model(qat_model, inputs, device, f"PT_{name}_QAT",
                                      lambda: compute_train_step_loss(qat_model, inputs, ground_truth),
                                      False, debug)

