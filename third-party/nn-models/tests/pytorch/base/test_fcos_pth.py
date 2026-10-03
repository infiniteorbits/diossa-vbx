import pytest
import torch

from . import unit_tests_pth as unit_tests
from nn_models.pytorch.object_detection import fcos


@pytest.fixture(scope="module")
def img_size():
    return (800, 800)

@pytest.fixture(scope="module")
def num_classes():
    return 2

@pytest.fixture(scope="class")
def base_model(img_size, channels, num_classes, device):
    mod = fcos.FCOS(img_size, channels, num_classes)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="class")
def full_model(img_size, channels, num_classes, device):
    mod = fcos.FCOS_Full(img_size, channels, num_classes)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="class")
def edge_model(img_size, device):
    mod =  fcos.FCOS_Edge(img_size)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="module")
def name():
    return "FCOS"

@pytest.fixture(scope="class")
def ground_truth(batchsize, img_size, device):
    w, h = img_size
    gt_box = torch.tensor([[w/4, h/4, w*3/4, h*3/4]], dtype=torch.float32, device=device)
    gt_labels = torch.ones(1, dtype=torch.int64, device=device)
    ground_truth = ([gt_box] * batchsize, [gt_labels] * batchsize)
    return ground_truth


def verify_interfaces(batchsize, img_size, channels, model, inputs, outputs):
    """Inputs and outputs shape and data type verification."""
    # verify correct model inputs
    assert isinstance(inputs, tuple)
    assert len(inputs) == 1
    assert isinstance(inputs[0], torch.Tensor)
    assert inputs[0].ndim == 4
    assert inputs[0].shape[0] == batchsize
    assert inputs[0].shape[1] == channels
    assert inputs[0].shape[2] == img_size[0]
    assert inputs[0].shape[3] == img_size[1]
    assert inputs[0].dtype == torch.float32
    # verify expected model outputs
    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    ## boxes
    assert isinstance(outputs[0], torch.Tensor)
    assert outputs[0].ndim == 3
    assert outputs[0].shape[0] == batchsize
    assert outputs[0].shape[1] == model.detections_per_img
    assert outputs[0].shape[2] == 4
    assert outputs[0].dtype == torch.float32
    ## scores
    assert isinstance(outputs[1], torch.Tensor)
    assert outputs[1].ndim == 2
    assert outputs[1].shape[0] == batchsize
    assert outputs[1].shape[1] == model.detections_per_img
    assert outputs[1].dtype == torch.float32
    ## labels
    assert isinstance(outputs[2], torch.Tensor)
    assert outputs[2].ndim == 2
    assert outputs[2].shape[0] == batchsize
    assert outputs[2].shape[1] == model.detections_per_img
    assert outputs[2].dtype == torch.int64

def compute_train_step_loss(model, inputs, ground_truth):
    """Model dependent training step loss computation."""
    raw_outputs = model(*inputs)
    losses: dict = model.compute_loss(*raw_outputs, *ground_truth)
    loss = sum(losses.values()) / len(losses)
    return loss


class TestMultiple(unit_tests.TestMultiple):
    """Tests two batch sizes, image sizes and number of classes."""

    @pytest.fixture(scope="class", params=[(800, 800), (448, 448), (640, 480)])
    def img_size(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[2, 7])
    def num_classes(self, request):
        return request.param

    def test_run_basemodel(self, base_model, inputs, name, batchsize, img_size, channels):
        raw_outputs = unit_tests.run_float_inference(base_model, inputs, name)
        outputs = base_model.post(*raw_outputs)
        verify_interfaces(batchsize, img_size, channels, base_model, inputs, outputs)

    def test_run_float_inference(self, full_model, inputs, name, debug, batchsize, img_size, channels):
        outputs = unit_tests.run_float_inference(full_model, inputs, name, debug)
        verify_interfaces(batchsize, img_size, channels, full_model, inputs, outputs)

    def test_run_train_step(self, base_model, inputs, ground_truth, name):
        unit_tests.run_train_step(base_model, lambda: compute_train_step_loss(base_model, inputs, ground_truth), name)


class TestSingle:
    """Tests a single image batch and default image size."""

    def test_save_float_model(self, full_model, name):
        unit_tests.save_float_model(full_model, name)

    def test_export_onnx(self, full_model, inputs, name, debug):
        unit_tests.export_onnx(full_model, inputs, f"PT_{name}", debug)

    @pytest.mark.xfail(reason="[TorchScript] currently fails.")
    def test_compile_torchscript(self, full_model, name, debug):
        unit_tests.compile_torchscript(full_model, name, debug)

    def test_trace_model(self, full_model, inputs, name, debug):
        unit_tests.trace_model(full_model, inputs, name, debug)

