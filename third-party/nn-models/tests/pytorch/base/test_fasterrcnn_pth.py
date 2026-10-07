import pytest
import torch
from packaging import version

from . import unit_tests_pth as unit_tests
from nn_models.pytorch.object_detection import faster_rcnn


@pytest.fixture(scope="module")
def img_size():
    return (800, 800)

@pytest.fixture(scope="module")
def num_classes():
    return 2

@pytest.fixture(scope="class")
def full_model(img_size, channels, num_classes, device):
    mod = faster_rcnn.FasterRCNN(img_size, channels, num_classes)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="class")
def edge_model(img_size, channels, num_classes, device):
    mod =  faster_rcnn.FasterRCNN_Edge(img_size, channels, num_classes)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="module")
def name():
    return "Faster_R-CNN"


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
    assert outputs[0].shape[1] == model.detector.detections_per_img
    assert outputs[0].shape[2] == 4
    assert outputs[0].dtype == torch.float32
    ## scores
    assert isinstance(outputs[1], torch.Tensor)
    assert outputs[1].ndim == 2
    assert outputs[1].shape[0] == batchsize
    assert outputs[1].shape[1] == model.detector.detections_per_img
    assert outputs[1].dtype == torch.float32
    ## labels
    assert isinstance(outputs[2], torch.Tensor)
    assert outputs[2].ndim == 2
    assert outputs[2].shape[0] == batchsize
    assert outputs[2].shape[1] == model.detector.detections_per_img
    assert outputs[2].dtype == torch.int64


class TestMultiple(unit_tests.TestMultiple):
    """Tests two batch and image sizes."""

    @pytest.fixture(scope="class", params=[(800, 800), (448, 448)])
    def img_size(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[2, 7])
    def num_classes(self, request):
        return request.param

    def test_run_float_inference(self, full_model, inputs, name, debug, batchsize, img_size, channels):
        outputs = unit_tests.run_float_inference(full_model, inputs, name, debug)
        verify_interfaces(batchsize, img_size, channels, full_model, inputs, outputs)


class TestSingle:
    """Tests a single image batch and default image size."""

    def test_save_float_model(self, full_model, name):
        unit_tests.save_float_model(full_model, name)

    @pytest.mark.skipif(version.parse(torch.__version__) >= version.parse("2.0"),
                        reason="[ONNX] currently only on old PyTorch.")
    def test_export_onnx(self, full_model, inputs, name, debug):
        unit_tests.export_onnx(full_model, inputs, f"PT_{name}", debug)

    @pytest.mark.xfail(reason="[TorchScript] currently fails.")
    def test_compile_torchscript(self, full_model, name, debug):
        unit_tests.compile_torchscript(full_model, name, debug)

    def test_trace_model(self, full_model, inputs, name, debug):
        unit_tests.trace_model(full_model, inputs, name, debug)

