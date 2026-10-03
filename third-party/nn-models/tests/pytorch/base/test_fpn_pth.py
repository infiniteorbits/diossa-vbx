import pytest
import torch

from . import unit_tests_pth as unit_tests
from nn_models.pytorch import fpn, resnet


@pytest.fixture(scope="module")
def img_size():
    return (800, 800)

@pytest.fixture(scope="class")
def model(device, channels):
    backbone = resnet.ResNet50Layers(channels)
    body = fpn.FPN(backbone.filters)
    mod = torch.nn.Sequential(backbone, body)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="module")
def name():
    return "ResNet50withFPN"


class TestMultiple(unit_tests.TestMultiple):
    """Tests two batch and image sizes."""

    @pytest.fixture(scope="class", params=[(800, 800), (448, 448)])
    def img_size(self, request):
        return request.param

    def test_run_float_inference(self, model, inputs, name, debug, batchsize, img_size, channels):
        outputs = unit_tests.run_float_inference(model, inputs, name, debug)

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
        strides = (8, 16, 32)
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3
        for idx in range(3):
            assert isinstance(outputs[idx], torch.Tensor)
            assert outputs[idx].ndim == 4
            assert outputs[idx].shape[0] == batchsize
            assert outputs[idx].shape[1] == 256
            assert outputs[idx].shape[2] == -(-inputs[0].shape[2] // strides[idx])
            assert outputs[idx].shape[3] == -(-inputs[0].shape[3] // strides[idx])
            assert outputs[idx].dtype == torch.float32


class TestSingle(unit_tests.TestSingle):
    """Tests a single image batch and default image size."""

    pass

