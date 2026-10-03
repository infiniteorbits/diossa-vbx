import pytest
import torch

from . import unit_tests_pth as unit_tests
from nn_models.pytorch.keypoints_regression import mobilepose


@pytest.fixture(scope="module")
def keypoints():
    return 8

@pytest.fixture(scope="class")
def base_model(channels, keypoints, device):
    mod = mobilepose.MobilePose(channels, keypoints)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="class")
def full_model(channels, keypoints, device):
    mod = mobilepose.MobilePose_Full(channels, keypoints)
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="class")
def edge_model(device):
    mod = mobilepose.MobilePose_Edge()
    mod.to(device)
    yield mod
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="module")
def name():
    return "MobilePose"

@pytest.fixture(scope="class")
def ground_truth(batchsize, keypoints, device):    
    gt_coords = torch.randn([batchsize, keypoints, 2], dtype=torch.float32, device=device)
    gt_vis = torch.ones([batchsize, keypoints], dtype=torch.uint8, device=device)
    ground_truth = (gt_coords, gt_vis)
    return ground_truth


def verify_interfaces(batchsize, img_size, channels, keypoints, inputs, outputs):
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
    assert len(outputs) == 2
    ## coords
    assert isinstance(outputs[0], torch.Tensor)
    assert outputs[0].ndim == 3
    assert outputs[0].shape[0] == batchsize
    assert outputs[0].shape[1] == keypoints
    assert outputs[0].shape[2] == 2
    assert outputs[0].dtype == torch.float32
    ## heatmaps
    assert isinstance(outputs[1], torch.Tensor)
    assert outputs[1].ndim == 4
    assert outputs[1].shape[0] == batchsize
    assert outputs[1].shape[1] == keypoints
    assert outputs[1].shape[2] == inputs[0].shape[2] / 4
    assert outputs[1].shape[3] == inputs[0].shape[3] / 4
    assert outputs[1].dtype == torch.float32

def compute_train_step_loss(model, inputs, ground_truth):
    """Model dependent training step loss computation."""
    raw_outputs = model(*inputs)
    outputs = model.post(raw_outputs)
    losses: dict = model.compute_loss(*outputs, *ground_truth)
    loss = torch.mean(sum(losses.values()))
    return loss


class TestMultiple(unit_tests.TestMultiple):
    """Tests two batch and image sizes."""

    @pytest.fixture(scope="class", params=[(224, 224), (256, 256)])
    def img_size(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[8, 26])
    def keypoints(self, request):
        return request.param

    def test_run_basemodel(self, base_model, inputs, name, batchsize, img_size, channels, keypoints):
        raw_outputs = unit_tests.run_float_inference(base_model, inputs, name)
        outputs = base_model.post(raw_outputs)
        verify_interfaces(batchsize, img_size, channels, keypoints, inputs, outputs)

    def test_run_float_inference(self, full_model, inputs, name, debug, batchsize, img_size, channels, keypoints):
        outputs = unit_tests.run_float_inference(full_model, inputs, name, debug)
        verify_interfaces(batchsize, img_size, channels, keypoints, inputs, outputs)

    def test_run_train_step(self, base_model, inputs, ground_truth, name):
        unit_tests.run_train_step(base_model, lambda: compute_train_step_loss(base_model, inputs, ground_truth), name)


class TestSingle(unit_tests.TestSingle):
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

