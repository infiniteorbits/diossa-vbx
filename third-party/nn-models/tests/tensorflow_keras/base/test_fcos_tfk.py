import pytest
import tensorflow as tf
from packaging import version

from . import unit_tests_tfk as unit_tests
from nn_models.tensorflow_keras.object_detection import fcos


@pytest.fixture(scope="module")
def img_size():
    return (800, 800)

@pytest.fixture(scope="module")
def num_classes():
    return 2

@pytest.fixture(scope="function")
def full_model(img_size, channels, num_classes):
    tf.keras.backend.clear_session()
    mod = fcos.FCOS_Full(img_size, channels, num_classes)
    yield mod
    del mod

@pytest.fixture(scope="function")
def quant_model(img_size, channels, num_classes):
    tf.keras.backend.clear_session()
    mod = fcos.FCOS(img_size, channels, num_classes)
    yield mod
    del mod

@pytest.fixture(scope="function")
def noquant_model(img_size):
    tf.keras.backend.clear_session()
    mod = fcos.FCOS_Edge(img_size)
    yield mod
    del mod

@pytest.fixture(scope="module")
def name():
    return "FCOS"

@pytest.fixture(scope="class")
def ground_truth(batchsize, img_size):
    w, h = img_size
    gt_box = tf.constant([[w/4, h/4, w*3/4, h*3/4]], dtype=tf.float32)
    gt_labels = tf.ones(1, dtype=tf.int64)
    ground_truth = ([gt_box] * batchsize, [gt_labels] * batchsize)
    return ground_truth


def verify_interfaces(batchsize, img_size, channels, model, inputs, outputs):
    """Inputs and outputs shape and data type verification."""
    # verify correct model inputs
    assert isinstance(inputs, tuple)
    assert len(inputs) == 1
    assert isinstance(inputs[0], tf.Tensor)
    assert inputs[0].ndim == 4
    assert inputs[0].shape[0] == batchsize
    assert inputs[0].shape[1] == img_size[0]
    assert inputs[0].shape[2] == img_size[1]
    assert inputs[0].shape[-1] == channels
    assert inputs[0].dtype == tf.float32
    # verify expected model outputs
    assert isinstance(outputs, list)
    assert len(outputs) == 3
    ## boxes
    assert isinstance(outputs[0], tf.Tensor)
    assert outputs[0].ndim == 3
    assert outputs[0].shape[0] == batchsize
    assert outputs[0].shape[1] == model.detections_per_img
    assert outputs[0].shape[2] == 4
    assert outputs[0].dtype == tf.float32
    ## scores
    assert isinstance(outputs[1], tf.Tensor)
    assert outputs[1].ndim == 2
    assert outputs[1].shape[0] == batchsize
    assert outputs[1].shape[1] == model.detections_per_img
    assert outputs[1].dtype == tf.float32
    ## labels
    assert isinstance(outputs[2], tf.Tensor)
    assert outputs[2].ndim == 2
    assert outputs[2].shape[0] == batchsize
    assert outputs[2].shape[1] == model.detections_per_img
    assert outputs[2].dtype == tf.int64

def compute_train_step_loss(model, inputs, ground_truth):
    """Model dependent training step loss computation."""
    raw_outputs = model(*inputs)
    losses: dict = model.compute_loss(*raw_outputs, *ground_truth)
    loss = sum(losses.values()) / len(losses)
    return loss


class TestMultiple(unit_tests.TestMultiple):
    """Tests two batch and image sizes."""

    @pytest.fixture(scope="class", params=[(800, 800), (448, 448)])
    def img_size(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[2, 7])
    def num_classes(self, request):
        return request.param

    def test_run_basemodel(self, quant_model, inputs, name, debug, batchsize, img_size, channels):
        raw_outputs = unit_tests.run_float_inference(quant_model, inputs, name, debug)
        outputs = quant_model.post(*raw_outputs)
        verify_interfaces(batchsize, img_size, channels, quant_model, inputs, outputs)

    def test_run_float_inference(self, full_model, inputs, name, debug, batchsize, img_size, channels):
        outputs = unit_tests.run_float_inference(full_model, inputs, name, debug)
        verify_interfaces(batchsize, img_size, channels, full_model, inputs, outputs)

    @pytest.mark.xfail(reason="[Train] not implemented yet.")
    def test_run_train_step(self, quant_model, inputs, ground_truth, name):
        unit_tests.run_train_step(quant_model, lambda: compute_train_step_loss(quant_model, inputs, ground_truth), name)

class TestSingle(unit_tests.TestSingle):
    """Tests a single image batch and default image size."""

    def test_save_float_model(self, full_model, name, debug):
        unit_tests.save_float_model(full_model, name, debug)

    @pytest.mark.skipif(version.parse(tf.keras.__version__) >= version.parse("3.0"),
                        reason="[TFlite] currently only on old Keras.")
    def test_convert_tflite(self, quant_model, inputs, name):
        unit_tests.convert_tflite(quant_model, inputs, name)

    def test_export_onnx(self, quant_model, inputs, name, debug):
        unit_tests.export_onnx(quant_model, inputs, f"TF_{name}", debug)

