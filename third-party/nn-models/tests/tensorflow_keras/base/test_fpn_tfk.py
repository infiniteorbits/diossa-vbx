import pytest
import tensorflow as tf

from . import unit_tests_tfk as unit_tests
from nn_models.tensorflow_keras import fpn, resnet


@pytest.fixture(scope="module")
def img_size():
    return (800, 800)

@pytest.fixture(scope="class")
def model(img_size, channels):
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(*img_size, channels), name="img")
    backbone = resnet.resnet50(inputs, output_layers=True)
    body = fpn.fpn(backbone)
    mod = tf.keras.Model(inputs, body, name="ResNet50withFPN")
    yield mod
    del mod


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
        assert isinstance(inputs[0], tf.Tensor)
        assert inputs[0].ndim == 4
        assert inputs[0].shape[0] == batchsize
        assert inputs[0].shape[1] == img_size[0]
        assert inputs[0].shape[2] == img_size[1]
        assert inputs[0].shape[-1] == channels
        assert inputs[0].dtype == tf.float32
        # verify expected model outputs
        strides = (4, 8, 16, 32, 64)
        assert isinstance(outputs, list)
        assert len(outputs) == 5
        for idx in range(5):
            assert isinstance(outputs[idx], tf.Tensor)
            assert outputs[idx].ndim == 4
            assert outputs[idx].shape[0] == batchsize
            assert outputs[idx].shape[-1] == 256
            assert outputs[idx].shape[1] == -(-inputs[0].shape[1] // strides[idx])
            assert outputs[idx].shape[2] == -(-inputs[0].shape[2] // strides[idx])
            assert outputs[idx].dtype == tf.float32


class TestSingle(unit_tests.TestSingle):
    """Tests a single image batch and default image size."""

    pass

