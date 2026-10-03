import pytest
import tensorflow as tf

from . import unit_tests_tfk as unit_tests
from nn_models.tensorflow_keras.mobilenet import MobileNetV2


@pytest.fixture(scope="module")
def num_classes():
    return 100

@pytest.fixture(scope="module")
def width_multiplier():
    return 1.0

@pytest.fixture(scope="class")
def model(img_size, channels, num_classes, width_multiplier):
    tf.keras.backend.clear_session()
    mod = MobileNetV2(img_size, channels, num_classes, width_multiplier)
    yield mod
    del mod


class TestMultiple(unit_tests.TestMultiple):
    """Tests two batch and image sizes."""

    @pytest.fixture(scope="class", params=[(224, 224), (256, 256)])
    def img_size(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[100, 7])
    def num_classes(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1.0, 1.3])
    def width_multiplier(self, request):
        return request.param

    def test_run_float_inference(self, model, inputs, name, debug, batchsize, img_size, channels, num_classes):
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
        assert isinstance(outputs, tf.Tensor)
        assert outputs.ndim == 2
        assert outputs.shape[0] == batchsize
        assert outputs.shape[1] == num_classes
        assert outputs.dtype == tf.float32


class TestSingle(unit_tests.TestSingle):
    """Tests a single image batch and default image size."""

    pass

