import gc
import pytest
import tensorflow as tf


@pytest.fixture(scope="package")
def batchsize():
    return 1

@pytest.fixture(scope="package")
def img_size():
    return (224, 224)

@pytest.fixture(scope="package")
def channels():
    return 3

@pytest.fixture(scope="class")
def inputs(batchsize, img_size, channels):
    tf.keras.backend.clear_session()
    inps = (tf.random.normal((batchsize, *img_size, channels)), )
    yield inps
    del inps

@pytest.fixture(scope="class")
def name(model: tf.keras.Model):
    return model.name

@pytest.fixture(scope="function", autouse=True)
def garbage_collect():
    yield
    gc.collect()
    tf.keras.backend.clear_session()

