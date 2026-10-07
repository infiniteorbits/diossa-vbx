import pytest
from tensorflow import keras

from . import unit_tests_vaiq_tfk as unit_tests
from ..base.unit_tests_tfk import run_float_inference, save_float_model, convert_tflite, export_onnx


# (first and last) operations from a ResNet
conv = keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False, name="conv1")
bn = keras.layers.BatchNormalization(name="bn1")
relu = keras.layers.ReLU(name="relu1")
maxpool = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="maxpool")
avgpool = keras.layers.GlobalAveragePooling2D(name="avgpool")
fc = keras.layers.Dense(8, activation="softmax", name="fc")

# Functional API with Model subclassing
class Functional(keras.Model):
    def __init__(self, inp, name):
        x = conv(inp)
        x = bn(x)
        x = relu(x)
        x = maxpool(x)
        x = avgpool(x)
        out = fc(x)
        super().__init__(inputs=inp, outputs=out, name=name)

# Model subclassing
class Model(keras.Model):
    def __init__(self, name):
        super().__init__(name=name)
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.maxpool = maxpool
        self.avgpool = avgpool
        self.fc = fc

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        out = self.fc(x)
        return out

# Layer subclassing
class Layer(keras.layers.Layer):
    def __init__(self, name):
        super().__init__(name=name)
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.maxpool = maxpool
        self.avgpool = avgpool
        self.fc = fc

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        out = self.fc(x)
        return out

@pytest.fixture(scope="module", autouse=True)
def skiptest(vitis_ai):
    if not vitis_ai: pytest.skip("Vitis-AI API testing disabled.")

@pytest.fixture(scope="module")
def sequential_model(img_size):
    model = keras.Sequential([conv, bn, relu, maxpool, avgpool, fc], name="sequential")
    return model

@pytest.fixture(scope="module")
def functional_model(img_size):
    inp = keras.Input(shape=(*img_size, 3), name="img")
    x = conv(inp)
    x = bn(x)
    x = relu(x)
    x = maxpool(x)
    x = avgpool(x)
    out = fc(x)
    model = keras.Model(inp, out, name="functional")
    return model

@pytest.fixture(scope="module")
def nested_sequential_model(img_size):
    inp = keras.Input(shape=(*img_size, 3), name="img")
    first = keras.Sequential([conv, bn, relu, maxpool], name="first")
    last = keras.Sequential([avgpool, fc], name="last")
    model = keras.Model(inp, last(first(inp)), name="nested_sequential")
    return model

@pytest.fixture(scope="module")
def nested_functional_model(img_size):
    inp = keras.Input(shape=(*img_size, 3), name="img")
    x = conv(inp)
    x = bn(x)
    x = relu(x)
    mid = maxpool(x)
    first = keras.Model(inp, mid, name="first")
    x = avgpool(mid)
    out = fc(x)
    last = keras.Model(mid, out, name="last")
    model = keras.Model(inp, last(first(inp)), name="nested_functional")
    return model

@pytest.fixture(scope="module")
def subclassing_functional(img_size):
    inp = keras.Input(shape=(*img_size, 3), name="img")
    model = Functional(inp, "subclassing_functional")
    return model

@pytest.fixture(scope="module")
def subclassing_model(img_size):
    model = Model("subclassing_model")
    return model

@pytest.fixture(scope="module")
def subclassing_layer(img_size):
    inp = keras.Input(shape=(*img_size, 3), name="img")
    layer = Layer("layer")
    model = keras.Model(inp, layer(inp), name="subclassing_layer")
    return model

@pytest.fixture(scope="module", params=["sequential_model",
                                        "functional_model",
                                        "nested_sequential_model",
                                        "nested_functional_model",
                                        "subclassing_functional",
                                        "subclassing_model",
                                        "subclassing_layer"])
def model(request):
    mod = request.getfixturevalue(request.param)
    yield mod
    del mod
    keras.backend.clear_session()


def test_run_float_inference(model, inputs, name, debug):
    run_float_inference(model, inputs, name, debug)

def test_save_float_model(model, name, debug):
    save_float_model(model, name, debug)

def test_convert_tflite(model, inputs, name):
    convert_tflite(model, inputs, name)

def test_export_onnx(model, inputs, name, debug):
    export_onnx(model, inputs, name, debug)

@pytest.mark.xfail
def test_inspect_model(model, debug):
    if model.name == "subclassing_model":
        custom_layers = {"Model": Model}
    elif model.name == "subclassing_layer":
        custom_layers = {"Layer": Layer}
    else:
        custom_layers = {}
    unit_tests.inspect_model(model, custom_layers, debug)

@pytest.mark.xfail
def test_quantize_ptq_model(model, inputs, name, debug):
    if model.name == "subclassing_model":
        custom_layers = {"Model": Model}
    elif model.name == "subclassing_layer":
        custom_layers = {"Layer": Layer}
    else:
        custom_layers = {}
    unit_tests.quantize_ptq_model(model, inputs, name, custom_layers, debug)

@pytest.mark.xfail
def test_quantize_qat_model(model, inputs, name, debug):
    if model.name == "subclassing_model":
        custom_layers = {"Model": Model}
    elif model.name == "subclassing_layer":
        custom_layers = {"Layer": Layer}
    else:
        custom_layers = {}
    unit_tests.quantize_qat_model(model, inputs, name, custom_layers, debug)

