"""Keras NNs tests module.

Tests:
    Float model (inference and saving)
    TensorFlow Lite (conversion and saving)
    ONNX export and inference
"""

import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnxruntime
from timeit import timeit
from typing import Sequence, Callable
import pytest


def run_float_inference(model: keras.Model, inputs: Sequence[tf.Tensor],
                        name: str, debug: bool = False) -> None:
    """Runs the inference on a NN model.

    Parameters
    ----------
    model : keras.Model
        Model to be inferred.
    inputs : sequence of tf.Tensor
        Tensors to be fed to the model.
    name : str
        Name of the model.
    debug : bool, optional
        If true prints the model summary.
    """
    outputs = model(inputs, training=False)
    if debug: model.summary()
    return outputs

def run_train_step(model: keras.Model,
                   compute_train_step_loss: Callable[[], tf.Tensor],
                   name: str) -> None:
    """Runs a training step on a NN model.

    Parameters
    ----------
    model : keras.Model
        Model to be trained.
    compute_train_step_loss : callable
        Function to compute the training step loss.
    name : str
        Name of the model.
    """
    optim = tf.keras.optimizers.SGD(learning_rate=0.001)
    with tf.GradientTape() as tape:
        # compute forward loss
        loss = compute_train_step_loss()
        # back propagate
        grads = tape.gradient(loss, model.trainable_weights)
        optim.apply_gradients(zip(grads, model.trainable_weights))

def save_float_model(model: keras.Model, name: str, debug: bool = False) -> None:
    """Saves a NN float model.

    Parameters
    ----------
    model : keras.Model
        Model to be saved.
    name : str
        Name of the model.
    debug : bool, optional
        If true exports the model graph.
    """
    model.save(f"{name}.keras")
    if debug: keras.utils.plot_model(model, f"{name}.png",
                                     show_shapes=True, expand_nested=True)

def convert_tflite(model: keras.Model, inputs: Sequence[tf.Tensor], name: str) -> None:
    """Converts a NN model to TensorFlow Lite and saves it.

    Parameters
    ----------
    model : keras.Model
        Model to be converted.
    inputs : sequence of tf.Tensor
        Tensors to be fed to the model while converting.
    name : str
        Name of the model (used for the filename).
    """
    # TensorFlow Lite conversion
    def representative_dataset():
        for _ in range(2):
            # new random data tensor for each loop iteration
            # based on 'inputs' shape and dtype
            yield [tf.random.normal(t.shape, dtype=t.dtype) for t in inputs]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()
    with open(f"{name}.tflite", "wb") as f:
        f.write(tflite_quant_model)

def export_onnx(model: keras.Model, inputs: Sequence[tf.Tensor],
                name: str, debug: bool = False) -> None:
    """Exports an ONNX model and runs inference on it.

    Parameters
    ----------
    model : keras.Model
        Model to be exported.
    inputs : sequence of tf.Tensor
        Tensors to be fed to the model while exporting.
    name : str
        Name of the model (used for the filename).
    debug : bool, optional
        If true prints the ONNX model signature.
    """
    signature = [tf.TensorSpec.from_tensor(t) for t in inputs]
    # ONNX export
    tf2onnx.convert.from_keras(model, signature, output_path=f"{name}.onnx")
    # ONNX inference
    session = onnxruntime.InferenceSession(f"{name}.onnx",
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"])
    ortinputs = {session.get_inputs()[idx].name:
            onnxruntime.OrtValue.ortvalue_from_numpy(inputs[idx].numpy())
            for idx in range(len(inputs))}
    ortoutputs = [out.name for out in session.get_outputs()]
    session.run(ortoutputs, ortinputs)
    if debug:
        for t in session.get_inputs():
            print(f"{t.name} {t.shape} {t.type}")
        for t in session.get_outputs():
            print(f"{t.name} {t.shape} {t.type}")

def measure_latency(model: keras.Model, inputs: Sequence[tf.Tensor],
                    name: str, repetitions: int = 1000) -> None:
    """Measures the inference latency of a NN model.

    Parameters
    ----------
    model : keras.Model
        Model to be timed.
    inputs : sequence of tf.Tensor
        Tensors to be fed to the model while timing.
    name : str
        Name of the model.
    repetitions : int, optional
        Number of times to measure the latency.
    """
    print(f"Measuring {name} float model latency ({repetitions} runs)...")
    # the first few runs usually have increased latency,
    # since the inference device is likely idling,
    # so do a warm-up first to avoid skewing the measurements
    warmup = 10

    # graph compilation
    @tf.function
    def infer(tensors):
        return model(tensors, training=False)

    # warmup function
    def setup():
        for _ in range(warmup):
            infer(inputs)

    # timing function
    def loop():
        tensors = [tf.random.normal(t.shape, dtype=t.dtype) for t in inputs]
        infer(tensors)

    elapsedtime = timeit(loop, setup, number=repetitions)
    print(f"Inference latency:\t{elapsedtime/repetitions:.5f} s\t({repetitions/elapsedtime:.2f} FPS)")


class TestMultiple:
    """Tests two batch and image sizes."""

    @pytest.fixture(scope="class", params=[1, 2])
    def batchsize(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[3, 1])
    def channels(self, request):
        return request.param

    def test_run_float_inference(self, model, inputs, name, debug):
        run_float_inference(model, inputs, name, debug)


class TestSingle:
    """Tests a single image batch and default image size."""

    def test_save_float_model(self, model, name, debug):
        save_float_model(model, name, debug)

    def test_convert_tflite(self, model, inputs, name):
        convert_tflite(model, inputs, name)

    def test_export_onnx(self, model, inputs, name, debug):
        export_onnx(model, inputs, f"TF_{name}", debug)



if __name__ == "__main__":
    # import all model definitions to be tested
    import nn_models.tensorflow_keras as classification
    from nn_models.tensorflow_keras import object_detection
    from nn_models.tensorflow_keras import keypoints_regression
    # list with all the models to be tested
    models = [
                classification.resnet.ResNet18,
                classification.resnet.ResNet34,
                keras.applications.ResNet50,
                classification.resnet.ResNet50,
                classification.resnet.ResNet101,
                classification.resnet.ResNet152,
                keras.applications.MobileNetV2,
                classification.mobilenet.MobileNetV2,
                # object_detection.fcos.FCOS_Full,
                keypoints_regression.mobilepose.MobilePose_Full
            ]
    for mod in models:
        try:
            if mod is object_detection.fcos.FCOS_Full:
                img_shape = (800, 800, 3)
                # create model
                model = mod(img_size=(800, 800))
            else:
                img_shape = (224, 224, 3)
                # create model
                model = mod()
        except Exception as error:
            print("Model definition error!")
            print(error)
        # only if there is no error in the model definition
        else:
            # model name
            name = model.name
            # model inputs, with random data
            inputs = (tf.random.normal((1, *img_shape)), )
            # measure latency
            measure_latency(model, inputs, name)

