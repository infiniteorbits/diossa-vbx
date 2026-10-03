"""Keras NNs tests module.

Tests:
    Vitis-AI inspection, quantization (PTQ and QAT) and compilation
"""

import tensorflow as tf
from tensorflow import keras
from vaic import vai_c_tensorflow2
import os.path
import subprocess
from typing import Sequence


def inspect_model(model: keras.Model, custom_layers: dict = {}, debug: bool = False) -> None:
    """Inspects a NN model for MPSoC deployment feasibility.

    Parameters
    ----------
    model : keras.Model
        Model to be inspected.
    custom_layers : dict, optional
        Dictionary with all the custom layers included in the model definition.
    debug : bool, optional
        If true saves the model graph as svg.
    """
    # import inside the function to enable testing in a separate process
    from tensorflow_model_optimization.python.core.quantization.keras.vitis.vitis_inspect import VitisInspector
    inspector = VitisInspector(custom_objects=custom_layers,
                               target="DPUCZDX8G_ISA1_B4096")
    inspector.inspect_model(model,
                            plot = debug,
                            plot_file = os.path.join("quantize_result", "model.svg"),
                            dump_model = debug,
                            dump_model_file = os.path.join("quantize_result", "inspect_model.h5"),
                            dump_results = debug,
                            dump_results_file = os.path.join("quantize_result", "inspect_results.txt"))

def quantize_ptq_model(model: keras.Model, inputs: Sequence[tf.Tensor],
                       name: str, custom_layers: dict = {}, debug: bool = False) -> None:
    """Quantizes a NN model, with post training quantization, for MPSoC deployment.

    Parameters
    ----------
    model : keras.Model
        Model to be quantized and compiled.
    inputs : sequence of tf.Tensor
        Tensors to be fed to the model while quantizing.
    name : str
        Name of the model (used for the filename).
    custom_layers : dict, optional
        Dictionary with all the custom layers included in the model definition.
    debug : bool, optional
        If true saves quantized model ONNX.
    """
    # import inside the function to enable testing in a separate process
    from tensorflow_model_optimization.python.core.quantization.keras.vitis.vitis_quantize import VitisQuantizer
    # dataset with random data for quantization,
    # same shape and dtype as 'inputs', but with a batch of 2
    dataset = [tf.random.normal((2, *t.shape[1:]), dtype=t.dtype) for t in inputs]
    quantizer = VitisQuantizer(model,
                               custom_objects=custom_layers,
                               target="DPUCZDX8G_ISA1_B4096")
    if not debug:
        quantized_model = quantizer.quantize_model(calib_dataset=dataset)
    else:
        quantized_model = quantizer.quantize_model(calib_dataset=dataset,
                                                   output_format="onnx",
                                                   output_dir="quantize_result")
    quantized_model.predict(dataset)
    modelfile = os.path.join("quantize_result", f"{name}.h5")
    quantized_model.save(modelfile)
    # if debug: VitisQuantizer.dump_model(quantized_model, weights_only=True)

    compile_model(modelfile, name)

def quantize_qat_model(model: keras.Model, inputs: Sequence[tf.Tensor],
                       name: str, custom_layers: dict = {}, debug: bool = False) -> None:
    """Quantizes a NN model, with quantization aware training, for MPSoC deployment.

    Does not train the model,
    juts converts it to a quantization aware trainable model.

    Parameters
    ----------
    model : keras.Model
        Model to be quantized and compiled.
    inputs : sequence of tf.Tensor
        Tensors to be fed to the model while quantizing.
    name : str
        Name of the model (used for the filename).
    custom_layers : dict, optional
        Dictionary with all the custom layers included in the model definition.
    debug : bool, optional
        If true saves quantized model ONNX.
    """
    # import inside the function to enable testing in a separate process
    from tensorflow_model_optimization.python.core.quantization.keras.vitis.vitis_quantize import VitisQuantizer
    # dataset with random data for quantization,
    # same shape and dtype as 'inputs', but with a batch of 2
    dataset = [tf.random.normal((2, *t.shape[1:]), dtype=t.dtype) for t in inputs]
    quantizer = VitisQuantizer(model,
                               custom_objects=custom_layers,
                               target="DPUCZDX8G_ISA1_B4096")
    # model for training
    train_model = quantizer.get_qat_model(init_quant=True, calib_dataset=dataset)
    # on models with custom layers the VitisQuantizer.get_deploy_model()
    # has to infer the shapes of the custom layers, otherwise the compilation will fail.
    # unfortunately current get_deploy_model() interface only allows single tensor shape input,
    # so, using assertions, it is verified here that the API requirements are met.
    if custom_layers:
        assert isinstance(inputs, (tuple, list)), f"Expected inputs as a tuple or list, but got {type(inputs)}!"
        assert len(inputs) == 1, "Vitis-AI API for models with custom layers only supports single tensor input!"
        add_shape_info = True
        input_shape = tuple(inputs[0].shape[1:])
    else:
        add_shape_info = False
        input_shape = None
    # model for deployment (and validation)
    if not debug:
        quant_model = VitisQuantizer.get_deploy_model(train_model,
                                                      add_shape_info=add_shape_info,
                                                      input_shape=input_shape)
    else:
        quant_model = VitisQuantizer.get_deploy_model(train_model,
                                                      add_shape_info=add_shape_info,
                                                      input_shape=input_shape,
                                                      output_format="onnx",
                                                      output_dir="quantize_result")
    quant_model.predict(dataset)
    modelfile = os.path.join("quantize_result", f"{name}.h5")
    quant_model.save(modelfile)
    # if debug: VitisQuantizer.dump_model(quant_model, weights_only=True)

    compile_model(modelfile, name)

def compile_model(filepath: str, modelname: str) -> None:
    """Compiles a NN model for MPSoC deployment.

    Parameters
    ----------
    filepath : str
        Input model file path.
    modelname : str
        Name of the model, for output file name.
    """
    # get the arch.json from vaic package
    vaic_dir = os.path.dirname(vai_c_tensorflow2.__file__)
    # using the arch.json of ZCU104
    arch_json = os.path.join(vaic_dir, "arch/DPUCZDX8G/ZCU104/arch.json")
    args = [
        "vai_c_tensorflow2",
        "-m", filepath,
        "-a", arch_json,
        "-o", "compiled",
        "-n", modelname
    ]
    # it would be better to use the imported vai_c_tensorflow2 for the compilation,
    # but it does not import dependencies correctly when used as module,
    # which means it only works fine if it is run as main,
    # so vai_c_xir is being run in a subprocess
    subprocess.run(args, check=True)


class TestVitisai:
    """Tests Vitis-AI quantization."""

    def test_inspect_model(self, model, debug):
        inspect_model(model, debug=debug)

    def test_quantize_ptq_model(self, model, inputs, name, debug):
        quantize_ptq_model(model, inputs, f"TF_{name}_PTQ", debug=debug)

    def test_quantize_qat_model(self, model, inputs, name, debug):
        quantize_qat_model(model, inputs, f"TF_{name}_QAT", debug=debug)

