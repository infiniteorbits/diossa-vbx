"""PyTorch NNs tests module.

Tests:
    Float model (inference and saving)
    TorchScript (script and trace)
    ONNX export and inference
"""

import torch
import onnxruntime
from timeit import timeit
from typing import Tuple, Union, Callable
import pytest


def run_float_inference(model: torch.nn.Module,
                        inputs: Tuple[torch.Tensor, ...],
                        name: str, debug: bool = False) \
                            -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Runs the inference on a NN model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be inferred.
    inputs : tuple of torch.Tensor
        Tensors to be fed to the model.
    name : str
        Name of the model.
    debug : bool, optional
        If true prints the model layers.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Model output(s).
    """
    # set model in evaluation mode
    model.eval()
    # infer float model
    with torch.no_grad():
        outputs = model(*inputs)
    if debug: print(model)
    return outputs

def run_train_step(model: torch.nn.Module,
                   compute_train_step_loss: Callable[[], torch.Tensor],
                   name: str) -> None:
    """Runs a training step on a NN model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained.
    compute_train_step_loss : callable
        Function to compute the training step loss.
    name : str
        Name of the model.
    """
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    # set model in training mode
    model.train()
    # compute forward loss
    loss = compute_train_step_loss()
    # back propagate
    loss.backward()
    optim.step()

def save_float_model(model: torch.nn.Module, name: str) -> None:
    """Saves a NN float model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be saved.
    name : str
        Name of the model (used for the filename).
    """
    torch.save(model.state_dict(), f"{name}.pth")

def compile_torchscript(model: torch.nn.Module, name: str, debug: bool = False) -> None:
    """Compiles a NN model into TorchScript format.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be compiled.
    name : str
        Name of the model.
    debug : bool, optional
        If true prints the TorchScript model code.
    """
    # set model in evaluation mode
    model.eval()
    # compile TorchScript
    scriptmodel = torch.jit.script(model)
    if debug: print(scriptmodel.code)

def trace_model(model: torch.nn.Module , inputs: Tuple[torch.Tensor, ...],
                name: str, debug: bool = False) -> None:
    """Runs a trace on a NN model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be traced.
    inputs : tuple of torch.Tensor
        Tensors to be fed to the model while tracing.
    name : str
        Name of the model.
    debug : bool, optional
        If true prints the traced model layers.
    """
    # set model in evaluation mode
    model.eval()
    # trace test
    tracemodel = torch.jit.trace(model, inputs)
    if debug: print(tracemodel)

def export_onnx(model: torch.nn.Module, inputs: Tuple[torch.Tensor, ...],
                name: str, debug: bool = False) -> None:
    """Exports an ONNX model and runs inference on it.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be exported.
    inputs : tuple of torch.Tensor
        Tensors to be fed to the model while exporting.
    name : str
        Name of the model (used for the filename).
    debug : bool, optional
        If true prints the ONNX model signature.
    """
    # ONNX export
    torch.onnx.export(model, inputs, f"{name}.onnx")
    # ONNX inference
    session = onnxruntime.InferenceSession(f"{name}.onnx",
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"])
    # if the inputs are a tuple of tuples of torch.Tensors,
    # then they will be flattened to a tuple of torch.Tensors
    # during the tracing in the ONNX export,
    # so they also need to be flattened before feeding to the ONNX model
    flat_tuple = inputs \
            if isinstance(inputs[0], torch.Tensor) \
            else tuple(tensor for tupl in inputs for tensor in tupl)
    ortinputs = {session.get_inputs()[idx].name:
            onnxruntime.OrtValue.ortvalue_from_numpy(flat_tuple[idx].cpu().numpy())
            for idx in range(len(flat_tuple))}
    ortoutputs = [out.name for out in session.get_outputs()]
    session.run(ortoutputs, ortinputs)
    if debug:
        for t in session.get_inputs():
            print(f"{t.name} {t.shape} {t.type}")
        for t in session.get_outputs():
            print(f"{t.name} {t.shape} {t.type}")

def measure_latency(model: torch.nn.Module, inputs: Tuple[torch.Tensor, ...],
                    name: str, repetitions: int = 1000) -> None:
    """Measures the inference latency of a NN model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be timed.
    inputs : tuple of torch.Tensor
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
    # set model in evaluation mode
    model.eval()

    unpack = inputs[0].dim() == 4

    # warmup function
    def setup():
        with torch.no_grad():
            for _ in range(warmup):
                if unpack: model(*inputs)
                else: model(inputs)

    # timing function
    def loop():
        with torch.no_grad():
            tensors = tuple(torch.rand(t.shape, dtype=t.dtype, device=t.device) for t in inputs)
            if unpack: model(*tensors)
            else: model(tensors)

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

    def test_save_float_model(self, model, name):
        save_float_model(model, name)

    def test_export_onnx(self, model, inputs, name, debug):
        export_onnx(model, inputs, f"PT_{name}", debug)

    def test_compile_torchscript(self, model, name, debug):
        compile_torchscript(model, name, debug)

    def test_trace_model(self, model, inputs, name, debug):
        trace_model(model, inputs, name, debug)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # import all model definitions to be tested
    import nn_models.pytorch as classification
    from nn_models.pytorch import object_detection
    from nn_models.pytorch import keypoints_regression
    import torchvision
    from ots.mobilepose.network import CoordRegressionNetwork
    # list with all the models to be tested
    models = [
                classification.resnet.ResNet18,
                classification.resnet.ResNet34,
                torchvision.models.resnet50,
                classification.resnet.ResNet50,
                classification.resnet.ResNet101,
                classification.resnet.ResNet152,
                torchvision.models.mobilenet_v2,
                classification.mobilenet.MobileNetV2,
                torchvision.models.detection.fasterrcnn_resnet50_fpn,
                object_detection.faster_rcnn.FasterRCNN,
                torchvision.models.detection.fcos_resnet50_fpn,
                object_detection.fcos.FCOS_Full,
                CoordRegressionNetwork,
                keypoints_regression.mobilepose.MobilePose_Full
            ]
    for mod in models:
        try:
            if mod in (object_detection.faster_rcnn.FasterRCNN,
                       object_detection.fcos.FCOS_Full):
                # create model
                model = mod(img_size=(800, 800))
                model.to(device)
                # model inputs, with random data
                inputs = (torch.rand((1, 3, 800, 800), device=device), )
            elif mod is keypoints_regression.mobilepose.MobilePose_Full:
                # create model
                model = mod(keypoints=8)
                model.to(device)
                # model inputs, with random data
                inputs = (torch.rand((1, 3, 224, 224), device=device), )
            elif mod in (torchvision.models.detection.fasterrcnn_resnet50_fpn,
                         torchvision.models.detection.fcos_resnet50_fpn):
                # create model
                model = mod()
                model.to(device)
                # model inputs, with random data
                inputs = (torch.rand((3, 800, 800), device=device), )
            elif mod is CoordRegressionNetwork:
                # create model
                model = mod(8, "mobilenetv2")
                model.to(device)
                # model inputs, with random data
                inputs = (torch.rand((1, 3, 224, 224), device=device), )
            else:
                # create model
                model = mod()
                model.to(device)
                # model inputs, with random data
                inputs = (torch.rand((1, 3, 224, 224), device=device), )
        except Exception as error:
            print("Model definition error!")
            print(error)
        # only if there is no error in the model definition
        else:
            # model name
            name: str = model.__class__.__name__
            # remove suffix
            suffix = "_Full"
            if name.endswith(suffix):
                name = name[:-len(suffix)]
            # measure latency
            measure_latency(model, inputs, name)

