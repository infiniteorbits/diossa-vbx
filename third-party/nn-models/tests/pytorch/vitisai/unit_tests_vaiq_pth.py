"""PyTorch NNs Vitis-AI quantizer tests module.

Tests:
    Vitis-AI inspection, quantization (PTQ and QAT) and compilation
"""

import torch
from pytorch_nndct.apis import Inspector, torch_quantizer
from pytorch_nndct import QatProcessor
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS # for workaround to reset shared device variable
from vaic import vai_c_xir
import os.path
import subprocess
from typing import Tuple, Callable, Optional
from ..base.unit_tests_pth import run_train_step, trace_model


def inspect_model(model: torch.nn.Module, inputs: Tuple[torch.Tensor, ...],
                    device: torch.device, debug: bool = False) -> None:
    """Inspects a NN model for MPSoC deployment feasibility.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be inspected.
    inputs : tuple of torch.Tensor
        Tensors to be fed to the model while inspecting.
    device : torch.device
        Device where to process the operations (CPU, GPU).
    debug : bool, optional
        If true saves the model graph as svg.
    """
    inspector = Inspector("DPUCZDX8G_ISA1_B4096")
    inspector.inspect(model, inputs, device,
                      verbose_level = 2 if debug else 1,
                      image_format = "svg" if debug else None)

def quantize_ptq_model(model: torch.nn.Module, inputs: Tuple[torch.Tensor, ...],
                       device: torch.device, name: str, debug: bool = False) -> None:
    """Quantizes a NN model, with post training quantization, for MPSoC deployment.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be quantized and compiled.
    inputs : tuple of torch.Tensor
        Tensors to be fed to the model while quantizing.
    device : torch.device
        Device where to process the operations (CPU, GPU).
    name : str
        Name of the model.
    debug : bool, optional
        If true also saves TorchScript and ONNX models
        and prints the quantized model layers.
    """
    # quantization requires a passing trace test
    trace_model(model, inputs, name)

    # run first calibration and then testing
    for quant_mode in ("calib", "test"):
        if quant_mode == "calib":
            # dataset with random data for calibration,
            # same shape and dtype as 'inputs', but with a batch of 2
            dataset = tuple(torch.rand((2, *t.shape[1:]), dtype=t.dtype, device=device) for t in inputs)
            quantizer = torch_quantizer(quant_mode, model, dataset, device=device, target="DPUCZDX8G_ISA1_B4096")
            quant_model = quantizer.quant_model
            quant_model.eval()
            with torch.no_grad():
                quant_model(*dataset)
            quantizer.export_quant_config()
        elif quant_mode == "test":
            quantizer = torch_quantizer(quant_mode, model, inputs, device=device, target="DPUCZDX8G_ISA1_B4096")
            quant_model = quantizer.quant_model
            quant_model.eval()
            with torch.no_grad():
                quant_model(*inputs)
            if debug:
                quantizer.export_torch_script()
                quantizer.export_onnx_model()
            quantizer.export_xmodel(deploy_check=debug)
    if debug: print(quant_model)

    compile_model(os.path.join("quantize_result", f"{model.__class__.__name__}_int.xmodel"), name)

def quantize_qat_model(model: torch.nn.Module, inputs: Tuple[torch.Tensor, ...],
                       device: torch.device, name: str,
                       compute_train_step_loss: Optional[Callable[[], torch.Tensor]] = None,
                       shared_layers: bool = False, debug: bool = False) -> None:
    """Quantizes a NN model, with quantization aware training, for MPSoC deployment.

    Does not train the model,
    juts converts it to a quantization aware trainable model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be quantized and compiled.
    inputs : tuple of torch.Tensor
        Tensors to be fed to the model while quantizing.
    device : torch.device
        Device where to process the operations (CPU, GPU).
    name : str
        Name of the model.
    compute_train_step_loss : callable, optional
        Function to compute the training step loss.
        If passed runs a training step,
        otherwise the training step is skipped.
    shared_layers: bool, default = False
        Allows reusing modules for models with shared layers.
    debug : bool, optional
        If true also saves TorchScript and ONNX models
        and prints the quantized model layers.
    """
    qat_processor = QatProcessor(model, inputs)
    # model for training
    train_model = qat_processor.trainable_model(allow_reused_module=shared_layers)
    if compute_train_step_loss is not None:
        run_train_step(train_model, compute_train_step_loss, name)
    # model for validation
    val_model = qat_processor.to_deployable(train_model, ".vai_qat")
    val_model.eval()
    with torch.no_grad():
        val_model(*inputs)
    # model for deployment
    quant_model = qat_processor.deployable_model(".vai_qat", True)
    quant_model.eval()
    with torch.no_grad():
        quant_model(*inputs)
    if debug:
        qat_processor.export_torch_script(".vai_qat")
        qat_processor.export_onnx_model(".vai_qat")
    qat_processor.export_xmodel(".vai_qat", deploy_check=debug)
    if debug: print(quant_model)

    compile_model(os.path.join(".vai_qat", f"{model.__class__.__name__}_0_int.xmodel"), name)

    # deployable_model(..., used_for_xmodel=True) sets shared QUANT_DEVICE="cpu"
    # workaround: reset shared variable to used device, before exiting
    GLOBAL_MAP.set_map(NNDCT_KEYS.QUANT_DEVICE, device)

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
    vaic_dir = os.path.dirname(vai_c_xir.__file__)
    # using the arch.json of ZCU104
    arch_json = os.path.join(vaic_dir, "arch/DPUCZDX8G/ZCU104/arch.json")
    args = [
        "vai_c_xir",
        "-x", filepath,
        "-a", arch_json,
        "-o", "compiled",
        "-n", modelname
    ]
    # it would be better to use the imported vai_c_xir for the compilation,
    # but it does not import dependencies correctly when used as module,
    # which means it only works fine if it is run as main,
    # so vai_c_xir is being run in a subprocess
    subprocess.run(args, check=True)


class TestVitisai:
    """Tests Vitis-AI quantization."""

    def test_inspect_model(self, model, inputs, device, debug):
        inspect_model(model, inputs, device, debug)

    def test_quantize_ptq_model(self, model, inputs, device, name, debug):
        quantize_ptq_model(model, inputs, device, f"PT_{name}_PTQ", debug)

    def test_quantize_qat_model(self, model, inputs, device, name, debug):
        quantize_qat_model(model, inputs, device, f"PT_{name}_QAT", None, False, debug)

