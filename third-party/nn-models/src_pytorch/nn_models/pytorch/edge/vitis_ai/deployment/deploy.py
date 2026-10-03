"""Model deployment script.

To generate the model files (.xmodel and .onnx)
required to run inference on the MPSoC.
"""

import argparse
import os.path
import subprocess
from inspect import Signature, signature
from typing import Optional, Sequence

import torch
import yaml


def compile(xmodel: str, arch: str, net_name: str,
            output_ops: Optional[Sequence[str]] = None) -> None:
    """Compiles a quantized model.

    Parameters
    ----------
    xmodel : str
        Quantized xmodel file path.
    arch : str
        DPU architecture JSON file path.
    net_name : str
        Network name, used for the filename.
    output_ops : str, optional
        Names of the output tensors for the compiled model.
        (If None, the compiled model
        will have the same outputs as the quantized model.)
    """
    args = [
        "vai_c_xir",
        "-x", xmodel,
        "-a", arch,
        "-o", "compiled",
        "-n", net_name
    ]
    if output_ops is not None:
        args.append("-e")
        args.append(f'{{"output_ops": "{",".join(output_ops)}"}}')
    print(args)
    # it would be better to use the imported vai_c_xir for the compilation,
    # but it does not import dependencies correctly when used as module,
    # which means it only works fine if it is run as main,
    # so vai_c_xir is being run in a subprocess
    subprocess.run(args, check=True)

def filter_configuration(config: dict, signature: Signature) -> dict:
    """Filters a model configuration based on its signature.

    Parameters
    ----------
    config : dict
        Model configuration (keyword arguments).
    signature : Signature
        Model constructor signature.

    Returns
    -------
    dict
        Filtered model configuration including
        only the arguments present in its signature.
    """
    cfg = {}
    for k, v in config.items():
        if k in signature.parameters:
            cfg[k] = v
    return cfg


def run():
    """Script entry point."""
    parser  = argparse.ArgumentParser(description="Embedded model files generator")
    parser.add_argument("--model", choices=["fcos", "mobilepose"], required=True, help="model name")
    parser.add_argument("--config", required=True, help="model configuration YAML file path")
    parser.add_argument("--qxmodel", required=True, help="quantized .xmodel file path")
    parser.add_argument("--qatpytorch", help="QAT PyTorch state_dict file path")
    parser.add_argument("--arch", help="DPU architecture JSON file path")
    args = parser.parse_args()

    # read configuration used in training
    with open(args.config, 'r') as yamlfile:
        config = yaml.safe_load(yamlfile)
    model_config = config["task"]["model"]["model_kwargs"]
    print(model_config)
    input_shape = config["task"]["model"].get("model_input_shape")
    print(input_shape)

    device = torch.device("cpu")

    if args.model == "fcos":
        from ....object_detection.fcos import FCOS, FCOS_Edge
        # create float base model (only used for raw data generation)
        sig = signature(FCOS)
        cfg = filter_configuration(model_config, sig)
        print(cfg)
        model = FCOS(**cfg).to(device)
        # create postprocessing model
        model_config.setdefault("normalize_boxsize",
                                sig.parameters["normalize_boxsize"].default)
        cfg = filter_configuration(model_config, signature(FCOS_Edge))
        print(cfg)
        postprocessing = FCOS_Edge(**cfg).to(device)
        # prepare raw data for ONNX export
        tensor = torch.rand((1, cfg.get("channels", 3), *cfg["img_size"][::-1]), device=device)
        model.eval()
        with torch.no_grad():
            raw = model(tensor)
        # set output tensors names for xmodel compilation
        output_ops = None
    elif args.model == "mobilepose":
        if args.qatpytorch is None:
            raise RuntimeError("MobilePose deployment requires QAT PyTorch state_dict file!")
        assert model_config.get("channels", 3) == input_shape[0], "number of channels mismatch!"

        from pytorch_nndct import QatProcessor

        from ..mobilepose import MobilePose, MobilePose_QAT
        from .mod.mobilepose import MobilePose_Edge_DUC
        # create quantized base model
        sig = signature(MobilePose)
        cfg = filter_configuration(model_config, sig)
        print(cfg)
        model = MobilePose_QAT(**cfg)
        # quantize the model and load quantized parameters from training
        tensor = torch.rand((1, *input_shape), device=device)
        qat_processor = QatProcessor(model, (tensor, ))
        train_model = qat_processor.trainable_model(allow_reused_module=False)
        train_model.load_state_dict(torch.load(args.qatpytorch, map_location="cpu"))
        # get intermediate tensor as raw data for ONNX export
        raw = None
        def hook(module, input, output):
            nonlocal raw
            raw = output.detach()
        train_model.conv_compress.register_forward_hook(hook)
        train_model(tensor)
        print(raw.shape)
        val_model = qat_processor.to_deployable(train_model, ".vai_qat")
        with torch.no_grad():
            val_model(tensor)
        # create postprocessing model
        model_config.setdefault("normalized_coordinates",
                                sig.parameters["normalized_coordinates"].default)
        cfg = filter_configuration(model_config, signature(MobilePose_Edge_DUC))
        print(cfg)
        edge_model = MobilePose_Edge_DUC(**cfg)
        # copy parameters from quantized model to postprocessing model
        with torch.no_grad():
            edge_model.duc1.conv.weight.copy_(val_model.inner_model.module_100.weight)
            edge_model.duc1.conv.bias.copy_(val_model.inner_model.module_100.bias)
            edge_model.duc2.conv.weight.copy_(val_model.inner_model.module_103.weight)
            edge_model.duc2.conv.bias.copy_(val_model.inner_model.module_103.bias)
            edge_model.duc3.conv.weight.copy_(val_model.inner_model.module_106.weight)
            edge_model.duc3.conv.bias.copy_(val_model.inner_model.module_106.bias)
            edge_model.conv_heatmap.weight.copy_(val_model.inner_model.module_109.weight)
        # set inputs for ONNX export and xmodel compilation
        postprocessing = edge_model
        output_ops = ["MobilePose_QAT__MobilePose_QAT_Conv2d_conv_compress__ret_233_fix"]
    else:
        raise NotImplementedError("Model not implemented yet!")

    print("Exporting ONNX model...")
    torch.onnx.export(postprocessing, raw, f"{args.model}.onnx")
    # if DPU architecture JSON is not given then use the one from ZCU104
    if args.arch is None:
        from vaic import vai_c_xir
        vaic_dir = os.path.dirname(vai_c_xir.__file__)
        # using the arch.json of ZCU104
        arch_json = os.path.join(vaic_dir, "arch/DPUCZDX8G/ZCU104/arch.json")
    else:
        arch_json = args.arch
    compile(args.qxmodel, arch_json, args.model, output_ops)


if __name__ == "__main__":
    run()

