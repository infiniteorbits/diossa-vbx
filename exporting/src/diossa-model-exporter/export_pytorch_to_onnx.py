"""Export a PyTorch module class to ONNX using the torch.onnx.export function.

Usage:
    python export_checkpoint_to_onnx.py \
        --ckpt <path/to/checkpoint.ckpt> \
        --onnx <path/to/onnx_file.onnx> \
        --input-shape <B> <C> <H> <W> \
        --model-class <path/to/model.py:ModelClass> \
        [--model-kwargs <key=value> ...] \
        [--postprocessing-model-class <path/to/model.py:PostClass>] \
        [--postprocessing-onnx <path/to/postprocessing.onnx>] \
        --device <device> \
        --input-names <input_name> <input_name> ... \
        --output-names <output_name> <output_name> ...

--model-kwargs values are JSON when they parse (numbers, true/false/null, lists)
and plain strings otherwise. Example:
    --model-kwargs img_size=[384,288] num_classes=2 normalize_boxsize=false

When --postprocessing-model-class is set, the backbone is run once and tensors
with those output shapes are used as the post-processing export inputs.
The post-processing constructor receives only the kwargs it accepts.
"""

import argparse
import inspect
import json
from pathlib import Path

import torch

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export a PyTorch module class to ONNX")
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to the model .ckpt or .pth file",
    )
    parser.add_argument(
        "--onnx", type=str, required=True,
        help="Where to save the exported onnx file",
    )
    parser.add_argument(
        "--input-shape", type=int, nargs="+", required=True,
        help="Input shape as space-separated ints in BCHW order, e.g. 1 3 256 256",
    )
    parser.add_argument(
        "--model-class", type=str, required=True,
        help="Full dotted class path to model, e.g. "
        "'nn_models_pytorch.keypoints_regression.mobilepose.MobilePose'",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to load weights and export ONNX (default: cpu)",
    )
    parser.add_argument(
        "--input-names", type=str, nargs="+", default=["input"],
        help="ONNX input tensor names (default: input)",
    )
    parser.add_argument(
        "--output-names", type=str, nargs="+", default=[],
        help="ONNX output tensor names (default: none)",
    )
    parser.add_argument(
        "--model-kwargs", type=parse_constructor_kwarg, nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Extra keyword arguments for the model constructor, as key=value. "
        "Values are JSON when they parse (numbers, true/false/null, lists) "
        "and plain strings otherwise. "
        "Example: img_size=[384,288] num_classes=2 normalize_boxsize=false",
    )
    parser.add_argument(
        "--postprocessing-model-class", type=str, default=None,
        help="Optional post-processing model class. Exported when set, using "
        "dummy inputs whose shapes come from one backbone forward pass.",
    )
    parser.add_argument(
        "--postprocessing-onnx", type=str, default=None,
        help="Where to save the post-processing ONNX file. "
        "Defaults to <onnx stem>_postprocessing.onnx.",
    )
    parser.add_argument(
        "--postprocessing-output-names", type=str, nargs="+", default=["output"],
        help="ONNX output tensor names for the post-processing model (default: output)",
    )
    return parser.parse_args()

def parse_constructor_kwarg(item: str) -> tuple[str, object]:
    """Parse one model-constructor keyword argument from key=value."""
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
    key, raw_value = item.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"missing key in {item!r}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value

def load_model_class(class_path: str) -> type:
    """Load a model class from a dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def constructor_kwargs(model_class: type, model_kwargs: dict[str, object]) -> dict[str, object]:
    """Keep kwargs that the model constructor accepts."""
    parameters = inspect.signature(model_class.__init__).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return model_kwargs
    accepted = {
        name
        for name, parameter in parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in model_kwargs.items() if key in accepted}

def iter_tensors(value: object):
    """Yield tensors from a tensor or a nested tuple/list of tensors."""
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_tensors(item)
        return
    raise TypeError(f"Unsupported model output type: {type(value)}")

def random_like(value: object) -> object:
    """Build random tensors with the same structure, shapes, and dtypes."""
    if isinstance(value, torch.Tensor):
        return torch.randn(tuple(value.shape), dtype=value.dtype, device=value.device)
    if isinstance(value, tuple):
        return tuple(random_like(item) for item in value)
    if isinstance(value, list):
        return [random_like(item) for item in value]
    raise TypeError(f"Unsupported model output type: {type(value)}")

def to_cpu(value: object) -> object:
    """Move tensors in a nested structure to CPU."""
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    raise TypeError(f"Unsupported model output type: {type(value)}")

def tensor_shapes(value: object) -> list[tuple[int, ...]]:
    """Return the shape of each tensor in a nested structure."""
    return [tuple(tensor.shape) for tensor in iter_tensors(value)]

def require_names(names: list[str], value: object, label: str) -> None:
    """Fail when the number of ONNX names does not match the tensors."""
    shapes = tensor_shapes(value)
    if len(names) != len(shapes):
        raise ValueError(
            f"{label} has {len(shapes)} tensor(s) with shapes {shapes}, "
            f"but {len(names)} name(s) were given: {names}"
        )

def default_postprocessing_onnx(onnx_path: str) -> str:
    """Derive the post-processing ONNX path from the backbone path."""
    path = Path(onnx_path)
    return str(path.with_name(f"{path.stem}_postprocessing{path.suffix}"))

def export_onnx(
    model: torch.nn.Module,
    example_inputs: object,
    onnx_path: str,
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Export a module with the legacy ONNX exporter at opset 12."""
    example_inputs = to_cpu(example_inputs)
    require_names(input_names, example_inputs, "ONNX inputs")
    # dynamo=True (the torch.onnx.export default) only implements opset >= 18
    # and then tries to downgrade. That downgrade fails for this graph:
    # ONNX has no Identity adapter from opset 16 to 12, so the saved model
    # stays at opset 18. The legacy exporter emits opset 12 directly.
    torch.onnx.export(
        model.cpu(),
        example_inputs,
        onnx_path,
        opset_version=12,
        dynamo=False,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )

def main() -> None:
    """Main function."""
    args = parse_args()
    device = args.device
    model_kwargs = dict(args.model_kwargs)

    model_class = load_model_class(args.model_class)
    print(f"Backbone Model class: {model_class}")
    print(f"Backbone Model kwargs: {model_kwargs}")
    model = model_class(**model_kwargs)
    model.to(device)
    model.eval()

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove 'module.' prefix if the model was trained with DataParallel
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)

    dummy_input = torch.randn(*tuple(args.input_shape), device=device)
    with torch.no_grad():
        backbone_outputs = model(dummy_input)
    print(f"Backbone output shapes: {tensor_shapes(backbone_outputs)}")
    if len(args.output_names) == 0:
        args.output_names = [f"output_{i}" for i in range(len(tensor_shapes(backbone_outputs)))]
    require_names(args.output_names, backbone_outputs, "Backbone outputs")

    export_onnx(model, dummy_input, args.onnx, args.input_names, args.output_names)
    print(f"ONNX model exported to {args.onnx}.")

    if args.postprocessing_model_class is None:
        return

    post_class = load_model_class(args.postprocessing_model_class)
    post_kwargs = constructor_kwargs(post_class, model_kwargs)
    print(f"Post-processing Model class: {post_class}")
    print(f"Post-processing Model constructor kwargs: {post_kwargs}")
    post_model = post_class(**post_kwargs)
    post_model.to(device)
    post_model.eval()

    post_inputs = random_like(backbone_outputs)
    post_onnx = args.postprocessing_onnx or default_postprocessing_onnx(args.onnx)
    print(f"Post-processing constructor kwargs: {post_kwargs}")
    print(f"Post-processing input shapes: {tensor_shapes(post_inputs)}")
    with torch.no_grad():
        post_outputs = post_model(*post_inputs) if isinstance(post_inputs, tuple) else post_model(post_inputs)
    require_names(args.postprocessing_output_names, post_outputs, "Post-processing outputs")

    export_onnx(
        post_model,
        post_inputs,
        post_onnx,
        args.output_names,
        args.postprocessing_output_names,
    )
    print(f"Post-processing ONNX model exported to {post_onnx}.")

if __name__ == "__main__":
    main()
