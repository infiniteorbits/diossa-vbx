"""Export a PyTorch module class to ONNX using the torch.onnx.export function.

Pass --ckpt to load weights. Omit it to export the class as constructed.

Usage:
    python export_checkpoint_to_onnx.py \
        --onnx <path/to/onnx_file.onnx> \
        --input-shape <B> <C> <H> <W> \
        --model-class <path/to/model.py:ModelClass> \
        [--ckpt <path/to/checkpoint.ckpt>] \
        --device <device> \
        --input-names <input_name> <input_name> ... \
        --output-names <output_name> <output_name> ...
"""

import argparse

import torch

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export a PyTorch module class to ONNX")
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to the model .ckpt or .pth file. "
        "Omit to export the class with no loaded weights.",
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
        "--output-names", type=str, nargs="+", default=["output"],
        help="ONNX output tensor names (default: output)",
    )
    return parser.parse_args()

def load_model_class(class_path: str) -> type:
    """Load a model class from a dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def main() -> None:
    """Main function."""
    args = parse_args()
    device = args.device

    # Dynamically import model class
    model_class = load_model_class(args.model_class)

    # Instantiate model (assume default constructor, user can modify)
    model = model_class()
    model.to(device)
    model.eval()

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove 'module.' prefix if the model was trained with DataParallel
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

    # Prepare dummy input
    input_shape = tuple(args.input_shape)
    dummy_input = torch.randn(*input_shape, device=device)

    # dynamo=True (the torch.onnx.export default) only implements opset >= 18
    # and then tries to downgrade. That downgrade fails for this graph:
    # ONNX has no Identity adapter from opset 16 to 12, so the saved model
    # stays at opset 18. The legacy exporter emits opset 12 directly.
    torch.onnx.export(
        model.cpu(),
        dummy_input.cpu(),
        args.onnx,
        opset_version=12,
        dynamo=False,
        do_constant_folding=True,
        input_names=args.input_names,
        output_names=args.output_names,
    )
    print(f"ONNX model exported to {args.onnx}.")

if __name__ == "__main__":
    main()
