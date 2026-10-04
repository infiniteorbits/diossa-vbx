#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="INT8 Quantization CLI for TensorFlow SavedModel using NumPy Calibration Data."
    )
    parser.add_argument(
        "-s",
        "--saved_model_dir",
        required=True,
        type=str,
        help="Path to input TensorFlow SavedModel directory.",
    )
    parser.add_argument(
        "-c",
        "--calib_npy",
        required=True,
        type=str,
        help="Path to .npy calibration dataset file.",
    )
    parser.add_argument(
        "-o",
        "--output_tflite",
        required=True,
        type=str,
        help="Path to output .tflite file.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="int8",
        choices=["int8", "uint8", "float32"],
        help="Inference input tensor data type (default: int8).",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="int8",
        choices=["int8", "uint8", "float32"],
        help="Inference output tensor data type (default: int8).",
    )
    return parser.parse_args()


def get_dtype(type_str):
    mapping = {
        "int8": tf.int8,
        "uint8": tf.uint8,
        "float32": tf.float32,
    }
    return mapping[type_str]


def main():
    args = parse_args()

    # Load calibration dataset
    print(f"Loading calibration data from: {args.calib_npy}")
    try:
        calib_data = np.load(args.calib_npy).astype(np.float32)
    except Exception as e:
        print(f"Error loading calibration .npy file: {e}")
        sys.exit(1)

    print(f"Calibration data array shape: {calib_data.shape}")

    # Representative dataset generator
    def representative_dataset():
        for i in range(len(calib_data)):
            # Feed sample batch by batch (1, C, H, W) or (1, H, W, C)
            yield [calib_data[i : i + 1]]

    # Configure TFLite Converter
    print(f"Loading SavedModel from: {args.saved_model_dir}")
    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # Enforce integer specs
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = get_dtype(args.input_type)
    converter.inference_output_type = get_dtype(args.output_type)

    print("Running INT8 calibration and conversion...")
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Quantization failed: {e}")
        sys.exit(1)

    # Save output .tflite file
    with open(args.output_tflite, "wb") as f:
        f.write(tflite_model)

    print(f"Successfully exported INT8 TFLite model to: {args.output_tflite}")


if __name__ == "__main__":
    main()