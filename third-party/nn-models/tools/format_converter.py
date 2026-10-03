"""Framework (Keras and PyTorch) parameters conversion script.

Use `-h` switch when running the script to print usage help.
"""

if __name__ == "__main__":
    import argparse
    from ast import literal_eval

    ap = argparse.ArgumentParser(description="Converts between Keras and PyTorch formats.")

    ap.add_argument("-c", "--conversion",
                    choices=("keras2pytorch", "pytorch2keras"),
                    required=True,
                    help="Conversion direction")
    ap.add_argument("-i", "--input_file",
                    required=True,
                    help="Input model parameters file")
    ap.add_argument("-o", "--output_file",
                    required=True,
                    help="Output model parameters file")
    ap.add_argument("-m", "--model",
                    choices=("fcos", "mobilepose"),
                    required=True,
                    help="Model architecture")
    ap.add_argument("-a", "--model_args",
                    nargs='*',
                    help="Model arguments, please check individual model class requirements...")

    args = ap.parse_args()
    # parse model arguments if any are passed to the script
    if args.model_args is not None:
        model_args = tuple(literal_eval(arg) for arg in args.model_args)

    if args.conversion == "pytorch2keras":
        if args.model == "fcos":
            from mapping.fcos import convert_pytorch_into_keras as convert
            if args.model_args is not None:
                convert(args.input_file, args.output_file, model_args)
            else:
                raise ValueError("Missing model arguments!")
        elif args.model == "mobilepose":
            from mapping.mobilepose import convert_pytorch_into_keras as convert
            if args.model_args is not None:
                convert(args.input_file, args.output_file, model_args)
            else:
                raise ValueError("Missing model arguments!")
        else:
            raise NotImplementedError(f"{args.conversion} of {args.model} not implemented!")
    else:
        raise NotImplementedError(f"{args.conversion} not implemented!")

