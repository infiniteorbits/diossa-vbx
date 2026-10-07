# NN models

The repository includes custom implementations of DNN models
based on papers and off-the-shelf implementations.  
The custom models are implemented in PyTorch and Keras and
are tailored for deployment onto edge devices.

Current implementations include:

- ResNet
- MobileNet
- Feature Pyramid Network
- Faster R-CNN
- FCOS
- MobilePose

## Tests

The unit tests are implemented in separate modules per framework
(_tests/pytorch/\*/unit_tests\_\*\_pth.py_ and
_tests/tensorflow_keras/\*/unit_tests\_\*\_tfk.py_).

1. Run float inference, including expected tensor shapes verification.
2. Run training step (only for some models).
3. Save float model.
4. Export ONNX.
5. (Keras) Convert to TensorFlow Lite.
6. (PyTorch) Compile TorchScript.
7. (PyTorch) Trace model.
8. Inspect model.
9. Quantize model (Post Training Quantization and Quantization Aware Training).

The framework used to automate unit testing is `pytest`
and all its test modules load the unit tests from
_tests/pytorch/\*/unit_tests\_\*\_pth.py_ and
_tests/tensorflow_keras/\*/unit_tests\_\*\_tfk.py_ modules.

```shell
# to run all the unit tests
## not recommended to load PyTorch and TensorFlow tests simultaneously
pytest
# useful options examples
## to run just a subset of the tests based on keyword expression matching
## (still loads all the tests for both frameworks)
pytest -k "pytorch"
pytest -k "pytorch and resnet"
pytest -k "pytorch and resnet and inference"
## if only one of PyTorch and Keras models are installed in the environment
## then to prevent import errors from loading the tests for both frameworks
pytest tests/pytorch # for PyTorch only
pytest tests/tensorflow_keras # for Keras only
## exit on first error
pytest -x
## increase verbosity by showing which test is running
pytest -v
## disable the display of warnings
## (the third party libraries have a large amount of warnings)
pytest --disable-warnings
## run the tests in subprocesses
## (required if running all the Keras models)
pytest --forked
## set TensorFlow C++ logging level (0==INFO; 1==WARNING; 2==ERROR; 3==NONE) while testing 
TF_CPP_MIN_LOG_LEVEL=1 pytest
```

### Manual tests

- Running the _tests/pytorch/base/unit_tests_pth.py_ or
  _tests/tensorflow_keras/base/unit_tests_tfk.py_ module
  directly will measure the frame rate of all
  the implemented models on the corresponding framework.
- Running the _tests/pytorch/base/validate...pth.py_ or
  _tests/tensorflow_keras/base/validate...tfk.py_ script
  will run the float inference and save an annotated image
  with the overlaid results for visual comparison.
  (The script's documentation describes the needed arguments.)

#### Examples

```shell
# measure framerate of the PyTorch models
python tests/pytorch/base/unit_tests_pth.py
# measure framerate of the Keras models
python tests/tensorflow_keras/base/unit_tests_tfk.py
```

```shell
# annotate testimage.png with the detection boxes from an off-the-shelf Faster R-CNN model
# and the custom model, using the same trained weights and bias from fasterrcnn.pth.
python tests/pytorch/base/validate_fasterrcnn_pth.py fasterrcnn.pth testimage.png
# annotate testimage.png with the detection boxes from an off-the-shelf FCOS model
# and the custom model, using the same trained weights and bias from fcos.pth.
python tests/pytorch/base/validate_fcos_pth.py fcos.pth testimage.png
# annotate testimage.png with the keypoints from an off-the-shelf MobilePose model
# and the custom model, using the same trained weights and bias from mobilepose.pth.
python tests/pytorch/base/validate_mobilepose_pth.py mobilepose.pth testimage.png groundtruth.json 8
# annotate testimage.png with the detection boxes from the custom model
# and the ground truth, using the trained weights and bias from fcos.keras.
python tests/tensorflow_keras/base/validate_fcos_tfk.py fcos.keras testimage.png groundtruth.json
# annotate testimage.png with the keypoints from the custom model
# and the ground truth, using the trained weights and bias from mobilepose.keras.
python tests/tensorflow_keras/base/validate_mobilepose_tfk.py mobilepose.keras testimage.png groundtruth.json 8
```

## Tools

The script _tools/format_converter.py_ can be used to transfer
trained parameters from models trained with PyTorch to TensorFlow.

### Embedded models generation

To generate the embedded models,
there is a script that loads the training parameters and
the quantized model and generates the model files ready for embedding.  
Running `python -m nn_models.pytorch.edge.vitis_ai.deployment.deploy -h`
displays its help information.
