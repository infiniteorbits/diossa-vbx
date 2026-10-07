"""Keras (TensorFlow 2) FCOS validation script.

Loads a trained `FCOS` model
and overlays the detected bounding boxes annotations on an image,
as well as the ground truth.

Examples
--------
>>> python validate_fcos_tfk.py model.keras image.png gt.json

An image will be saved in the current directory
with the same filename, but prefixed with "annotated_".
"""

import sys
import os.path
import json

import numpy as np
import cv2
from PIL import Image, ImageDraw

from nn_models.tensorflow_keras.object_detection import fcos


if len(sys.argv) != 4:
    print(f"Usage: python {os.path.basename(sys.argv[0])} <model.keras> <image.png> <gt.json>")
    sys.exit()

model_file = sys.argv[1]
image_file = sys.argv[2]
gt_file = sys.argv[3]

OTS_TRAINING = False

# read ground truth
image_filename = os.path.basename(image_file)
with open(gt_file, 'r') as file:
    for item in json.load(file):
        if item["filename"] == image_filename:
            gt_bbox = item["bbox"]
            gt_label = item["obj"]
            break
    else:
        print(f"{image_filename} ground truth not found in {gt_file}. Exiting...")
        sys.exit()

# load image with OpenCV
image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
print(image.shape, image.dtype)
if image.dtype == np.uint8:
    bits = 8
elif image.dtype == np.uint16:
    bits = 16
else:
    print("Unsupported pixel bit depth!")
    sys.exit()

## model input image size
in_img_w = 1056
in_img_h = 800

# resize image
## scale factor based on the smaller size, the height
factor = in_img_h / image.shape[0]
resize_image : np.ndarray = cv2.resize(image, None, fx=factor, fy=factor)
print(resize_image.shape, resize_image.dtype)

# crop image
## offset amount to be cropped on each side (on the width)
offset = (resize_image.shape[1] - in_img_w) // 2
crop_image = resize_image[:, offset : offset + in_img_w, ::-1].copy()
print(crop_image.shape, crop_image.dtype)

# normalize image
norm_image : np.ndarray = crop_image.astype(np.float32) / (2**bits-1)
print(norm_image.dtype, norm_image.min(), norm_image.max())
# image shape is HWC

# image to tensor
mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
std = np.array((0.229, 0.224, 0.225), dtype=np.float32)
tensor = np.expand_dims((norm_image - mean) / std, axis=0)
print(tensor.shape, tensor.dtype)
# tensor shape is NHWC

# LMO model
print("# LMO model")
try:
    img_size = (tensor.shape[2], tensor.shape[1]) # W, H
    # create model
    lmo_model = fcos.FCOS(img_size=img_size,
                          num_classes=2 if not OTS_TRAINING else 91,
                          dynamic_shapes=True)
    lmo_model.load_weights(model_file)
except Exception:
    print("Model definition error!")
    raise
else:
    # infer float model
    inputs = (tensor, )
    outputs = lmo_model.post(*lmo_model(*inputs))
    # first tensor is boxes
    # first box is top detection
    boxes = outputs[0][0,0]
    print(boxes)

    # scale ratio based on the smaller size, the height
    scale = image.shape[0] / tensor.shape[1]
    print(scale)
    # add the offset due to cropping
    boxes = boxes.numpy()
    boxes[0::2] += (image.shape[1]/scale - tensor.shape[2]) / 2
    # scale the coordinates to the original image size
    boxes *= scale
    print(boxes)
    lmo_boxes = boxes.tolist()

    # score
    lmo_score = outputs[1][0,0]
    print(lmo_score)
    # label
    lmo_label = outputs[2][0,0]
    print(lmo_label)

# annotate the image
with Image.open(image_file) as im:
    draw = ImageDraw.Draw(im)
    # bounding boxes
    boxes = (gt_bbox, lmo_boxes)
    labels = ("ground_truth", "LMO")
    colors = ("green", "red")
    width = 1
    for bbox, color, label in zip(boxes, colors, labels):
        draw.rectangle(bbox, width=width, outline=color)
        margin = width + 1
        draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=color)
    # save the annotated image
    im.save(f"annotated_{image_filename}")

