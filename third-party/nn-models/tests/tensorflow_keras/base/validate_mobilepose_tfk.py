"""Keras (TensorFlow 2) MobilePose validation script.

Loads a trained `MobilePose` model
and overlays the detected keypoints annotations on an image,
as well as the ground truth.

Examples
--------
>>> python validate_mobilepose_tfk.py model.keras image.png gt.json 8

An image will be saved in the current directory
with the same filename, but prefixed with "annotated_".
"""

import sys
import os.path
import json

import numpy as np
import cv2
from PIL import Image, ImageDraw

from nn_models.tensorflow_keras.keypoints_regression import mobilepose


if len(sys.argv) != 5:
    print(f"Usage: python {os.path.basename(sys.argv[0])} <model.keras> <image.png> <gt.json> <number_keypoints>")
    sys.exit()

model_file = sys.argv[1]
image_file = sys.argv[2]
gt_file = sys.argv[3]
nkpts = int(sys.argv[4])

NORMALIZED_COORDS = True

# read ground truth
image_filename = os.path.basename(image_file)
with open(gt_file, 'r') as file:
    for item in json.load(file):
        if item["filename"] == image_filename:
            gt_bbox = item["bbox"]
            gt_kpts = list(zip(item["keypoints"][0::3], item["keypoints"][1::3]))
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

# adjust the bounding box
(gt_left, gt_top, gt_right, gt_bottom) = gt_bbox
print(gt_bbox)
gt_width = gt_right - gt_left
gt_height = gt_bottom - gt_top
gt_cx = gt_left + gt_width / 2
gt_cy = gt_top + gt_height / 2
(img_height, img_width) = image.shape[:2]
img_small_side = min(image.shape[:2])
## get largest side from ground truth
side = gt_width if gt_width >= gt_height else gt_height
## clamp side if it does not fit the image
if side > img_small_side:
    side = img_small_side
half_side = side / 2
box_left = int(gt_cx - half_side)
box_right = int(gt_cx + half_side)
box_top = int(gt_cy - half_side)
box_bottom = int(gt_cy + half_side)
## shift box horizontally if outside image boundaries
if box_left < 0:
    box_left = 0
    box_right = side
if box_right > img_width:
    box_right = img_width
    box_left = img_width - side
## shift box vertically if outside image boundaries
if box_top < 0:
    box_top = 0
    box_bottom = side
if box_bottom > img_height:
    box_bottom = img_height
    box_top = img_height - side
box_width = box_right - box_left
box_height = box_bottom - box_top
adj_bbox = (box_left, box_top, box_right, box_bottom)
print(adj_bbox)

# crop image
crop_image = image[box_top : box_bottom, box_left : box_right, ::-1].copy()
print(crop_image.shape, crop_image.dtype)

# resize image
resize_image : np.ndarray = cv2.resize(crop_image, (224, 224))
print(resize_image.shape, resize_image.dtype)

# normalize image
norm_image : np.ndarray = resize_image.astype(np.float32) / (2**bits-1)
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
    # create model
    lmo_model = mobilepose.MobilePose(keypoints=nkpts,
                                      normalized_coordinates=NORMALIZED_COORDS)
    lmo_model.load_weights(model_file)
except Exception:
    print("Model definition error!")
    raise
else:
    # infer float model
    inputs = (tensor, )
    outputs = lmo_model.post(lmo_model(*inputs))
    # first tensor is coords
    kpts = outputs[0].numpy()
    print(kpts)

    # second tensor is heatmaps
    heatmap_size = outputs[1].shape[2:]
    print(heatmap_size)
    if not NORMALIZED_COORDS:
        # convert coordinates from heatmap output size to image size
        kpts[..., 0] = (kpts[..., 0] + 0.5) * box_width / heatmap_size[0] + box_left - 0.5
        kpts[..., 1] = (kpts[..., 1] + 0.5) * box_height / heatmap_size[1] + box_top - 0.5
    else:
        # convert normalized coordinates to image size
        kpts[..., 0] = (kpts[..., 0] + 1) / 2 * box_width + box_left - 0.5
        kpts[..., 1] = (kpts[..., 1] + 1) / 2 * box_height + box_top - 0.5
    print(kpts)
    lmo_kpts = kpts.tolist()

# annotate the image
with Image.open(image_file) as im:
    draw = ImageDraw.Draw(im)
    # bounding boxes
    boxes = (adj_bbox, gt_bbox)
    labels = ("adj_bbox", "ground_truth")
    colors = ("orange", "green")
    width = 1
    for bbox, color, label in zip(boxes, colors, labels):
        draw.rectangle(bbox, width=width, outline=color)
        margin = width + 1
        draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=color)
    # ground truth keypoints
    colors = "green"
    radius = 3
    for kpt in gt_kpts:
        x1 = kpt[0] - radius
        x2 = kpt[0] + radius
        y1 = kpt[1] - radius
        y2 = kpt[1] + radius
        draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)
    # custom model keypoints
    colors = "red"
    radius = 2
    for kpt in lmo_kpts[0]:
        x1 = kpt[0] - radius
        x2 = kpt[0] + radius
        y1 = kpt[1] - radius
        y2 = kpt[1] + radius
        draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)
    # save the annotated image
    im.save(f"annotated_{image_filename}")

