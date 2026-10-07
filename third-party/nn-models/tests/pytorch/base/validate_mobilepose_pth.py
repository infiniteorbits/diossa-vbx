"""PyTorch MobilePose validation script.

Loads a trained OTS `MobilePose` model
and overlays the detected keypoints annotations on an image,
from both the OTS model and
a custom model with the same trained parameters.

Examples
--------
>>> python validate_mobilepose_pth.py model.pth image.png gt.json 8

An image will be saved in the current directory
with the same filename, but prefixed with "annotated_".
"""

import sys
import os.path
import json

import torch
import torchvision
import numpy as np
import cv2
import dsntnn

from nn_models.pytorch.keypoints_regression import mobilepose

from ots.mobilepose.network import CoordRegressionNetwork


if len(sys.argv) != 5:
    print(f"Usage: python {os.path.basename(sys.argv[0])} <model.pth> <image.png> <gt.json> <number_keypoints>")
    sys.exit()

model_file = sys.argv[1]
image_file = sys.argv[2]
gt_file = sys.argv[3]
nkpts = int(sys.argv[4])

NORMALIZED_COORDS = True

# inference device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# normalize image
norm_image : np.ndarray = crop_image.astype(np.float32) / (2**bits-1)
print(norm_image.dtype, norm_image.min(), norm_image.max())
# image shape is HWC

# preprocessing
transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
# image to tensor
tensor : torch.Tensor = transform(norm_image).to(device)
print(tensor.shape, tensor.dtype)
# tensor shape is CHW

# LMO model
print("# LMO model")
try:
    # create model
    lmo_model = mobilepose.MobilePose(keypoints=nkpts,
                                      normalized_coordinates=NORMALIZED_COORDS)
    lmo_model.load_ots_model(model_file)
    lmo_model.to(device)
except Exception:
    print("Model definition error!")
    raise
else:
    # set model in evaluation mode
    lmo_model.eval()
    # infer float model
    inputs = (tensor.unsqueeze(0), )
    with torch.no_grad():
        outputs = lmo_model.post(lmo_model(*inputs))
    # first tensor is coords
    kpts = outputs[0].clone().detach().cpu()
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
    lmo_kpts = kpts

    # set model in training mode
    lmo_model.train()
    # ground truth boxes
    target_kpts = torch.tensor([gt_kpts], dtype=torch.float32, device=device)
    # remove the offset due to cropping
    target_kpts[...,0] -= box_left
    target_kpts[...,1] -= box_top
    # scale the coordinates to the heatmap size
    if not NORMALIZED_COORDS:
        target_kpts = (target_kpts + 0.5) * heatmap_size[0] / box_height - 0.5
    else:
        target_kpts = (target_kpts + 0.5) * 2 / box_height - 1.0
    print(target_kpts)
    target_vis = torch.ones(target_kpts.shape[:-1], dtype=torch.uint8, device=device)
    # compute loss
    lmo_loss = lmo_model.compute_loss(*lmo_model.post(lmo_model(*inputs)), target_kpts, target_vis)
    lmo_loss = torch.mean(sum(lmo_loss.values()))
    print(lmo_loss)
    # test backward propagation
    lmo_loss.backward()

# OTS model
print("# OTS model")
try:
    # create model
    ots_model = CoordRegressionNetwork(nkpts, "mobilenetv2")
    ots_model.load_state_dict(torch.load(model_file, map_location="cpu"))
    ots_model.to(device)
except Exception:
    print("Model definition error!")
    raise
else:
    # set model in evaluation mode
    ots_model.eval()
    # infer float model
    inputs = (tensor.unsqueeze(0), )
    with torch.no_grad():
        outputs = ots_model(*inputs)
    # first tensor is coords
    kpts = outputs[0].clone().detach().cpu()
    print(kpts)

    # convert normalized coordinates to image size
    kpts[..., 0] = (kpts[..., 0] + 1) / 2 * box_width + box_left - 0.5
    kpts[..., 1] = (kpts[..., 1] + 1) / 2 * box_height + box_top - 0.5
    print(kpts)
    ots_kpts = kpts

    # set model in training mode
    ots_model.train()
    # ground truth boxes
    target_kpts = torch.tensor([gt_kpts], dtype=torch.float32, device=device)
    # remove the offset due to cropping
    target_kpts[...,0] -= box_left
    target_kpts[...,1] -= box_top
    # scale the coordinates to the heatmap size
    target_kpts = dsntnn.pixel_to_normalized_coordinates(target_kpts, (box_height, box_width))
    print(target_kpts)
    # compute loss
    outputs = ots_model(*inputs)
    e_loss = dsntnn.euclidean_losses(outputs[0], target_kpts)
    reg_losses = dsntnn.js_reg_losses(outputs[1], target_kpts, sigma_t=1.0)
    ots_loss = dsntnn.average_loss(e_loss + reg_losses)
    print(ots_loss)

# annotate the image
# torchvision utils only support 8 bit images
kpts_image = torch.from_numpy(cv2.imread(image_file)[..., ::-1].copy()).permute((2,0,1))
# bounding boxes
kpts_image = torchvision.utils.draw_bounding_boxes(
                kpts_image,
                torch.stack((torch.tensor(adj_bbox), torch.tensor(gt_bbox))),
                labels=["adj_bbox", "ground_truth"],
                colors=["orange", "green"],
                width=1
            )
# off the shelf keypoints
kpts_image = torchvision.utils.draw_keypoints(
                kpts_image,
                ots_kpts,
                colors="green",
                radius=3
            )
# custom model keypoints
kpts_image = torchvision.utils.draw_keypoints(
                kpts_image,
                lmo_kpts,
                colors="red",
                radius=2
            )
# save the annotated image
cv2.imwrite(f"annotated_{image_filename}", kpts_image.permute((1,2,0)).numpy()[..., ::-1])

# print inference kpts comparison
print("#######")
print(torch.eq(lmo_kpts, ots_kpts), torch.sub(lmo_kpts, ots_kpts))

# print loss comparison
print(torch.eq(lmo_loss, ots_loss).item(), torch.sub(lmo_loss, ots_loss).item())

