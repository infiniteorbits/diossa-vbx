"""PyTorch FCOS validation script.

Loads a trained Torchvision's `fcos_resnet50_fpn` model
and overlays the detected bounding boxes annotations on an image,
from both the Torchvision model and
a custom model with the same trained parameters.

Examples
--------
>>> python validate_fcos_pth.py model.pth image.png [gt.json]

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

from nn_models.pytorch.object_detection import fcos


if len(sys.argv) < 3 or len(sys.argv) > 4:
    print(f"Usage: python {os.path.basename(sys.argv[0])} <model.pth> <image.png> [<gt.json>]")
    sys.exit()

model_file = sys.argv[1]
image_file = sys.argv[2]
gt_file = sys.argv[3] if len(sys.argv) == 4 else None

OTS_RESIZECROP = False
OTS_TRAINING = False

# inference device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read ground truth
image_filename = os.path.basename(image_file)
if gt_file is not None:
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

# normalize image
norm_image : np.ndarray = image[..., ::-1].copy().astype(np.float32) / (2**bits-1)
print(norm_image.dtype, norm_image.min(), norm_image.max())
# image shape is HWC

# LMO model
print("# LMO model")
try:
    # preprocessing
    transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(800),
                torchvision.transforms.CenterCrop([800, 1056]),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    # image to tensor
    tensor : torch.Tensor = transform(norm_image).to(device)
    print(tensor.shape, tensor.dtype)
    # tensor shape is CHW
    img_size = (tensor.size(2), tensor.size(1)) # W, H
    # create model
    lmo_model = fcos.FCOS(img_size=img_size,
                          num_classes=2 if not OTS_TRAINING else 91,
                          dynamic_shapes=True)
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
        outputs = lmo_model.post(*lmo_model(*inputs))
    # first tensor is boxes
    # first box is top detection
    boxes = outputs[0].clone().detach().cpu()[0,0:1]
    print(boxes)

    # scale ratio based on the smaller size, the height
    scale = image.shape[0] / tensor.size(1)
    print(scale)
    # add the offset due to cropping
    boxes[:,0::2] += (image.shape[1]/scale - tensor.size(2)) / 2
    # scale the coordinates to the original image size
    boxes *= scale
    print(boxes)
    lmo_boxes = boxes

    # score
    lmo_score = outputs[1].clone().detach().cpu()[0,0]
    print(lmo_score)
    # label
    lmo_label = outputs[2].clone().detach().cpu()[0,0]
    print(lmo_label)

    if gt_file is not None:
        # set model in training mode
        lmo_model.train()
        # ground truth boxes
        target_boxes = torch.tensor([gt_bbox], dtype=torch.float32, device=device)
        # remove the offset due to cropping
        target_boxes[:,0::2] -= (image.shape[1]/scale - tensor.size(2)) / 2
        # scale the coordinates to the resized image size
        target_boxes /= scale
        # ground truth labels
        target_labels = torch.tensor([gt_label], dtype=torch.int64, device=device)
        print(target_boxes, target_labels)
        # compute loss
        lmo_loss = lmo_model.compute_loss(*lmo_model(*inputs), [target_boxes], [target_labels])
        print(lmo_loss)
        # test backward propagation
        loss = sum(lmo_loss.values()) / len(lmo_loss)
        loss.backward()

# Torchvision model
print("# Torchvision model")
try:
    # preprocessing
    transforms = [torchvision.transforms.ToTensor()]
    if OTS_RESIZECROP:
        transforms += [torchvision.transforms.Resize(800),
                       torchvision.transforms.CenterCrop([800, 1056])]
    transform = torchvision.transforms.Compose(transforms)
    # image to tensor
    tensor : torch.Tensor = transform(norm_image).to(device)
    print(tensor.shape, tensor.dtype, tensor.min(), tensor. max())
    # tensor shape is CHW

    # create model
    tv_model = torchvision.models.detection.fcos_resnet50_fpn(weights=None, weights_backbone=None)
    tv_model.load_state_dict(torch.load(model_file, map_location="cpu"))
    tv_model.to(device)
except Exception:
    print("Model definition error!")
    raise
else:
    # set model in evaluation mode
    tv_model.eval()
    # infer float model
    inputs = (tensor, )
    with torch.no_grad():
        outputs = tv_model(inputs)
    print(outputs)
    boxes = outputs[0]["boxes"].clone().detach().cpu()[:1]
    # only needed if crop or resize is done outside of the model
    if OTS_RESIZECROP:
        scale = image.shape[0] / tensor.size(1)
        print(scale)
        boxes[:,0::2] += (image.shape[1]/scale - tensor.size(2)) / 2
        boxes *= scale
        print(boxes)
    tv_boxes = boxes

    # score
    tv_score = outputs[0]["scores"].clone().detach().cpu()[0]
    # label
    tv_label = outputs[0]["labels"].clone().detach().cpu()[0]

    if gt_file is not None:
        # set model in training mode
        tv_model.train()
        # only needed if crop or resize is done outside of the model
        if OTS_RESIZECROP:
            # ground truth boxes
            target_boxes = torch.tensor([gt_bbox], dtype=torch.float32, device=device)
            # remove the offset due to cropping
            target_boxes[:,0::2] -= (image.shape[1]/scale - tensor.size(2)) / 2
            # scale the coordinates to the resized image size
            target_boxes /= scale
            # ground truth labels
            target_labels = torch.tensor([gt_label], dtype=torch.int64, device=device)
            # ground truth target
            target = [{"boxes":target_boxes,
                    "labels":target_labels}]
        else:
            # ground truth target
            target = [{"boxes":torch.tensor([gt_bbox], dtype=torch.float32, device=device),
                    "labels":torch.tensor([gt_label], dtype=torch.int64, device=device)}]
        print(target)
        # compute loss
        tv_loss = tv_model(inputs, target)
        print(tv_loss)

# concatenate the bounding boxes
boxes = torch.cat((tv_boxes, lmo_boxes))

# annotate the image
# torchvision utils only support 8 bit images
box_image = torchvision.utils.draw_bounding_boxes(
                torch.from_numpy(cv2.imread(image_file)[..., ::-1].copy()).permute((2,0,1)),
                boxes,
                labels=["Torchvision", "LMO"],
                colors=["green", "red"],
                width=1
            )
# save the annotated image
cv2.imwrite(f"annotated_{image_filename}", box_image.permute((1,2,0)).numpy()[..., ::-1])

# print inference box comparison
print("#######")
print(torch.eq(lmo_boxes, tv_boxes), torch.sub(lmo_boxes, tv_boxes))
print(torch.eq(lmo_score, tv_score).item(), torch.sub(lmo_score, tv_score).item())
print(torch.eq(lmo_label, tv_label).item(), torch.sub(lmo_label, tv_label).item())

if gt_file is not None:
    # print loss comparison
    print(torch.eq(lmo_loss["classification"], tv_loss["classification"]).item(), torch.sub(lmo_loss["classification"], tv_loss["classification"]).item())
    print(torch.eq(lmo_loss["bbox_regression"], tv_loss["bbox_regression"]).item(), torch.sub(lmo_loss["bbox_regression"], tv_loss["bbox_regression"]).item())
    print(torch.eq(lmo_loss["bbox_ctrness"], tv_loss["bbox_ctrness"]).item(), torch.sub(lmo_loss["bbox_ctrness"], tv_loss["bbox_ctrness"]).item())

