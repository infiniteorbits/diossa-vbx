"""PyTorch Faster R-CNN validation script.

Loads a trained Torchvision's `fasterrcnn_resnet50_fpn` model
and overlays the detected bounding boxes annotations on an image,
from both the Torchvision model and
a custom model with the same trained parameters.

Examples
--------
>>> python validate_fasterrcnn_pth.py model.pth image.png

An image will be saved in the current directory
with the same filename, but prefixed with "annotated_".
"""

import sys
import os.path

import torch
import torchvision
import numpy as np
import cv2

from nn_models.pytorch.object_detection import faster_rcnn


if len(sys.argv) != 3:
    print(f"Usage: python {os.path.basename(sys.argv[0])} <model.pth> <image.png>")
    sys.exit()

model_file = sys.argv[1]
image_file = sys.argv[2]

OTS_RESIZECROP = False

# inference device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    lmo_model = faster_rcnn.FasterRCNN(img_size=img_size)
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
        outputs = lmo_model(*inputs)
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
    tv_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = tv_model.roi_heads.box_predictor.cls_score.in_features
    tv_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
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
cv2.imwrite(f"annotated_{os.path.basename(image_file)}", box_image.permute((1,2,0)).numpy()[..., ::-1])

# print inference box comparison
print("#######")
print(torch.eq(lmo_boxes, tv_boxes), torch.sub(lmo_boxes, tv_boxes))
print(torch.eq(lmo_score, tv_score).item(), torch.sub(lmo_score, tv_score).item())
print(torch.eq(lmo_label, tv_label).item(), torch.sub(lmo_label, tv_label).item())

