"""PyTorch bounding box utilities module."""

import torch
import math

from typing import Tuple, Optional

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    """
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    hw = 0.5 * w
    hh = 0.5 * h
    x1 = cx - hw
    y1 = cy - hh
    x2 = cx + hw
    y2 = cy + hh

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)

    return boxes

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.

    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    """
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]

    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h

    boxes = torch.stack((cx, cy, w, h), dim=-1)

    return boxes

def area(boxes: torch.Tensor) -> torch.Tensor:
    """Calculate the area of bounding boxes.

    Bounding boxes in (x1, y1, x2, y2) format.
    """
    widths = boxes[..., 2] - boxes[..., 0]
    heights = boxes[..., 3] - boxes[..., 1]
    areas = widths * heights

    return areas

def clip(boxes: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
    """Clip bounding boxes to the given image size.

    Bounding boxes in (x1, y1, x2, y2) format.
    """
    img_w, img_h = img_size

    xs = torch.clamp(boxes[..., 0::2], min=0, max=img_w)
    ys = torch.clamp(boxes[..., 1::2], min=0, max=img_h)

    clip_boxes = torch.stack((xs, ys), dim=-1).reshape(boxes.shape)

    return clip_boxes

def decode(ref_boxes: torch.Tensor, encodings: torch.Tensor,
            xform_weights: Optional[Tuple[float]] = None,
            xform_clip: float = math.log(1000.0 / 16)) -> torch.Tensor:
    """Decode bounding boxes.

    Apply encodings to reference bounding boxes
    to get decoded bounding boxes.

    All bounding boxes in (cx, cy, w, h) format.
    """
    ctrs_x = ref_boxes[..., 0].unsqueeze(-1)
    ctrs_y = ref_boxes[..., 1].unsqueeze(-1)
    widths = ref_boxes[..., 2].unsqueeze(-1)
    heights = ref_boxes[..., 3].unsqueeze(-1)

    # NOTE torchvision model includes weights for the transformers
    if xform_weights is None:
        dx = encodings[..., 0::4]
        dy = encodings[..., 1::4]
        dw = encodings[..., 2::4].clamp(max=xform_clip)
        dh = encodings[..., 3::4].clamp(max=xform_clip)
    else:
        (wx, wy, ww, wh) = xform_weights
        dx = encodings[..., 0::4].div(wx)
        dy = encodings[..., 1::4].div(wy)
        dw = encodings[..., 2::4].div(ww).clamp(max=xform_clip)
        dh = encodings[..., 3::4].div(wh).clamp(max=xform_clip)

    pred_x = dx * widths + ctrs_x
    pred_y = dy * heights + ctrs_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.stack((pred_x, pred_y, pred_w, pred_h), dim=-1)

    return pred_boxes

def centerdistance_to_box(center_locations: torch.Tensor, distances: torch.Tensor,
          scale: Optional[int] = None) -> torch.Tensor:
    """Compute bounding boxes from center locations and distances to center.

    All bounding boxes in (x1, y1, x2, y2) format.
    """
    ctrs_x = center_locations[..., 0]
    ctrs_y = center_locations[..., 1]

    if scale is not None:
        distances = distances * scale

    pred_left = ctrs_x - distances[..., 0]
    pred_top = ctrs_y - distances[..., 1]
    pred_right = ctrs_x + distances[..., 2]
    pred_bottom = ctrs_y + distances[..., 3]

    pred_boxes = torch.stack((pred_left, pred_top, pred_right, pred_bottom), dim=-1)

    return pred_boxes

def box_to_centerdistance(center_locations: torch.Tensor, boxes: torch.Tensor,
          scale: Optional[int] = None) -> torch.Tensor:
    """Compute distance from bounding boxes sides to the center locations.

    All bounding boxes in (x1, y1, x2, y2) format.
    """
    ctrs_x = center_locations[..., 0]
    ctrs_y = center_locations[..., 1]

    target_left = ctrs_x - boxes[..., 0]
    target_top = ctrs_y - boxes[..., 1]
    target_right = boxes[..., 2] - ctrs_x
    target_bottom = boxes[..., 3] - ctrs_y

    target_boxes = torch.stack((target_left, target_top, target_right, target_bottom), dim=-1)
    if scale is not None:
        target_boxes /= scale

    return target_boxes

def remove_small(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    """Keep only boxes with both sides larger then given size.

    Returns the indices of the boxes, not the boxes themselves.

    Bounding boxes in (x1, y1, x2, y2) format.
    """
    ws, hs = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    cond = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(cond)[0]
    return keep

