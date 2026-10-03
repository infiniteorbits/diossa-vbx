"""TensorFlow 2 bounding box utilities module."""

from typing import Optional, Tuple
import math

import tensorflow as tf

def cxcywh_to_xyxy(boxes: tf.Tensor) -> tf.Tensor:
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

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)

    return boxes

def xyxy_to_cxcywh(boxes: tf.Tensor) -> tf.Tensor:
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

    boxes = tf.stack([cx, cy, w, h], axis=-1)

    return boxes

def area(boxes: tf.Tensor) -> tf.Tensor:
    """Calculate the area of bounding boxes.

    Bounding boxes in (x1, y1, x2, y2) format.
    """
    widths = boxes[..., 2] - boxes[..., 0]
    heights = boxes[..., 3] - boxes[..., 1]
    areas = widths * heights

    return areas

def clip(boxes: tf.Tensor, img_size: Tuple[int, int]) -> tf.Tensor:
    """Clip bounding boxes to the given image size.

    Bounding boxes in (x1, y1, x2, y2) format.
    """
    img_w, img_h = img_size

    xs = tf.clip_by_value(boxes[..., 0::2], clip_value_min=0, clip_value_max=img_w)
    ys = tf.clip_by_value(boxes[..., 1::2], clip_value_min=0, clip_value_max=img_h)

    clip_boxes = tf.reshape(tf.stack([xs, ys], axis=-1), tf.shape(boxes))

    return clip_boxes

def decode(ref_boxes: tf.Tensor, encodings: tf.Tensor,
           xform_weights: Optional[Tuple[float]] = None,
           xform_clip: float = math.log(1000.0 / 16)) -> tf.Tensor:
    """Decode bounding boxes.

    Apply encodings to reference bounding boxes
    to get decoded bounding boxes.

    All bounding boxes in (cx, cy, w, h) format.
    """
    ctrs_x = tf.expand_dims(ref_boxes[..., 0], axis=-1)
    ctrs_y = tf.expand_dims(ref_boxes[..., 1], axis=-1)
    widths = tf.expand_dims(ref_boxes[..., 2], axis=-1)
    heights = tf.expand_dims(ref_boxes[..., 3], axis=-1)

    # NOTE torchvision model includes weights for the transformers
    if xform_weights is None:
        dx = encodings[..., 0::4]
        dy = encodings[..., 1::4]
        dw = tf.clip_by_value(encodings[..., 2::4], clip_value_max=xform_clip)
        dh = tf.clip_by_value(encodings[..., 3::4], clip_value_max=xform_clip)
    else:
        (wx, wy, ww, wh) = xform_weights
        dx = tf.divide(encodings[..., 0::4], wx)
        dy = tf.divide(encodings[..., 1::4], wy)
        dw = tf.clip_by_value(tf.divide(encodings[..., 2::4], ww),
                              clip_value_max=xform_clip)
        dh = tf.clip_by_value(tf.divide(encodings[..., 3::4], wh),
                              clip_value_max=xform_clip)

    pred_x = dx * widths + ctrs_x
    pred_y = dy * heights + ctrs_y
    pred_w = tf.math.exp(dw) * widths
    pred_h = tf.math.exp(dh) * heights

    pred_boxes = tf.stack((pred_x, pred_y, pred_w, pred_h), axis=-1)

    return pred_boxes

def centerdistance_to_box(center_locations: tf.Tensor, distances: tf.Tensor,
                          scale: Optional[int] = None) -> tf.Tensor:
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

    pred_boxes = tf.stack([pred_left, pred_top, pred_right, pred_bottom], axis=-1)

    return pred_boxes

def box_to_centerdistance(center_locations: tf.Tensor, boxes: tf.Tensor,
                          scale: Optional[int] = None) -> tf.Tensor:
    """Compute distance from bounding boxes sides to the center locations.

    All bounding boxes in (x1, y1, x2, y2) format.
    """
    ctrs_x = center_locations[..., 0]
    ctrs_y = center_locations[..., 1]

    target_left = ctrs_x - boxes[..., 0]
    target_top = ctrs_y - boxes[..., 1]
    target_right = boxes[..., 2] - ctrs_x
    target_bottom = boxes[..., 3] - ctrs_y

    target_boxes = tf.stack([target_left, target_top, target_right, target_bottom], axis=-1)
    if scale is not None:
        target_boxes /= scale

    return target_boxes

def remove_small(boxes: tf.Tensor, min_size: float) -> tf.Tensor:
    """Keep only boxes with both sides larger then given size.

    Returns the indices of the boxes, not the boxes themselves.

    Bounding boxes in (x1, y1, x2, y2) format.
    """
    ws, hs = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    cond = (ws >= min_size) & (hs >= min_size)
    keep = tf.where(cond)
    return keep

