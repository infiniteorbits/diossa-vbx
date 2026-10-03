"""Keras (TensorFlow 2) Fully Convolutional One-Stage Object Detection (FCOS) module.

Model implemented using Functional API.

Notes
-----
The implementation follows the original papers [1]_ and [2]_.

References
----------
.. [1] "FCOS: Fully Convolutional One-Stage Object Detection", https://arxiv.org/abs/1904.01355
.. [2] "FCOS: A Simple and Strong Anchor-free Object Detector", https://arxiv.org/abs/2006.09214

"""

from typing import Optional, Tuple, Sequence, List
from functools import partial
from abc import ABC

import tensorflow as tf
from tensorflow import keras

from ..resnet import resnet50
from ..fpn import fpn, FPNlayers, FPNstrides

from ..ops import box

from .object_detection import ObjectDetection


def fcos(tensor,
         num_classes: int = 2,
         num_convs: int = 4,
         norm_layer: keras.layers.Layer = keras.layers.GroupNormalization,
         num_groups: int = 32,
         filters: int = 256) -> List[List]:
    """FCOS model constructor.

    Submodels:
    1. ResNet50
    2. FPN

    Parameters
    ----------
    tensor : tensor
        Input tensor.
    num_classes : int, optional
        Number of classes to be produced.
    num_convs : int, optional
        Number of layers to add on the head.
    norm_layer : keras.layers normalization layer, optional
        Normalization layer to be used on the head.
    num_groups : int, optional
        Number of groups for the GroupNormalization.
    filters : int, optional
        Number of channels of each subnet.

    Returns
    -------
    reg_bboxes : list of tensor
        Regressor bounding boxes.
    cls_logits : list of tensor
        Classifier score logits.
    centerness : list of tensor
        Distance to the center of the object.
    """
    backbone = resnet50(tensor, output_layers=True)
    body = fpn(backbone, FPNlayers.P3toP6P7Conv)

    norm = partial(norm_layer, num_groups) if norm_layer is keras.layers.GroupNormalization else norm_layer

    cls_subnet = []
    for layer in range(num_convs):
        cls_subnet.append(keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", name=f"cls_layer{layer + 1}_conv"))
        cls_subnet.append(norm(name=f"cls_layer{layer + 1}_gn"))
        cls_subnet.append(keras.layers.ReLU(name=f"cls_layer{layer + 1}_relu"))
    cls_logits_conv = keras.layers.Conv2D(num_classes, kernel_size=3, strides=1, padding="same", name="cls_logits")
    
    reg_subnet = []
    for layer in range(num_convs):
        reg_subnet.append(keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", name=f"reg_layer{layer + 1}_conv"))
        reg_subnet.append(norm(name=f"reg_layer{layer + 1}_gn"))
        reg_subnet.append(keras.layers.ReLU(name=f"reg_layer{layer + 1}_relu"))

    reg_bbox_conv = keras.layers.Conv2D(4, kernel_size=3, strides=1, padding="same", name="reg_bbox_conv")
    reg_bbox_relu = keras.layers.ReLU(name="reg_bbox_relu")

    centerness_conv = keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="same", name="centerness")

    cls_logits = []
    reg_bboxes = []
    centerness = []
    for feature_map in body:
        x = feature_map
        for layer in cls_subnet:
            x = layer(x)
        cls_logits.append(cls_logits_conv(x))
        x = feature_map
        for layer in reg_subnet:
            x = layer(x)
        reg_bboxes.append(reg_bbox_relu(reg_bbox_conv(x)))
        centerness.append(centerness_conv(x))

    return [reg_bboxes, cls_logits, centerness]


class FCOS_ABC(ABC):
    """FCOS model abstract base class."""

    def _init(self, img_size: Tuple[int, int] = (800, 800),
              score_thresh: Optional[float] = 0.2, nms_thresh: float = 0.6,
              detections_per_img: int = 10, topk_candidates: int = 1000,
              normalize_boxsize: bool = False, dynamic_shapes: bool = False) -> None:
        """Initialization method,
        
        to be called in derived classes' constructors.

        Parameters
        ----------
        img_size : tuple of int
            Image size in (width, height) format.
        score_thresh : float, optional
            Box scoring threshold, applied before NMS.
        nms_thresh : float, optional
            NMS IoU threshold.
        detections_per_img : int, optional
            Number of (overall) best scoring detections to keep, after NMS.
        topk_candidates : int, optional
            Number of (per layer) best scoring detections to keep, before NMS.
        normalize_boxsize : bool, optional
            Whether the predicted box sizes are normalized or not.
        dynamic_shapes : bool, optional
            If False then the outputs will have a static shape (padded with zeros),
            otherwise (if True) the outputs will have a dynamic shape.
            Dynamic shapes should only be used for batchsize=1,
            for batches of multiple images runtime errors are likely to happen.
        """
        fpn_strides = FPNstrides.P3toP6P7Conv
        self.img_w, self.img_h = img_size
        self.locations = self.generate_locations(fpn_strides)
        self.strides = fpn_strides if normalize_boxsize else [None] * len(fpn_strides)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.dynamic_shapes = dynamic_shapes if dynamic_shapes else None

    def generate_locations(self,
                strides: Tuple[int, int, int, int, int]) \
                -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Generates the image pixel locations corresponding to the feature maps.

        Parameters
        ----------
        strides : tuple of int
            Stride of the feature map with respect to the input image, for each layer.

        Returns
        -------
        tuple of tf.Tensor
            Locations for each layer in (cx, cy) format.
        """
        locations = []
        for stride in strides:
            centers_x = tf.range(0, self.img_w, stride, dtype=tf.float32)
            centers_y = tf.range(0, self.img_h, stride, dtype=tf.float32)
            centers_y, centers_x = tf.meshgrid(centers_y, centers_x, indexing="ij")
            centers_x = tf.reshape(centers_x, [-1])
            centers_y = tf.reshape(centers_y, [-1])
            centers = tf.stack([centers_x, centers_y], axis=1)
            locations.append(centers)

        return tuple(locations)

    def post(self,
             reg_bboxes: Sequence[tf.Tensor],
             cls_logits: Sequence[tf.Tensor],
             centerness: Sequence[tf.Tensor]) \
             -> List:
        """Processes the predictions into detections.

        Parameters
        ----------
        reg_bboxes : sequence of tensor
            Regressor bounding boxes.
        cls_logits : sequence of tensor
            Classifier score logits.
        centerness : sequence of tensor
            Distance to the center of the object.

        Returns
        -------
        final_bboxes : tf.Tensor
            Bounding boxes, with batch as first dimension.
        final_scores : tf.Tensor
            Scores, with batch as first dimension.
        final_labels : tf.Tensor
            Labels, with batch as first dimension.
        """
        levels_bboxes = []
        levels_scores = []
        levels_labels = []
        for level_bboxes, level_logits, level_ctrnss, level_locations, level_stride in \
                zip(reg_bboxes, cls_logits, centerness, self.locations, self.strides):
            shape = tf.shape(level_logits)
            batchsize = shape[0]
            num_classes = shape[-1]
            # flatten, with major index as batch
            flat_logits = tf.reshape(level_logits, [batchsize, -1, num_classes])
            flat_bboxes = tf.reshape(level_bboxes, [batchsize, -1, 4])
            flat_ctrnss = tf.reshape(level_ctrnss, [batchsize, -1, 1])

            # calculate the scores
            calc_scores = tf.math.sqrt(tf.sigmoid(flat_logits) * tf.sigmoid(flat_ctrnss))
            # get the best score and correspondent label, per location
            max_scores = tf.reduce_max(calc_scores, axis=-1)
            max_labels = tf.argmax(calc_scores, axis=-1)

            batch_bboxes = tf.TensorArray(dtype=flat_bboxes.dtype,
                                          size=batchsize,
                                          infer_shape=False)
            batch_scores = tf.TensorArray(dtype=max_scores.dtype,
                                          size=batchsize,
                                          infer_shape=False)
            batch_labels = tf.TensorArray(dtype=max_labels.dtype,
                                          size=batchsize,
                                          infer_shape=False)
            for batch_idx in range(batchsize):
                img_bboxes = flat_bboxes[batch_idx]
                img_scores = max_scores[batch_idx]
                img_labels = max_labels[batch_idx]

                if self.score_thresh is not None:
                    # remove low scoring boxes
                    keep_mask = img_scores > self.score_thresh
                    kept_scores = tf.boolean_mask(img_scores, keep_mask)
                    kept_bboxes = tf.boolean_mask(img_bboxes, keep_mask)
                    kept_labels = tf.boolean_mask(img_labels, keep_mask)
                    kept_locs = tf.boolean_mask(level_locations, keep_mask)
                else:
                    kept_scores = img_scores
                    kept_bboxes = img_bboxes
                    kept_labels = img_labels
                    kept_locs = level_locations

                # keep only per level top scoring predictions, before NMS
                sort_idxs = tf.argsort(kept_scores, direction="DESCENDING")
                keep_idxs = sort_idxs[ :self.topk_candidates]
                top_scores = tf.gather(kept_scores, keep_idxs, axis=0, batch_dims=0)
                top_bboxes = tf.gather(kept_bboxes, keep_idxs, axis=0, batch_dims=0)
                top_labels = tf.gather(kept_labels, keep_idxs, axis=0, batch_dims=0)
                top_locs = tf.gather(kept_locs, keep_idxs, axis=0, batch_dims=0)

                # apply predicted boxes deltas to their correspondent location on the image
                pred_boxes = box.centerdistance_to_box(top_locs, top_bboxes, level_stride)

                batch_bboxes = batch_bboxes.write(batch_idx, pred_boxes)
                batch_scores = batch_scores.write(batch_idx, top_scores)
                batch_labels = batch_labels.write(batch_idx, top_labels)

            levels_bboxes.append(batch_bboxes)
            levels_scores.append(batch_scores)
            levels_labels.append(batch_labels)

        levels = len(cls_logits)
        overall_bboxes = tf.TensorArray(dtype=batch_bboxes.dtype,
                                        size=batchsize)
        overall_scores = tf.TensorArray(dtype=batch_scores.dtype,
                                        size=batchsize)
        overall_labels = tf.TensorArray(dtype=batch_labels.dtype,
                                        size=batchsize)
        for batch_idx in range(batchsize):
            # concatenate single image predictions resulting from the feature map levels
            concat_bboxes = tf.concat([levels_bboxes[lvl_idx].read(batch_idx)
                                       for lvl_idx in range(levels)], axis=0)
            concat_scores = tf.concat([levels_scores[lvl_idx].read(batch_idx)
                                       for lvl_idx in range(levels)], axis=0)
            concat_labels = tf.concat([levels_labels[lvl_idx].read(batch_idx)
                                       for lvl_idx in range(levels)], axis=0)

            # clip boxes to image size
            clip_boxes = box.clip(concat_bboxes, (self.img_w, self.img_h, ))

            classes_bboxes = []
            classes_scores = []
            classes_labels = []
            # start at class id one to ignore background label
            for class_idx in range(1, num_classes):
                class_mask = concat_labels == tf.cast(class_idx, tf.int64)
                class_bboxes = tf.boolean_mask(clip_boxes, class_mask, axis=0)
                class_scores = tf.boolean_mask(concat_scores, class_mask, axis=0)
                class_labels = tf.boolean_mask(concat_labels, class_mask, axis=0)

                # non-maximum suppression (NMS)
                keep_idxs = tf.image.non_max_suppression(class_bboxes, class_scores,
                                                         max_output_size=self.detections_per_img,
                                                         iou_threshold=self.nms_thresh)
                nms_bboxes = tf.gather(class_bboxes, keep_idxs, axis=0, batch_dims=0)
                nms_scores = tf.gather(class_scores, keep_idxs, axis=0, batch_dims=0)
                nms_labels = tf.gather(class_labels, keep_idxs, axis=0, batch_dims=0)

                classes_bboxes.append(nms_bboxes)
                classes_scores.append(nms_scores)
                classes_labels.append(nms_labels)

            image_bboxes = tf.concat(classes_bboxes, axis=0)
            image_scores = tf.concat(classes_scores, axis=0)
            image_labels = tf.concat(classes_labels, axis=0)

            # keep only overall top scoring predictions, after NMS
            sort_idxs = tf.argsort(image_scores, direction="DESCENDING")
            keep_idxs = sort_idxs[ :self.detections_per_img]
            top_scores = tf.gather(image_scores, keep_idxs, axis=0, batch_dims=0)
            top_bboxes = tf.gather(image_bboxes, keep_idxs, axis=0, batch_dims=0)
            top_labels = tf.gather(image_labels, keep_idxs, axis=0, batch_dims=0)

            if self.dynamic_shapes is None:
                # zero padding to the number of detections per image
                paddings = [[0, self.detections_per_img - tf.shape(top_scores)[0]], [0, 0]]
                pad_bboxes = tf.pad(top_bboxes, paddings)
                paddings = [[0, self.detections_per_img - tf.shape(top_scores)[0]]]
                pad_scores = tf.pad(top_scores, paddings)
                pad_labels = tf.pad(top_labels, paddings)

                overall_bboxes = overall_bboxes.write(batch_idx, pad_bboxes)
                overall_scores = overall_scores.write(batch_idx, pad_scores)
                overall_labels = overall_labels.write(batch_idx, pad_labels)
            else:
                overall_bboxes = overall_bboxes.write(batch_idx, top_bboxes)
                overall_scores = overall_scores.write(batch_idx, top_scores)
                overall_labels = overall_labels.write(batch_idx, top_labels)

        final_bboxes = overall_bboxes.stack()
        final_scores = overall_scores.stack()
        final_labels = overall_labels.stack()

        return [final_bboxes, final_scores, final_labels]


class FCOS_Edge(keras.Model, FCOS_ABC):
    """(Partial) FCOS model class, non quantizable (post-processing) operations.

    Source code can be embedded as post-processing of the edge model.

    Parameters
    ----------
    img_size : tuple of int
        Image size in (width, height) format.
    score_thresh : float, optional
        Box scoring threshold, applied before NMS.
    nms_thresh : float, optional
        NMS IoU threshold.
    detections_per_img : int, optional
        Number of (overall) best scoring detections to keep, after NMS.
    topk_candidates : int, optional
        Number of (per layer) best scoring detections to keep, before NMS.
    normalize_boxsize : bool, optional
        Whether the predicted box sizes are normalized or not.
    dynamic_shapes : bool, optional
        If False then the outputs will have a static shape (padded with zeros),
        otherwise (if True) the outputs will have a dynamic shape.
        Dynamic shapes should only be used for batchsize=1,
        for batches of multiple images runtime errors are likely to happen.
    """

    def __init__(self, img_size: Tuple[int, int],
                 score_thresh: Optional[float] = 0.2, nms_thresh: float = 0.6,
                 detections_per_img: int = 10, topk_candidates: int = 1000,
                 normalize_boxsize: bool = False, dynamic_shapes: bool = False) -> None:
        self._init(img_size,
                   score_thresh, nms_thresh,
                   detections_per_img, topk_candidates,
                   normalize_boxsize, dynamic_shapes)
        super().__init__(name="FCOS")

    def call(self, inputs, training=None, mask=None):
        return self.post(*inputs)


class FCOS(keras.Model, FCOS_ABC, ObjectDetection):
    """Fully Convolutional One-Stage Object Detection model class.

    Parameters
    ----------
    img_size : tuple of int
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    score_thresh : float, optional
        Box scoring threshold, applied before NMS.
    nms_thresh : float, optional
        NMS IoU threshold.
    detections_per_img : int, optional
        Number of (overall) best scoring detections to keep, after NMS.
    topk_candidates : int, optional
        Number of (per layer) best scoring detections to keep, before NMS.
    normalize_boxsize : bool, optional
        Whether the predicted box sizes are normalized or not.
    dynamic_shapes : bool, optional
        If False then the outputs will have a static shape (padded with zeros),
        otherwise (if True) the outputs will have a dynamic shape.
        Dynamic shapes should only be used for batchsize=1,
        for batches of multiple images runtime errors are likely to happen.
    """

    def __init__(self, img_size: Tuple[int, int], channels: int = 3, num_classes: int = 2,
                 score_thresh: Optional[float] = 0.2, nms_thresh: float = 0.6,
                 detections_per_img: int = 100, topk_candidates: int = 1000,
                 normalize_boxsize: bool = True, dynamic_shapes: bool = False) -> None:
        self._init(img_size,
                   score_thresh, nms_thresh,
                   detections_per_img, topk_candidates,
                   normalize_boxsize, dynamic_shapes)

        inp = keras.Input(shape=(*reversed(img_size), channels), name="img")
        # for quantization normalization must be BatchNormalization
        out = fcos(inp, num_classes)
        super().__init__(inputs=inp, outputs=out, name="FCOS")


class FCOS_Full(keras.Model, FCOS_ABC):
    """FCOS complete model class.

    Parameters
    ----------
    img_size : tuple of int
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    score_thresh : float, optional
        Box scoring threshold, applied before NMS.
    nms_thresh : float, optional
        NMS IoU threshold.
    detections_per_img : int, optional
        Number of (overall) best scoring detections to keep, after NMS.
    topk_candidates : int, optional
        Number of (per layer) best scoring detections to keep, before NMS.
    normalize_boxsize : bool, optional
        Whether the predicted box sizes are normalized or not.
    dynamic_shapes : bool, optional
        If False then the outputs will have a static shape (padded with zeros),
        otherwise (if True) the outputs will have a dynamic shape.
        Dynamic shapes should only be used for batchsize=1,
        for batches of multiple images runtime errors are likely to happen.
    """

    def __init__(self, img_size: Tuple[int, int],
                 channels: int = 3, num_classes: int = 2,
                 score_thresh: Optional[float] = 0.2, nms_thresh: float = 0.6,
                 detections_per_img: int = 100, topk_candidates: int = 1000,
                 normalize_boxsize: bool = True, dynamic_shapes: bool = False) -> None:
        self._init(img_size,
                   score_thresh, nms_thresh,
                   detections_per_img, topk_candidates,
                   normalize_boxsize, dynamic_shapes)

        inp = keras.Input(shape=(*reversed(img_size), channels), name="img")
        out = fcos(inp, num_classes)
        super().__init__(inputs=inp, outputs=out, name="FCOS")

    def call(self, inputs, training=None, mask=None):
        return self.post(*super().call(inputs, training, mask))

