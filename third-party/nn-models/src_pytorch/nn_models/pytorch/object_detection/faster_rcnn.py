"""PyTorch Faster R-CNN module.

Model implemented using torch.nn.Module subclassing and Sequential containers.

Notes
-----
The implementation follows the original paper [1]_ (Faster R-CNN),
with the enhancements described in [2]_ (FPN).

References
----------
.. [1] "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", https://arxiv.org/abs/1506.01497
.. [2] "Feature Pyramid Networks for Object Detection", https://arxiv.org/abs/1612.03144

"""

from typing import Tuple, Optional
from collections import OrderedDict

import torch
import torchvision

from ..resnet import ResNet50Layers as ResNet50
from ..fpn import FPN_P2toP6MaxPool as FPN

from ..ops import box

from ..utils.loaders import load_torchvision_fasterrcnn

from .object_detection import ObjectDetection


class RPN(torch.nn.Module):
    """Region Proposal Network model class.

    Parameters
    ----------
    img_size : tuple of int
        Image size in (width, height) format.
    pyramid_strides : tuple of int
        Stride of the feature map with respect to the input image, for each layer.
    anchor_scales : tuple of int, optional
        Anchor scale, for each layer.
    anchor_ratios : tuple of float, optional
        Anchor aspect ratios, for all layers.
    xform_weights : tuple of float, optional
        Box transformer weights.
    pre_nms_top_n : int, optional
        Number of (per layer) best scoring boxes to keep, before NMS.
    post_nms_top_n : int, optional
        Number of (overall) best scoring boxes to keep, after NMS.
    nms_thresh : float, optional
        NMS IoU threshold.
    score_thresh : float, optional
        Box scoring threshold, applied before NMS.
    min_size : float, optional
        Box size threshold, applied before NMS.
    in_channels : int, optional
        Number of input channels to the model.
    head_channels : int, optional
        Number of channels fed into the classifier and regressor layers.
    """

    def __init__(self, img_size: Tuple[int, int],
                 pyramid_strides: Tuple[int, int, int, int, int],
                 anchor_scales: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
                 anchor_ratios: Tuple[float, float, float] = (0.5, 1., 2.),
                 xform_weights: Optional[Tuple[float]] = None,
                 pre_nms_top_n: int = 1000,
                 post_nms_top_n: int = 1000,
                 nms_thresh: float = 0.7,
                 score_thresh: Optional[float] = None,
                 min_size: Optional[float] = None,
                 in_channels: int = 256,
                 head_channels: int = 256) -> None:
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels, head_channels, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.reg_bbox = torch.nn.Conv2d(head_channels, 4 * len(anchor_ratios), kernel_size=1, stride=1)
        self.cls_logits = torch.nn.Conv2d(head_channels, len(anchor_ratios), kernel_size=1, stride=1)

        self.img_w, self.img_h = img_size
        anchors = self.generate_anchors(pyramid_strides, anchor_scales, anchor_ratios)
        # register `anchors` as buffer so .to(device) handles it
        for index, tensor in enumerate(anchors):
            # RPN uses pyramid levels 2 to 6
            self.register_buffer(f"anchors{index + 2}", tensor, False)
        
        self.xform_weights = xform_weights
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = min_size

    def forward(self,
            feature_maps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) \
            -> Tuple[
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                ]:
        """Defines the computation performed by the forward pass.

        Runs only the quantizable operations, without further post-processing.

        Parameters
        ----------
        feature_maps : tuple of torch.Tensor
            Backbone pyramid feature maps.

        Returns
        -------
        bboxes : tuple of torch.Tensor
            Regressor bounding box transformers, for each layer.
        scores : tuple of torch.Tensor
            Classifier score logits, for each layer.
        """
        (p2, p3, p4, p5, p6) = feature_maps

        feat2 = self.relu(self.conv(p2))
        bbox2 = self.reg_bbox(feat2)
        scor2 = self.cls_logits(feat2)

        feat3 = self.relu(self.conv(p3))
        bbox3 = self.reg_bbox(feat3)
        scor3 = self.cls_logits(feat3)

        feat4 = self.relu(self.conv(p4))
        bbox4 = self.reg_bbox(feat4)
        scor4 = self.cls_logits(feat4)

        feat5 = self.relu(self.conv(p5))
        bbox5 = self.reg_bbox(feat5)
        scor5 = self.cls_logits(feat5)

        feat6 = self.relu(self.conv(p6))
        bbox6 = self.reg_bbox(feat6)
        scor6 = self.cls_logits(feat6)

        bboxes = (bbox2, bbox3, bbox4, bbox5, bbox6)
        scores = (scor2, scor3, scor4, scor5, scor6)

        return bboxes, scores

    def generate_anchors(self,
                         strides: Tuple[int, int, int, int, int],
                         scales: Tuple[int, int, int, int, int],
                         ratios: Tuple[float, float, float]) \
                         -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the anchors for all the feature maps.

        Parameters
        ----------
        strides : tuple of int
            Stride of the feature map with respect to the input image, for each layer.
        scales : tuple of int
            Anchor scale, for each layer.
        ratios : tuple of float
            Anchor aspect ratios, for all layers.

        Returns
        -------
        tuple of torch.Tensor
            Anchors for each layer in (cx, cy, w, h) format.
        """
        aspect_ratios = torch.as_tensor(ratios, dtype=torch.float32)
        # assuming aspect ratio = height / width
        base_h = torch.sqrt(aspect_ratios)
        base_w = 1 / base_h

        anchors = []
        for stride, scale in zip(strides, scales):
            widths = base_w * scale
            heights = base_h * scale

            # base_anchors -> zero-centered anchors
            ## 'xyxy' format:
            # base_anchors = torch.round(torch.stack((-widths, -heights, widths, heights), dim=1) / 2)
            ## 'cxcywh' format:
            zero = torch.zeros_like(widths)
            base_anchors = torch.round(torch.stack((zero, zero, widths, heights), dim=1))

            centers_x = torch.arange(0, self.img_w, stride, dtype=torch.int32)
            centers_y = torch.arange(0, self.img_h, stride, dtype=torch.int32)
            centers_y, centers_x = torch.meshgrid(centers_y, centers_x, indexing="ij")
            centers_x = centers_x.flatten()
            centers_y = centers_y.flatten()
            ## 'xyxy' format:
            # centers = torch.stack((centers_x, centers_y, centers_x, centers_y), dim=1)
            ## 'cxcywh' format:
            zero = torch.zeros_like(centers_x)
            centers = torch.stack((centers_x, centers_y, zero, zero), dim=1)

            # anchors -> anchor centers + zero-centered anchors (base anchors)
            anchors.append((centers.unsqueeze(1) + base_anchors.unsqueeze(0)).flatten(0, 1))

        return tuple(anchors)

    def post(self,
             transformers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
             objectnesses: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) \
             -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates the proposals from the predictions and the anchors.

        Parameters
        ----------
        transformers : tuple of torch.Tensor
            Regressor bounding box transformers, for each layer.
        objectnesses : tuple of torch.Tensor
            Classifier score logits, for each layer.

        Returns
        -------
        proposal_bboxes : torch.Tensor
            Proposal bounding boxes, all batches concatenated.
            Batch information is embedded in the tensor data.
            Tensor[box index, batch index x1 y1 x2 y2]
        proposal_scores : torch.Tensor
            Proposal scores, all batches concatenated.
        """
        anchors = (self.anchors2, self.anchors3, self.anchors4, self.anchors5, self.anchors6)
        levels_bboxes = []
        levels_scores = []
        for level_trfs, level_objs, level_anchors in zip(transformers, objectnesses, anchors):
            batchsize, _, height, width = level_objs.shape
            # flatten transformers and objectnesses, with major index as batch
            flat_objs = level_objs.view(batchsize, -1, 1, height, width).permute(0, 3, 4, 1, 2).reshape(batchsize, -1)
            flat_trfs = level_trfs.view(batchsize, -1, 4, height, width).permute(0, 3, 4, 1, 2).reshape(batchsize, -1, 4)

            # decode boxes, applying transformers to anchors
            pred_boxes = box.cxcywh_to_xyxy(box.decode(level_anchors, flat_trfs, self.xform_weights).squeeze(-2))

            # keep only per level top scoring predictions, before NMS
            _, sort_idxs = torch.sort(flat_objs, descending=True)
            keep_idxs = sort_idxs[..., :self.pre_nms_top_n]
            top_objs = torch.gather(flat_objs, dim=1, index=keep_idxs)
            top_boxes = torch.gather(pred_boxes, dim=1, index=keep_idxs.unsqueeze(-1).expand(-1, -1, 4))
            top_scores = torch.sigmoid(top_objs)

            # clip boxes to image size
            clip_boxes = box.clip(top_boxes, (self.img_w, self.img_h, ))

            batch_bboxes = []
            batch_scores = []
            for batch_idx in range(batchsize):
                img_boxes = clip_boxes[batch_idx]
                img_scores = top_scores[batch_idx]

                if self.min_size is not None:
                    # remove small boxes
                    keep_idxs = box.remove_small(img_boxes, self.min_size)
                    kept_boxes = img_boxes[keep_idxs]
                    kept_scores = img_scores[keep_idxs]
                else:
                    kept_boxes = img_boxes
                    kept_scores = img_scores

                if self.score_thresh is not None:
                    # remove low scoring boxes
                    keep_idxs = torch.where(kept_scores > self.score_thresh)[0]
                    kept_boxes = kept_boxes[keep_idxs]
                    kept_scores = kept_scores[keep_idxs]

                # non-maximum suppression (NMS)
                keep_idxs = torchvision.ops.nms(kept_boxes, kept_scores, self.nms_thresh)
                nms_boxes = kept_boxes[keep_idxs]
                nms_scores = kept_scores[keep_idxs]
                batch_ids = torch.full_like(nms_boxes[:, :1], batch_idx)

                # proposals in roi format, which eliminates the need for padding
                # (in roi format the batch index is embedded in the tensor together with the box data)
                # standard boxes format -> [batch index, box index, x1 y1 x2 y2] ndim=3 shape=(N, C, 4)
                # roi boxes format -> [box index, batch index x1 y1 x2 y2] ndim=2 shape=(N*C, 5)

                batch_bboxes.append(torch.cat((batch_ids, nms_boxes), dim=1))
                batch_scores.append(nms_scores)

            levels_bboxes.append(batch_bboxes)
            levels_scores.append(batch_scores)

        levels = len(objectnesses)
        overall_bboxes = []
        overall_scores = []
        for batch_idx in range(batchsize):
            # concatenate single image proposals resulting from the feature map levels
            concat_boxes = torch.cat([levels_bboxes[lvl_idx][batch_idx] for lvl_idx in range(levels)])
            concat_scores = torch.cat([levels_scores[lvl_idx][batch_idx] for lvl_idx in range(levels)])

            # keep only overall top scoring predictions, after NMS
            _, sort_idxs = torch.sort(concat_scores, descending=True)
            keep_idxs = sort_idxs[ :self.post_nms_top_n]
            overall_scores.append(torch.gather(concat_scores, dim=0, index=keep_idxs))
            overall_bboxes.append(torch.gather(concat_boxes, dim=0, index=keep_idxs.unsqueeze(-1).expand(-1, 5)))

        proposal_bboxes = torch.cat(overall_bboxes)
        proposal_scores = torch.cat(overall_scores)

        return proposal_bboxes, proposal_scores


class FastRCNNDetector(torch.nn.Module):
    """Fast R-CNN detector model class.

    Parameters
    ----------
    img_size : tuple of int
        Image size in (width, height) format.
    pyramid_strides : tuple of int
        Stride of the feature map with respect to the input image, for each layer.
    num_classes : int, optional
        Number of classes to be produced.
    xform_weights : tuple of float, optional
        Box transformer weights.
    min_size : float, optional
        Box size threshold, applied before NMS.
    score_thresh : float, optional
        Box scoring threshold, applied before NMS.
    nms_thresh : float, optional
        NMS IoU threshold.
    detections_per_img : int, optional
        Number of best scoring detections to keep.
    in_channels : int, optional
        Number of input channels to the model.
    head_channels : int, optional
        Number of channels fed into the classifier and regressor layers.
    roi_output_size : tuple of int, optional
        RoI pool layer output size.

    Attributes
    ----------
    canonical_size : int
        Canonical ImageNet image size.
    canonical_level : int
        Target pyramid feature map level for the canonical image size.
    k_min : int
        Minimum pyramid feature map level.
    k_max : int
        Maximum pyramid feature map level.
    """

    canonical_size: int = 224
    canonical_level: int = 4
    k_min: int = 2
    k_max: int = 5

    def __init__(self, img_size: Tuple[int, int],
                 pyramid_strides: Tuple[int, int, int, int, int],
                 num_classes: int = 2,
                 xform_weights: Optional[Tuple[float]] = None,
                 min_size: Optional[float] = None,
                 score_thresh: Optional[float] = 0.05,
                 nms_thresh: float = 0.5,
                 detections_per_img: int = 100,
                 in_channels: int = 256,
                 head_channels: int = 1024,
                 roi_output_size: Tuple[int, int] = (7, 7)) -> None:
        super().__init__()

        self.hidden = torch.nn.Sequential(OrderedDict([
            ("flat", torch.nn.Flatten()),
            ("fc1", torch.nn.Linear(in_channels * roi_output_size[0] * roi_output_size[1], head_channels)),
            ("relu1", torch.nn.ReLU()),
            ("fc2", torch.nn.Linear(head_channels, head_channels)),
            ("relu2", torch.nn.ReLU())
            ]))
        self.reg_bbox = torch.nn.Linear(head_channels, 4 * num_classes)
        self.cls_logits = torch.nn.Linear(head_channels, num_classes)

        self.img_w, self.img_h = img_size
        self.xform_weights = xform_weights
        self.min_size = min_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.roi_size = roi_output_size
        self.spatial_scales = tuple([1/stride for stride in pyramid_strides[:-1]])

    def forward(self, rois: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Runs only the quantizable operations, without pre or post-processing.

        Parameters
        ----------
        rois : torch.Tensor
            RoI feature maps.

        Returns
        -------
        transformers : torch.Tensor
            Regressor bounding box transformers.
        objectnesses : torch.Tensor
            Classifier score logits.
        """
        features = self.hidden(rois)
        transformers = self.reg_bbox(features)
        objectnesses = self.cls_logits(features)

        return transformers, objectnesses

    def pre(self, proposals: torch.Tensor,
            feature_maps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) \
            -> torch.Tensor:
        """Processes the pyramid feature maps into RoI feature maps.

        Parameters
        ----------
        proposals : torch.Tensor
            Proposal bounding boxes, all batches concatenated.
            Tensor[box index, batch index x1 y1 x2 y2]
        feature_maps : tuple of torch.Tensor
            Backbone pyramid feature maps.

        Returns
        -------
        rois_features : torch.Tensor
            RoI feature maps.
        """
        # assign rois to feature map levels
        # Eqn.(1) in FPN paper
        boxes_area = box.area(proposals[:, 1:])
        boxes_size = torch.sqrt(boxes_area)
        target_lvls = torch.floor(self.canonical_level + torch.log2(boxes_size / self.canonical_size))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max).to(torch.int64) - self.k_min

        # a zeros tensor is created to be filled with the roi pool results
        # in order to keep the order of the results matching the proposals,
        # because the roi pooling is run per feature map level
        rois_features = torch.zeros(proposals.size(0), feature_maps[0].size(1), *self.roi_size, dtype=feature_maps[0].dtype, device=feature_maps[0].device)
        for lvl in range(self.k_max - self.k_min + 1):
            lvl_idxs = torch.where(target_lvls == lvl)[0]
            lvl_boxes = proposals[lvl_idxs]

            # torchvision.ops.roi_pool is the public documented function,
            # however it produces a TracerWarning, but can safely be ignored
            #
            # torch.ops.torchvision.roi_pool does not produce any warning,
            # but is probably better to not use it directly since it is not documented

            lvl_rois = torchvision.ops.roi_pool(feature_maps[lvl], lvl_boxes, self.roi_size, self.spatial_scales[lvl])
            # lvl_rois, _ = torch.ops.torchvision.roi_pool(feature_maps[lvl], lvl_boxes, self.spatial_scales[lvl], *self.roi_size)

            # NOTE torchvision model uses roi_align instead of roi_pool:
            # lvl_rois = torchvision.ops.roi_align(feature_maps[lvl], lvl_boxes, self.roi_size, self.spatial_scales[lvl], 2)

            rois_features[lvl_idxs] = lvl_rois.to(rois_features.dtype)

        return rois_features

    def post(self, proposals: torch.Tensor,
             transformers: torch.Tensor,
             objectnesses: torch.Tensor) \
             -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Processes the predictions into detections.

        Parameters
        ----------
        proposals : torch.Tensor
            Proposal bounding boxes, all batches concatenated.
            Tensor[box index, batch index x1 y1 x2 y2]
        transformers : torch.Tensor
            Regressor bounding box transformers.
        objectnesses : torch.Tensor
            Classifier score logits.

        Returns
        -------
        final_bboxes : torch.Tensor
            Bounding boxes, with batch as first dimension.
        final_scores : torch.Tensor
            Scores, with batch as first dimension.
        final_labels : torch.Tensor
            Labels, with batch as first dimension.
        """
        batch_ids = proposals[:, 0].unique()
        batchsize = batch_ids.size(0)
        num_classes = objectnesses.size(-1)

        # decode boxes, applying transformers to proposals
        pred_boxes = box.cxcywh_to_xyxy(box.decode(box.xyxy_to_cxcywh(proposals[:, 1:]), transformers, self.xform_weights))
        pred_scores = torch.nn.functional.softmax(objectnesses, dim=-1)

        # clip boxes to image size
        clip_boxes = box.clip(pred_boxes, (self.img_w, self.img_h, ))

        batches_bboxes = []
        batches_scores = []
        batches_labels = []
        for batch_idx in range(batchsize):
            batch_id = batch_ids[batch_idx]
            img_idxs = torch.where(proposals[:, 0] == batch_id)[0]

            img_boxes = clip_boxes[img_idxs]
            img_scores = pred_scores[img_idxs]

            # create labels for each prediction
            img_labels = torch.arange(num_classes, device=img_scores.device)
            img_labels = img_labels.unsqueeze(0).expand_as(img_scores)

            classes_boxes = []
            classes_scores = []
            classes_labels = []
            # start at class id one to ignore background label
            for class_idx in range(1, num_classes):
                class_boxes = img_boxes[:, class_idx]
                class_scores = img_scores[:, class_idx]
                class_labels = img_labels[:, class_idx]

                if self.score_thresh is not None:
                    # remove low scoring boxes
                    keep_idxs = torch.where(class_scores > self.score_thresh)[0]
                    kept_boxes = class_boxes[keep_idxs]
                    kept_scores = class_scores[keep_idxs]
                    kept_labels = class_labels[keep_idxs]
                else:
                    kept_boxes = class_boxes
                    kept_scores = class_scores
                    kept_labels = class_labels

                if self.min_size is not None:
                    # remove small boxes
                    keep_idxs = box.remove_small(kept_boxes, self.min_size)
                    kept_boxes = kept_boxes[keep_idxs]
                    kept_scores = kept_scores[keep_idxs]
                    kept_labels = kept_labels[keep_idxs]

                # non-maximum suppression (NMS)
                keep_idxs = torchvision.ops.nms(kept_boxes, kept_scores, self.nms_thresh)
                nms_boxes = kept_boxes[keep_idxs]
                nms_scores = kept_scores[keep_idxs]
                nms_labels = kept_labels[keep_idxs]

                classes_boxes.append(nms_boxes)
                classes_scores.append(nms_scores)
                classes_labels.append(nms_labels)

            batch_boxes = torch.cat(classes_boxes)
            batch_scores = torch.cat(classes_scores)
            batch_labels = torch.cat(classes_labels)

            # keep only overall top scoring predictions, after NMS
            _, sort_idxs = torch.sort(batch_scores, descending=True)
            keep_idxs = sort_idxs[ :self.detections_per_img]
            top_scores = torch.gather(batch_scores, dim=0, index=keep_idxs)
            top_bboxes = torch.gather(batch_boxes, dim=0, index=keep_idxs.unsqueeze(-1).expand(-1, 4))
            top_labels = torch.gather(batch_labels, dim=0, index=keep_idxs)

            # zero padding to the number of detections per image
            pad_boxes = torch.zeros(self.detections_per_img - top_bboxes.size(0), 4,
                                    dtype=top_bboxes.dtype, device=top_bboxes.device)
            pad_scores = torch.zeros(self.detections_per_img - top_scores.size(0),
                                    dtype=top_scores.dtype, device=top_scores.device)
            pad_labels = torch.zeros(self.detections_per_img - top_labels.size(0),
                                    dtype=top_labels.dtype, device=top_labels.device)

            batches_bboxes.append(torch.cat((top_bboxes, pad_boxes, )))
            batches_scores.append(torch.cat((top_scores, pad_scores, )))
            batches_labels.append(torch.cat((top_labels, pad_labels, )))

        final_bboxes = torch.stack(batches_bboxes)
        final_scores = torch.stack(batches_scores)
        final_labels = torch.stack(batches_labels)

        return final_bboxes, final_scores, final_labels


class FasterRCNN(torch.nn.Module, ObjectDetection):
    """Faster R-CNN model class.

    Submodels:
    1. ResNet50
    2. FPN
    3. RPN
    4. FastRCNNDetector

    Parameters
    ----------
    img_size : tuple of int
        Image size in (width, height) format.
    channels : int, optional
        Number of channels on the image.
    num_classes : int, optional
        Number of classes to be produced.
    anchor_scales : tuple of int, optional
        Anchor scale, for each layer.
    anchor_ratios : tuple of float, optional
        Anchor aspect ratios, for all layers.
    rpn_xform_weights : tuple of float, optional
        (RPN) Box transformer weights.
    rpn_pre_nms_top_n : int, optional
        (RPN) Number of (per layer) best scoring boxes to keep, before NMS.
    rpn_post_nms_top_n : int, optional
        (RPN) Number of (overall) best scoring boxes to keep, after NMS.
    rpn_nms_thresh : float, optional
        (RPN) NMS IoU threshold.
    rpn_score_thresh : float, optional
        (RPN) Box scoring threshold, applied before NMS.
    rpn_min_size : float, optional
        (RPN) Box size threshold, applied before NMS.
    det_xform_weights : tuple of float, optional
        (Fast R-CNN detector) Box transformer weights.
    det_min_size : float, optional
        (Fast R-CNN detector) Box size threshold, applied before NMS.
    det_score_thresh : float, optional
        (Fast R-CNN detector) Box scoring threshold, applied before NMS.
    det_nms_thresh : float, optional
        (Fast R-CNN detector) NMS IoU threshold.
    det_detections_per_img : int, optional
        (Fast R-CNN detector) Number of best scoring detections to keep.
    """

    # operation to be used for tensor addition
    # (if None then uses standard operator '+')
    _op_add = None

    def __init__(self, img_size: Tuple[int, int], channels: int = 3, num_classes: int = 2,
                 anchor_scales: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
                 anchor_ratios: Tuple[float, float, float] = (0.5, 1., 2.),
                 rpn_xform_weights: Optional[Tuple[float]] = None,
                 rpn_pre_nms_top_n: int = 1000,
                 rpn_post_nms_top_n: int = 1000,
                 rpn_nms_thresh: float = 0.7,
                 rpn_score_thresh: Optional[float] = None,
                 rpn_min_size: Optional[float] = 1e-3,
                 det_xform_weights: Optional[Tuple[float]] = (10.0, 10.0, 5.0, 5.0),
                 det_min_size: Optional[float] = 1e-2,
                 det_score_thresh: Optional[float] = 0.05,
                 det_nms_thresh: float = 0.5,
                 det_detections_per_img: int = 100) -> None:
        super().__init__()

        self.resnet = ResNet50(channels, op_add=self._op_add)
        self.fpn = FPN(self.resnet.filters, op_add=self._op_add)
        self.rpn = RPN(img_size, self.fpn.strides,
                       anchor_scales, anchor_ratios, rpn_xform_weights,
                       rpn_pre_nms_top_n, rpn_post_nms_top_n,
                       rpn_nms_thresh, rpn_score_thresh, rpn_min_size)
        self.detector = FastRCNNDetector(img_size, self.fpn.strides, num_classes,
                                         det_xform_weights, det_min_size,
                                         det_score_thresh, det_nms_thresh,
                                         det_detections_per_img)

    def forward(self, batch_imgs: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Parameters
        ----------
        batch_imgs : torch.Tensor
            Batch of images (NCHW).

        Returns
        -------
        bboxes : torch.Tensor
            Bounding boxes, with batch as first dimension.
        scores : torch.Tensor
            Scores, with batch as first dimension.
        labels : torch.Tensor
            Labels, with batch as first dimension.
        """
        feature_maps = self.resnet(batch_imgs)
        pyramid_feature_maps = self.fpn(feature_maps)

        rpn_transformers, rpn_objectnesses = self.rpn(pyramid_feature_maps)
        proposal_bboxes, proposal_scores = self.rpn.post(rpn_transformers, rpn_objectnesses)

        rois = self.detector.pre(proposal_bboxes, pyramid_feature_maps[:-1])
        rcnn_transformers, rcnn_objectnesses = self.detector(rois)
        bboxes, scores, labels = self.detector.post(proposal_bboxes, rcnn_transformers, rcnn_objectnesses)

        return bboxes, scores, labels

    def load_ots_model(self, model_file: str):
        """Loads parameters from an off-the-shelf model.

        Parameters
        ----------
        model_file : str
            Off-the-shelf model file path.
        """
        load_torchvision_fasterrcnn(self, model_file)


class FasterRCNN_Edge(FasterRCNN):
    """(Partial) Faster R-CNN model class, non quantizable (post-processing) operations.

    Overrides the forward path from `FasterRCNN` class
    to only include the non torch.nn operations.

    Source code can be embedded as post-processing of the edge model.
    """

    def forward(self,
            pyramid_feature_maps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            rpn_transformers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            rpn_objectnesses: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Parameters
        ----------
        pyramid_feature_maps : tuple of torch.Tensor
            FPN pyramid feature maps.
        rpn_transformers : tuple of torch.Tensor
            RPN proposal transformers.
        rpn_objectnesses : tuple of torch.Tensor
            RPN proposal objectnesses.

        Returns
        -------
        bboxes : torch.Tensor
            Bounding boxes, with batch as first dimension.
        scores : torch.Tensor
            Scores, with batch as first dimension.
        labels : torch.Tensor
            Labels, with batch as first dimension.
        """
        proposal_bboxes, proposal_scores = self.rpn.post(rpn_transformers, rpn_objectnesses)

        rois = self.detector.pre(proposal_bboxes, pyramid_feature_maps)
        # NOTE the detector head could eventually also be quantized,
        # but that would mean splitting the model in 4 parts
        rcnn_transformers, rcnn_objectnesses = self.detector(rois)
        bboxes, scores, labels = self.detector.post(proposal_bboxes, rcnn_transformers, rcnn_objectnesses)

        return bboxes, scores, labels

