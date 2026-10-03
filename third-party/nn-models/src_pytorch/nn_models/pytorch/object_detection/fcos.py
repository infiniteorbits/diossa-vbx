"""PyTorch Fully Convolutional One-Stage Object Detection (FCOS) module.

Model implemented using torch.nn.Module subclassing and Sequential containers.

Notes
-----
The implementation follows the original papers [1]_ and [2]_.

References
----------
.. [1] "FCOS: Fully Convolutional One-Stage Object Detection", https://arxiv.org/abs/1904.01355
.. [2] "FCOS: A Simple and Strong Anchor-free Object Detector", https://arxiv.org/abs/2006.09214

"""

from typing import Optional, Tuple, Sequence, Dict
from collections import OrderedDict
from functools import partial

import torch
import torchvision

from ..resnet import ResNet50Layers as ResNet50
from ..fpn import FPN_P3toP6P7Conv as FPN

from ..ops import box

from ..utils.loaders import load_torchvision_fcos

from .object_detection import ObjectDetection


class FCOS_Edge(torch.nn.Module):
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
        super().__init__()

        self.img_w, self.img_h = img_size
        locations = self.generate_locations(FPN.strides)
        # register `locations` as buffer so .to(device) handles it
        for index, tensor in enumerate(locations):
            # FCOS uses pyramid levels 3 to 7
            self.register_buffer(f"locations{index + 3}", tensor, False)

        self.strides = FPN.strides if normalize_boxsize else [None] * len(FPN.strides)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.dynamic_shapes = dynamic_shapes if dynamic_shapes else None

    def forward(self,
                reg_bboxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                cls_logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                centerness: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) \
                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Runs only the post-processing operations.

        Parameters
        ----------
        reg_bboxes : tuple of torch.Tensor
            Regressor bounding boxes.
        cls_logits : tuple of torch.Tensor
            Classifier score logits.
        centerness : tuple of torch.Tensor
            Distance to the center of the object.

        Returns
        -------
        bboxes : torch.Tensor
            Bounding boxes, with batch as first dimension.
        scores : torch.Tensor
            Scores, with batch as first dimension.
        labels : torch.Tensor
            Labels, with batch as first dimension.
        """
        return self.post(reg_bboxes, cls_logits, centerness)

    def generate_locations(self,
                strides: Tuple[int, int, int, int, int]) \
                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates the image pixel locations corresponding to the feature maps.

        Parameters
        ----------
        strides : tuple of int
            Stride of the feature map with respect to the input image, for each layer.

        Returns
        -------
        tuple of torch.Tensor
            Locations for each layer in (cx, cy) format.
        """
        locations = []
        for stride in strides:
            centers_x = torch.arange(0, self.img_w, stride, dtype=torch.int32)
            centers_y = torch.arange(0, self.img_h, stride, dtype=torch.int32)
            centers_y, centers_x = torch.meshgrid(centers_y, centers_x, indexing="ij")
            centers_x = centers_x.flatten()
            centers_y = centers_y.flatten()
            centers = torch.stack((centers_x, centers_y), dim=1)
            locations.append(centers)

        return tuple(locations)

    def post(self,
             reg_bboxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
             cls_logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
             centerness: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) \
             -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Processes the predictions into detections.

        Parameters
        ----------
        reg_bboxes : tuple of torch.Tensor
            Regressor bounding boxes.
        cls_logits : tuple of torch.Tensor
            Classifier score logits.
        centerness : tuple of torch.Tensor
            Distance to the center of the object.

        Returns
        -------
        final_bboxes : torch.Tensor
            Bounding boxes, with batch as first dimension.
        final_scores : torch.Tensor
            Scores, with batch as first dimension.
        final_labels : torch.Tensor
            Labels, with batch as first dimension.
        """
        locations = (self.locations3, self.locations4, self.locations5, self.locations6, self.locations7)
        levels_bboxes = []
        levels_scores = []
        levels_labels = []
        for level_bboxes, level_logits, level_ctrnss, level_locations, level_stride in \
                zip(reg_bboxes, cls_logits, centerness, locations, self.strides):
            batchsize, num_classes = level_logits.shape[:2]
            # flatten, with major index as batch
            flat_logits = level_logits.permute(0, 2, 3, 1).reshape(batchsize, -1, num_classes)
            flat_bboxes = level_bboxes.permute(0, 2, 3, 1).reshape(batchsize, -1, 4)
            flat_ctrnss = level_ctrnss.permute(0, 2, 3, 1).reshape(batchsize, -1, 1)

            # calculate the scores
            calc_scores = torch.sqrt(torch.sigmoid(flat_logits) * torch.sigmoid(flat_ctrnss))
            # get the best score and correspondent label, per location
            max_scores, max_labels = torch.max(calc_scores, dim=-1)

            batch_bboxes = []
            batch_scores = []
            batch_labels = []
            for batch_idx in range(batchsize):
                img_bboxes = flat_bboxes[batch_idx]
                img_scores = max_scores[batch_idx]
                img_labels = max_labels[batch_idx]

                if self.score_thresh is not None:
                    # remove low scoring boxes
                    keep_mask = img_scores > self.score_thresh
                    kept_scores = img_scores[keep_mask]
                    kept_bboxes = img_bboxes[keep_mask]
                    kept_labels = img_labels[keep_mask]
                    kept_locs = level_locations[keep_mask]
                else:
                    kept_scores = img_scores
                    kept_bboxes = img_bboxes
                    kept_labels = img_labels
                    kept_locs = level_locations

                # keep only per level top scoring predictions, before NMS
                _, sort_idxs = torch.sort(kept_scores, descending=True)
                keep_idxs = sort_idxs[ :self.topk_candidates]
                top_scores = kept_scores[keep_idxs]
                top_bboxes = kept_bboxes[keep_idxs]
                top_labels = kept_labels[keep_idxs]
                top_locs = kept_locs[keep_idxs]

                # apply predicted boxes deltas to their correspondent location on the image
                pred_boxes = box.centerdistance_to_box(top_locs, top_bboxes, level_stride)

                batch_bboxes.append(pred_boxes)
                batch_scores.append(top_scores)
                batch_labels.append(top_labels)

            levels_bboxes.append(batch_bboxes)
            levels_scores.append(batch_scores)
            levels_labels.append(batch_labels)

        levels = len(cls_logits)
        overall_bboxes = []
        overall_scores = []
        overall_labels = []
        for batch_idx in range(batchsize):
            # concatenate single image predictions resulting from the feature map levels
            concat_bboxes = torch.cat([levels_bboxes[lvl_idx][batch_idx] for lvl_idx in range(levels)])
            concat_scores = torch.cat([levels_scores[lvl_idx][batch_idx] for lvl_idx in range(levels)])
            concat_labels = torch.cat([levels_labels[lvl_idx][batch_idx] for lvl_idx in range(levels)])

            # clip boxes to image size
            clip_boxes = box.clip(concat_bboxes, (self.img_w, self.img_h, ))

            classes_bboxes = []
            classes_scores = []
            classes_labels = []
            # start at class id one to ignore background label
            for class_idx in range(1, num_classes):
                class_idxs = concat_labels == class_idx
                class_bboxes = clip_boxes[class_idxs]
                class_scores = concat_scores[class_idxs]
                class_labels = concat_labels[class_idxs]

                # non-maximum suppression (NMS)
                keep_idxs = torchvision.ops.nms(class_bboxes, class_scores, self.nms_thresh)
                nms_bboxes = class_bboxes[keep_idxs]
                nms_scores = class_scores[keep_idxs]
                nms_labels = class_labels[keep_idxs]

                classes_bboxes.append(nms_bboxes)
                classes_scores.append(nms_scores)
                classes_labels.append(nms_labels)

            image_bboxes = torch.cat(classes_bboxes)
            image_scores = torch.cat(classes_scores)
            image_labels = torch.cat(classes_labels)

            # keep only overall top scoring predictions, after NMS
            _, sort_idxs = torch.sort(image_scores, descending=True)
            keep_idxs = sort_idxs[ :self.detections_per_img]
            top_scores = image_scores[keep_idxs]
            top_bboxes = image_bboxes[keep_idxs]
            top_labels = image_labels[keep_idxs]

            if self.dynamic_shapes is None:
                # zero padding to the number of detections per image
                pad_bboxes = torch.zeros(self.detections_per_img, 4,
                                         dtype=top_bboxes.dtype, device=top_bboxes.device)
                pad_scores = torch.zeros(self.detections_per_img,
                                         dtype=top_scores.dtype, device=top_scores.device)
                pad_labels = torch.zeros(self.detections_per_img,
                                         dtype=top_labels.dtype, device=top_labels.device)
                detections = top_scores.size(0)
                pad_bboxes[:detections] = top_bboxes
                pad_scores[:detections] = top_scores
                pad_labels[:detections] = top_labels

                overall_bboxes.append(pad_bboxes)
                overall_scores.append(pad_scores)
                overall_labels.append(pad_labels)
            else:
                overall_bboxes.append(top_bboxes)
                overall_scores.append(top_scores)
                overall_labels.append(top_labels)

        final_bboxes = torch.stack(overall_bboxes)
        final_scores = torch.stack(overall_scores)
        final_labels = torch.stack(overall_labels)

        return final_bboxes, final_scores, final_labels


class FCOS(FCOS_Edge, ObjectDetection):
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
    center_sampling_radius: float, optional
        Radius used in the pairing of the ground truth boxes centers
        with the locations on the feature maps.
    dynamic_shapes : bool, optional
        If False then the outputs will have a static shape (padded with zeros),
        otherwise (if True) the outputs will have a dynamic shape.
        Dynamic shapes should only be used for batchsize=1,
        for batches of multiple images runtime errors are likely to happen.

    Attributes
    ----------
    num_convs : int, default=4
        Number of layers to add on the head.
    norm_layer : torch.nn normalization layer, default=GroupNorm
        Normalization layer to be used on the head.
    num_groups : int, default=32
        Number of groups for the GroupNorm.
    """

    num_convs: int = 4
    norm_layer = torch.nn.GroupNorm
    num_groups: int = 32

    # operation to be used for tensor addition
    # (if None then uses standard operator '+')
    _op_add = None

    def __init__(self, img_size: Tuple[int, int], channels: int = 3, num_classes: int = 2,
                 score_thresh: Optional[float] = 0.2, nms_thresh: float = 0.6,
                 detections_per_img: int = 100, topk_candidates: int = 1000,
                 normalize_boxsize: bool = True, center_sampling_radius: float = 1.5,
                 dynamic_shapes: bool = False) -> None:
        super().__init__(img_size, score_thresh, nms_thresh,
                         detections_per_img, topk_candidates,
                         normalize_boxsize, dynamic_shapes)
        self.resnet = ResNet50(channels, op_add=self._op_add)
        self.fpn = FPN(self.resnet.filters, op_add=self._op_add)

        channels = self.fpn.channels
        norm = partial(self.norm_layer, self.num_groups) if self.norm_layer is torch.nn.GroupNorm else self.norm_layer
        self.cls_subnet = torch.nn.Sequential()
        self.cls_logits = torch.nn.Conv2d(channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.reg_subnet = torch.nn.Sequential()
        self.reg_bbox = torch.nn.Sequential(OrderedDict([
            ("conv", torch.nn.Conv2d(channels, 4, kernel_size=3, stride=1, padding=1)),
            ("relu", torch.nn.ReLU())
            ]))
        self.centerness = torch.nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1)
        for layer in range(self.num_convs):
            self.cls_subnet.add_module(f"layer{layer + 1}" , torch.nn.Sequential(OrderedDict([
                ("conv", torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)),
                ("gn", norm(channels)),
                ("relu", torch.nn.ReLU())
                ])))
            self.reg_subnet.add_module(f"layer{layer + 1}" , torch.nn.Sequential(OrderedDict([
                ("conv", torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)),
                ("gn", norm(channels)),
                ("relu", torch.nn.ReLU())
                ])))

        self.center_sampling_radius = center_sampling_radius

    def forward(self, batch_imgs: torch.Tensor) \
            -> Tuple[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Defines the computation performed by the forward pass.

        Runs only the quantizable operations, without further post-processing.

        Parameters
        ----------
        batch_imgs : torch.Tensor
            Batch of images (NCHW).

        Returns
        -------
        reg_bboxes : tuple of torch.Tensor
            Bounding boxes shift predictions, with batch as first dimension
        cls_logits : tuple of torch.Tensor
            Class logits predictions, with batch as first dimension.
        centerness : tuple of torch.Tensor
            Object centerness predictions, with batch as first dimension.
        """
        (p3, p4, p5, p6, p7) = self.fpn(self.resnet(batch_imgs))

        (box3, cls3, ctr3) = self._layer(p3)
        (box4, cls4, ctr4) = self._layer(p4)
        (box5, cls5, ctr5) = self._layer(p5)
        (box6, cls6, ctr6) = self._layer(p6)
        (box7, cls7, ctr7) = self._layer(p7)

        reg_bboxes = (box3, box4, box5, box6, box7)
        cls_logits = (cls3, cls4, cls5, cls6, cls7)
        centerness = (ctr3, ctr4, ctr5, ctr6, ctr7)

        return (reg_bboxes, cls_logits, centerness)

    def compute_loss(self,
                     reg_bboxes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                     cls_logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                     centerness: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                     ground_truth_bboxes: Sequence[torch.Tensor],
                     ground_truth_labels: Sequence[torch.Tensor]) \
                     -> Dict[str, torch.Tensor]:
        """Computes the inference loss related to the ground truth.

        Parameters
        ----------
        reg_bboxes : tuple of torch.Tensor
            Regressor bounding boxes.
        cls_logits : tuple of torch.Tensor
            Classifier score logits.
        centerness : tuple of torch.Tensor
            Distance to the center of the object.
        ground_truth_bboxes : list of torch.Tensor
            Ground truth of the object's bounding boxes.
        ground_truth_labels : list of torch.Tensor
            Ground truth of the object's labels.

        Returns
        -------
        losses : Dict[str, torch.Tensor]
            Dictionary with all loss components. Classifier loss, bounding box regressor loss
             and bounding box centerness loss.
        """
        batchsize, num_classes = cls_logits[0].shape[:2]
        locations = (self.locations3, self.locations4, self.locations5, self.locations6, self.locations7)
        # flatten H, W dimensions, per pyramid level
        lvls_logits = [t.permute(0, 2, 3, 1).reshape(batchsize, -1, num_classes) for t in cls_logits]
        lvls_bboxes = [t.permute(0, 2, 3, 1).reshape(batchsize, -1, 4) for t in reg_bboxes]
        lvls_ctrnss = [t.permute(0, 2, 3, 1).reshape(batchsize, -1) for t in centerness]
        # concatenate pyramid levels
        cat_logits = torch.cat(lvls_logits, dim=1)
        cat_ctrnss = torch.cat(lvls_ctrnss, dim=1)
        cat_locs = torch.cat(locations, dim=0)
        # calculate (square) box size, per location
        lvls_boxsizes = []
        for lvl in range(len(locations)):
            lvls_boxsizes.append(torch.full((locations[lvl].size(0), 1),
                                            self.fpn.strides[lvl],
                                            dtype=torch.float32,
                                            device=locations[lvl].device))
        # concatenate box sizes pyramid levels
        cat_boxsizes = torch.cat(lvls_boxsizes)

        # compute indexes of locations matching gt boxes
        matched_idxs = []
        for gt_bboxes in ground_truth_bboxes:
            gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2
            # center sampling: anchor point must be close enough to gt center
            pairwise_match = (cat_locs.unsqueeze(1) - gt_centers.unsqueeze(0)).abs_().max(dim=2).values \
                                < self.center_sampling_radius * cat_boxsizes
            # compute pairwise distance between N points and M boxes
            x, y = cat_locs.unsqueeze(2).unbind(dim=1)  # (N, 1)
            x0, y0, x1, y1 = gt_bboxes.unsqueeze(0).unbind(dim=2)  # (1, M)
            pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # (N, M)

            # anchor point must be inside gt
            pairwise_match &= pairwise_dist.min(dim=2).values > 0

            # each anchor is only responsible for certain scale range
            lower_bound = cat_boxsizes * 4
            lower_bound[: locations[0].size(0)] = 0
            upper_bound = cat_boxsizes * 8
            upper_bound[-locations[-1].size(0) :] = float("inf")
            pairwise_dist = pairwise_dist.max(dim=2).values
            pairwise_match &= (pairwise_dist > lower_bound) & (pairwise_dist < upper_bound)

            # match the gt box with minimum area, if there are multiple gt matches
            gt_areas = box.area(gt_bboxes)
            pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas.unsqueeze(0))
            min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match
            matched_idx[min_values < 1e-5] = -1  # unmatched anchors are assigned -1

            matched_idxs.append(matched_idx)

        # compute target masks
        imgs_classes_targets = []
        imgs_boxes_targets = []
        for img_bboxes, img_labels, img_matched_idxs in \
                zip(ground_truth_bboxes, ground_truth_labels, matched_idxs):
            if len(img_labels) == 0:
                img_classes_targets = img_labels.new_zeros((len(img_matched_idxs),))
                img_boxes_targets = img_bboxes.new_zeros((len(img_matched_idxs), 4))
            else:
                img_classes_targets = img_labels[img_matched_idxs.clip(min=0)]
                img_boxes_targets = img_bboxes[img_matched_idxs.clip(min=0)]
            img_classes_targets[img_matched_idxs < 0] = -1  # background
            imgs_classes_targets.append(img_classes_targets)
            imgs_boxes_targets.append(img_boxes_targets)

        classes_targets = torch.stack(imgs_classes_targets)
        boxes_targets = torch.stack(imgs_boxes_targets)

        # compute foreground
        foreground_mask = classes_targets >= 0
        num_foreground = foreground_mask.sum().item()

        # classification loss
        scores_targets = torch.zeros_like(cat_logits)
        scores_targets[foreground_mask, classes_targets[foreground_mask]] = 1.0
        loss_cls = torchvision.ops.sigmoid_focal_loss(cat_logits, scores_targets, reduction="sum")

        pred_boxes = torch.cat([box.centerdistance_to_box(lc, bx, sc) for lc, bx, sc in
                                zip(locations, lvls_bboxes, self.fpn.strides)], dim=1)

        # regression loss: GIoU loss
        loss_bbox_reg = torchvision.ops.generalized_box_iou_loss(
                            pred_boxes[foreground_mask],
                            boxes_targets[foreground_mask],
                            reduction="sum")

        lvls_boxes_targets = boxes_targets.split([lc.size(0) for lc in locations], dim=1)
        bbox_reg_targets = torch.cat([box.box_to_centerdistance(lc, bx, sc) for lc, bx, sc in
                                      zip(locations, lvls_boxes_targets, self.fpn.strides)], dim=1)
        if len(bbox_reg_targets) == 0:
            ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.shape[:-1])
        else:
            left_right = bbox_reg_targets[..., [0, 2]]
            top_bottom = bbox_reg_targets[..., [1, 3]]
            ctrness_targets = torch.sqrt(   
                                    (left_right.min(dim=-1).values / left_right.max(dim=-1).values)
                                    * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values))

        # ctrness loss
        loss_bbox_ctrness = torch.nn.functional.binary_cross_entropy_with_logits(
                                cat_ctrnss[foreground_mask],
                                ctrness_targets[foreground_mask],
                                reduction="sum")

        return {
            "classification": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
        }
    
    def load_ots_model(self, model_file: str, load_final_conv_layer: bool = True):
        """Loads parameters from an off-the-shelf model.

        Parameters
        ----------
        model_file : str
            Off-the-shelf model file path.
        load_final_conv_layer : bool = True
            Whether to load final conv layer or not. This layer defines the number of keypoints.
 
        """
        load_torchvision_fcos(self, model_file, load_final_conv_layer=load_final_conv_layer)

    def _layer(self, feature_map: torch.Tensor) \
                -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared layer to be applied to all the pyramid feature maps."""
        reg_feat = self.reg_subnet(feature_map)
        reg_bbox = self.reg_bbox(reg_feat)

        cls_feat = self.cls_subnet(feature_map)
        cls_scor = self.cls_logits(cls_feat)

        cntrness = self.centerness(reg_feat)

        return (reg_bbox, cls_scor, cntrness)


class FCOS_Full(FCOS):
    """FCOS complete model class."""

    def forward(self, batch_imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defines the computation performed by the forward pass.

        Runs the full model, both quantizable and non-quantizable operations.

        Parameters
        ----------
        batch_imgs : torch.Tensor
            Batch of images (NCHW).

        Returns
        -------
        final_bboxes : torch.Tensor
            Bounding boxes, with batch as first dimension.
        final_scores : torch.Tensor
            Scores, with batch as first dimension.
        final_labels : torch.Tensor
            Labels, with batch as first dimension.
        """
        reg_bboxes, cls_logits, centerness = super().forward(batch_imgs)
        return self.post(reg_bboxes, cls_logits, centerness)

