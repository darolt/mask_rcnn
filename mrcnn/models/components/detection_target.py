import logging

import torch

from roialign.roi_align.crop_and_resize import CropAndResizeFunction
from mrcnn.config import ExecutionConfig as ExeCfg
from mrcnn.utils import utils
from mrcnn.utils.utils import MRCNNTarget


def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks,
                           config):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    no_crowd_bool = _handle_crowds(proposals, gt_class_ids,
                                   gt_boxes, gt_masks)

    # Compute overlaps matrix [nb_batches, proposals, gt_boxes]
    overlaps = _bbox_overlaps(proposals, gt_boxes)

    # Determine positive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5

    # Subsample ROIs. Aim for ROI_POSITIVE_RATIO positive
    # Positive ROIs
    if torch.nonzero(positive_roi_bool).nelement() != 0:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count].to(ExeCfg.DEVICE)
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices, :]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices, :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment, :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment]

        # Compute bbox refinement for positive ROIs
        deltas = utils.box_refinement(positive_rois,
                                      roi_gt_boxes)
        std_dev = torch.from_numpy(config.BBOX_STD_DEV).float()
        std_dev = std_dev.to(ExeCfg.DEVICE)
        deltas /= std_dev

        # Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment, :, :]

        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = (torch.arange(roi_masks.size()[0]).int()
                   .to(ExeCfg.DEVICE))
        masks = CropAndResizeFunction(
            config.MASK_SHAPE[0],
            config.MASK_SHAPE[1],
            0)(roi_masks.unsqueeze(1), boxes, box_ids)
        masks = masks.squeeze(1)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = torch.round(masks)
    else:
        positive_count = 0

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    logging.debug(f"pos: {positive_roi_bool.sum()}, "
                  f"neg: {negative_roi_bool.sum()}")
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    if torch.nonzero(negative_roi_bool).nelement() != 0 and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count].to(ExeCfg.DEVICE)
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices, :]
    else:
        negative_count = 0

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = torch.zeros(negative_count, dtype=torch.int,
                            device=ExeCfg.DEVICE)
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = torch.zeros(negative_count, 4, dtype=torch.float32,
                            device=ExeCfg.DEVICE)
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = torch.zeros(negative_count, config.MASK_SHAPE[0],
                            config.MASK_SHAPE[1], dtype=torch.float32,
                            device=ExeCfg.DEVICE)
        masks = torch.cat([masks, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        roi_gt_class_ids = torch.zeros(negative_count,
                                       device=ExeCfg.DEVICE)
        deltas = torch.zeros(negative_count, 4, dtype=torch.int,
                             device=ExeCfg.DEVICE)
        masks = torch.zeros(negative_count, config.MASK_SHAPE[0],
                            config.MASK_SHAPE[1], device=ExeCfg.DEVICE)
    else:
        rois = torch.tensor([], dtype=torch.float32,
                            device=ExeCfg.DEVICE)
        roi_gt_class_ids = torch.tensor([], dtype=torch.int,
                                        device=ExeCfg.DEVICE)
        deltas = torch.tensor([], dtype=torch.float32,
                              device=ExeCfg.DEVICE)
        masks = torch.tensor([], dtype=torch.float32,
                             device=ExeCfg.DEVICE)

    mrcnn_target = MRCNNTarget(roi_gt_class_ids, deltas, masks)
    return rois, mrcnn_target


def _bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every box1 against every box2 without loops.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = torch.zeros(y1.size()[0], requires_grad=False, dtype=torch.float32,
                        device=ExeCfg.DEVICE)
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, 0] + b2_area[:, 0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    nans = (iou != iou)
    iou[nans] = -1
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps


def _handle_crowds(proposals, gt_class_ids, gt_boxes, gt_masks):
    # Handle crowds
    # A crowd box is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = torch.nonzero(gt_class_ids < 0)  # [:, 0]
    if crowd_ix.nelement() != 0:
        crowd_ix = crowd_ix[:, 0]
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]
        crowd_boxes = gt_boxes[crowd_ix, :]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix, :]
        gt_masks = gt_masks[non_crowd_ix, :]

        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = _bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = torch.tensor(proposals.size()[0]*[True],
                                     dtype=torch.uint8,
                                     device=ExeCfg.DEVICE,
                                     requires_grad=False)
    return no_crowd_bool
