import torch
import numpy as np
import torch.nn.functional as F

import mrcnn.config

from mrcnn import utils
from mrcnn.detection import detection_layer2


class Losses():
    def __init__(self, rpn_class=0.0, rpn_bbox=0.0, mrcnn_class=0.0,
                 mrcnn_bbox=0.0, mrcnn_mask=0.0):
        self.rpn_class = rpn_class
        self.rpn_bbox = rpn_bbox
        self.mrcnn_class = mrcnn_class
        self.mrcnn_bbox = mrcnn_bbox
        self.mrcnn_mask = mrcnn_mask
        self.update_total_loss()

    def to_item(self):
        return Losses(self.rpn_class.item(), self.rpn_bbox.item(),
                      self.mrcnn_class.item(), self.mrcnn_bbox.item(),
                      self.mrcnn_mask.item())

    def to_list(self):
        return [self.total, self.rpn_class, self.rpn_bbox,
                self.mrcnn_class, self.mrcnn_bbox, self.mrcnn_mask]

    def update_total_loss(self):
        self.total = self.rpn_class + self.rpn_bbox + self.mrcnn_class + \
                     self.mrcnn_bbox + self.mrcnn_mask

    def __truediv__(self, b):
        new_rpn_class = self.rpn_class/b
        new_rpn_bbox = self.rpn_bbox/b
        new_mrcnn_class = self.mrcnn_class/b
        new_mrcnn_bbox = self.mrcnn_bbox/b
        new_mrcnn_mask = self.mrcnn_mask/b
        return Losses(new_rpn_class, new_rpn_bbox, new_mrcnn_class,
                      new_mrcnn_bbox, new_mrcnn_mask)

    def __add__(self, other):
        new_rpn_class = self.rpn_class + other.rpn_class
        new_rpn_bbox = self.rpn_bbox + other.rpn_bbox
        new_mrcnn_class = self.mrcnn_class + other.mrcnn_class
        new_mrcnn_bbox = self.mrcnn_bbox + other.mrcnn_bbox
        new_mrcnn_mask = self.mrcnn_mask + other.mrcnn_mask
        return Losses(new_rpn_class, new_rpn_bbox, new_mrcnn_class,
                      new_mrcnn_bbox, new_mrcnn_mask)


############################################################
#  Loss Functions
############################################################


def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.detach()[:, 0],
                                        indices.detach()[:, 1], :]
    anchor_class = anchor_class[indices.detach()[:, 0], indices.detach()[:, 1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unused bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match == 1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.detach()[:, 0], indices.detach()[:, 1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0, :rpn_bbox.size()[0], :]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """
    # Loss
    if target_class_ids.nelement() != 0 and pred_class_logits.nelement() != 0:
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = torch.tensor([0], dtype=torch.float32,
                            device=mrcnn.config.DEVICE)

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if target_class_ids.nelement() != 0:
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.detach()].long()

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[positive_roi_ix[:].detach(), :]
        pred_bbox = pred_bbox[positive_roi_ix[:].detach(),
                              positive_roi_class_ids[:].detach(), :]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = torch.tensor([0], dtype=torch.float32,
                            device=mrcnn.config.DEVICE)

    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, num_classes, height, width] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.nelement() != 0:
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.detach()].long()

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[positive_ix[:].detach(), :, :]
        y_pred = pred_masks[positive_ix[:].detach(),
                            positive_class_ids[:].detach(),
                            :, :]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = torch.tensor([0], dtype=torch.float32,
                            device=mrcnn.config.DEVICE)

    return loss


def compute_iou_loss2(gt_masks, pred_masks):
    ious = compute_ious(gt_masks, pred_masks)

    return compute_mAP(ious)


def compute_iou_loss(gt_masks, gt_boxes, gt_class_ids, image_metas,
                     predictions, config, detections=None, mrcnn_masks=None):
    """Compute loss for single image according to:
       https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    """
    if not detections:
        mrcnn_masks = predictions[7][0]
        mrcnn_boxes = predictions[5][0]
        mrcnn_class_logits = predictions[3][0]
        mrcnn_probs = predictions[9][0]
        rois = predictions[8][0]
        print(f"mrcnn_masks {mrcnn_masks.shape}")  # X, 28, 28

        if mrcnn_class_logits.nelement() == 0:
            print("no predictions")
            return

        detections = detection_layer2(config, rois, mrcnn_probs, mrcnn_boxes,
                                      image_metas.unsqueeze(0))

    # remove zeros (padding)
    image_shape = np.array(image_metas[1:4])
    window = np.array(image_metas[4:8])
    zero_ix = np.where(gt_class_ids == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else gt_class_ids.shape[0]

    # 1 - unmold gt and mrcnn masks
    mrcnn_masks = mrcnn_masks.permute(0, 2, 3, 1)
    final_masks = utils.unmold_detections(detections, mrcnn_masks,
                                          image_shape[:2], window)[3]

    gt_masks = gt_masks[:N].detach()
    gt_boxes = gt_boxes[:N].detach()
    print(f"gt_masks {gt_masks.shape}")
    print(f"gt_boxes {gt_boxes.shape}")

    full_gt_masks = utils.unmold_boxes(gt_boxes, gt_class_ids, gt_masks,
                                       image_shape[:2], window)[3]

    ious = compute_ious(full_gt_masks, final_masks)

    precision = compute_mAP(ious)
    print(f"precision: {precision}")

    return precision


def compute_ious(gt_masks, pred_masks):
    # compute IOUs
    gt_masks = gt_masks.to(torch.uint8)
    pred_masks = pred_masks.to(torch.uint8)
    print(f"gt_masks  {gt_masks.shape}")
    print(f"pred_masks {pred_masks.shape}")
    ious = torch.zeros((gt_masks.shape[2], pred_masks.shape[2]),
                       dtype=torch.float)
    print(f"{gt_masks.shape[2]} x {pred_masks.shape[2]}")
    for gt_idx in range(0, gt_masks.shape[2]):
        for pred_idx in range(0, pred_masks.shape[2]):
            intersection = pred_masks[:, :, pred_idx] & gt_masks[:, :, gt_idx]
            intersection = torch.nonzero(intersection).shape[0]
            union = pred_masks[:, :, pred_idx] | gt_masks[:, :, gt_idx]
            union = torch.nonzero(union).shape[0]
            iou = intersection/union if union != 0.0 else 0.0
            ious[gt_idx, pred_idx] = iou
    return ious


def compute_mAP(ious):
    """Compute mean average precision."""
    # compute hits
    thresholds = torch.arange(0.5, 1.0, 0.05)
    precisions = torch.empty_like(thresholds, device=mrcnn.config.DEVICE)
    for thresh_idx, threshold in enumerate(thresholds):
        hits = ious > threshold
        tp = torch.nonzero(hits.sum(dim=0)).shape[0]
        fp = torch.nonzero(hits.sum(dim=1) == 0).shape[0]
        fn = torch.nonzero(hits.sum(dim=0) == 0).shape[0]
        precisions[thresh_idx] = tp/(tp + fp + fn)

    # average precisions
    return precisions.mean()


def compute_losses(rpn_target, rpn_out, mrcnn_targets, mrcnn_outs):

    rpn_class_loss = compute_rpn_class_loss(rpn_target.match,
                                            rpn_out.class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_target.deltas,
                                          rpn_target.match, rpn_out.deltas)
    mrcnn_class_loss = torch.tensor([0.0], dtype=torch.float32,
                                    device=mrcnn.config.DEVICE)
    mrcnn_bbox_loss = torch.tensor([0.0], dtype=torch.float32,
                                   device=mrcnn.config.DEVICE)
    mrcnn_mask_loss = torch.tensor([0.0], dtype=torch.float32,
                                   device=mrcnn.config.DEVICE)
    for batch in range(0, len(mrcnn_targets)):
        mrcnn_class_loss += compute_mrcnn_class_loss(
            mrcnn_targets[batch].class_ids, mrcnn_outs[batch].class_logits)
        mrcnn_bbox_loss += compute_mrcnn_bbox_loss(
            mrcnn_targets[batch].deltas, mrcnn_targets[batch].class_ids,
            mrcnn_outs[batch].deltas)
        mrcnn_mask_loss += compute_mrcnn_mask_loss(
            mrcnn_targets[batch].masks, mrcnn_targets[batch].class_ids,
            mrcnn_outs[batch].masks)

    if len(mrcnn_outs) != 0:
        mrcnn_class_loss /= len(mrcnn_outs)
        mrcnn_bbox_loss /= len(mrcnn_outs)
        mrcnn_mask_loss /= len(mrcnn_outs)

    return Losses(rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss,
                  mrcnn_bbox_loss, mrcnn_mask_loss)
