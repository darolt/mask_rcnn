import logging
import numpy as np

import torch.nn.functional as F
import torch

from mrcnn.config import ExecutionConfig as ExeCfg
from mrcnn.utils import unmold_boxes_x, unmold_detections_x, register_hook


class Losses():
    def __init__(self, rpn_class=0.0, rpn_bbox=0.0, mrcnn_class=0.0,
                 mrcnn_bbox=0.0, mrcnn_mask=0.0):
        self.rpn_class = rpn_class
        self.rpn_bbox = rpn_bbox
        self.mrcnn_class = mrcnn_class
        self.mrcnn_bbox = mrcnn_bbox
        self.mrcnn_mask = mrcnn_mask
        self.update_total_loss()

    def item(self):
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
    indices = torch.nonzero(rpn_match != 0).detach()

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices[:, 0],
                                        indices[:, 1], :]
    anchor_class = anchor_class[indices[:, 0], indices[:, 1]]

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
    indices = torch.nonzero(rpn_match == 1).detach()

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices[:, 0], indices[:, 1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    ranges_per_img = torch.empty((indices.shape[0]), dtype=torch.long)
    count = 0
    for img_idx in range(target_bbox.shape[0]):
        nb_elem = torch.nonzero(indices[:, 0] == img_idx).shape[0]
        ranges_per_img[count:count+nb_elem] = torch.arange(nb_elem)
        count += nb_elem
    target_bbox = target_bbox[indices[:, 0], ranges_per_img]

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
        loss = torch.tensor([0], dtype=torch.float32, device=ExeCfg.DEVICE)

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
        loss = torch.tensor([0], dtype=torch.float32, device=ExeCfg.DEVICE)

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
        loss = torch.tensor([0], dtype=torch.float32, device=ExeCfg.DEVICE)

    return loss


def compute_rpn_losses(rpn_target, rpn_out):

    rpn_class_loss = compute_rpn_class_loss(rpn_target.match,
                                            rpn_out.class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_target.deltas,
                                          rpn_target.match, rpn_out.deltas)
    zero = torch.tensor([0.0], dtype=torch.float32, device=ExeCfg.DEVICE)

    return Losses(rpn_class_loss, rpn_bbox_loss, zero.clone(),
                  zero.clone(), zero.clone())


def compute_mrcnn_losses(mrcnn_targets, mrcnn_outs):
    zero = torch.tensor([0.0], dtype=torch.float32, device=ExeCfg.DEVICE)
    mrcnn_class_loss = zero.clone()
    mrcnn_bbox_loss = zero.clone()
    mrcnn_mask_loss = zero.clone()

    for mrcnn_target, mrcnn_out in zip(mrcnn_targets, mrcnn_outs):
        mrcnn_class_loss += compute_mrcnn_class_loss(
            mrcnn_target.class_ids, mrcnn_out.class_logits)
        mrcnn_bbox_loss += compute_mrcnn_bbox_loss(
            mrcnn_target.deltas, mrcnn_target.class_ids, mrcnn_out.deltas)
        mrcnn_mask_loss += compute_mrcnn_mask_loss(
            mrcnn_target.masks, mrcnn_target.class_ids, mrcnn_out.masks)

    if len(mrcnn_outs) != 0:
        mrcnn_class_loss /= len(mrcnn_outs)
        mrcnn_bbox_loss /= len(mrcnn_outs)
        mrcnn_mask_loss /= len(mrcnn_outs)

    return Losses(zero.clone(), zero.clone(), mrcnn_class_loss,
                  mrcnn_bbox_loss, mrcnn_mask_loss)


def compute_losses(rpn_target, rpn_out, mrcnn_targets, mrcnn_outs):
    rpn_loss = compute_rpn_losses(rpn_target, rpn_out)
    mrcnn_loss = compute_mrcnn_losses(mrcnn_targets, mrcnn_outs)

    return rpn_loss + mrcnn_loss


def compute_map_loss(gt_masks, gt_boxes, gt_class_ids, image_metas,
                     detections, mrcnn_masks):
    """Compute loss for single image according to:
       https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    """
    image_shape = np.array(image_metas[1:4])
    window = np.array(image_metas[4:8])

    # unmold mrcnn masks
    pred_boxes, pred_masks = unmold_detections_x(detections, mrcnn_masks,
                                                 image_shape[:2], window)

    # ground truth does not require grad computation
    with torch.no_grad():
        # remove zeros (padding)
        zero_ix = np.where(gt_class_ids == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else gt_class_ids.shape[0]
        gt_boxes, gt_masks = unmold_boxes_x(
            gt_boxes[:N], gt_class_ids[:N], gt_masks[:N], image_shape[:2],
            window)

    # register_hook(pred_boxes, 'pred_boxes:')
    # register_hook(pred_masks[0], 'pred_masks:')
    ious = _compute_ious(gt_boxes, gt_masks, pred_boxes, pred_masks)
    precision = _compute_map(ious)
    # assert 0 <= precision <= 1, \
    #     f"Precision ({precision}) must be in range [0, 1]."

    return precision


UNEXPECTED_CONDITION_STR = 'Boxes overlaping reached unexpected condition.'
NEGATIVE_VALUE_ON_INTERSECTION = 'Negative value on intersection.'
POSITIVE_VALUE_ON_INTERSECTION = 'Positive value on intersection.'


def _overlap_idx(box1_x1, box1_x2, box2_x1, box2_x2):
    if box1_x1 - box2_x2 >= -1:  # box2 then box1
        inter1_x1 = (box2_x1 - box1_x1)
        inter1_x2 = (box2_x2 - box1_x2)
        inter2_x1 = (box2_x1 - box1_x1)
        inter2_x2 = (box2_x2 - box1_x2)
        assert (inter1_x1 < 0 and inter1_x2 < 0 and inter2_x1 < 0 and
                inter2_x2 < 0), POSITIVE_VALUE_ON_INTERSECTION
    elif box2_x1 - box1_x2 >= -1:  # box1 then box2
        inter1_x1 = (box1_x1 - box2_x1)
        inter1_x2 = (box1_x2 - box2_x2)
        inter2_x1 = (box1_x1 - box2_x1)
        inter2_x2 = (box1_x2 - box2_x2)
        assert (inter1_x1 < 0 and inter1_x2 < 0 and inter2_x1 < 0 and
                inter2_x2 < 0), POSITIVE_VALUE_ON_INTERSECTION
    else:
        if box2_x1 <= box1_x1 <= box2_x2:
            inter1_x1 = (box2_x2 - box1_x1)/1000.0
            inter2_x1 = box1_x1 - box2_x1
        elif box1_x1 <= box2_x1 <= box1_x2:
            inter1_x1 = box2_x1 - box1_x1
            inter2_x1 = (box1_x2 - box2_x1)/1000.0
        else:
            raise Exception(f"{UNEXPECTED_CONDITION_STR}"
                            f" ({box1_x1}, {box1_x2} and"
                            f"{box2_x1}, {box2_x2}).")

        if box2_x1 <= box1_x2 <= box2_x2:
            inter1_x2 = box1_x2 - box1_x1
            inter2_x2 = box1_x2 - box2_x1
        elif box1_x1 <= box2_x2 <= box1_x2:
            inter1_x2 = box2_x2 - box1_x1
            inter2_x2 = box2_x2 - box2_x1
        else:
            raise Exception(f"{UNEXPECTED_CONDITION_STR}"
                            f" ({box1_x1}, {box1_x2} and"
                            f"{box2_x1}, {box2_x2}).")

        assert (inter1_x1 >= 0 and inter1_x2 >= 0 and inter2_x1 >= 0 and
                inter2_x2 >= 0), NEGATIVE_VALUE_ON_INTERSECTION

    inter1_x = (inter1_x1, inter1_x2)
    inter2_x = (inter2_x1, inter2_x2)
    return (inter1_x, inter2_x)


def _get_intersection_idx(box1, box2):
    """Compute the intersections between box1 and box2 and return the indexes
    of the intersection box relative to both boxes.

    Note:
        Boxes coordinates are in the following order: (y1, x1, y2, x2).

    Args:
        box1 (torch.FloatTensor((4))):
        Absolute coordinates of box1 (predicted box).
        box2 (torch.FloatTensor((4))):
        Absolute coordinates of box2 (ground truth box).

    Returns:
        intersect1_idx (torch.FloatTensor((4))):
            Relative coordinates of intersection inside box1.
        intersect2_idx (torch.FloatTensor((4))):
            Relative coordinates of intersection inside box2.
        has_intersect (bool): If box1 and box2 intersect.
    """
    box1_y1, box1_x1, box1_y2, box1_x2 = box1
    box2_y1, box2_x1, box2_y2, box2_x2 = box2
    intersect_idx_x = _overlap_idx(box1_x1, box1_x2, box2_x1, box2_x2)
    intersect_idx_y = _overlap_idx(box1_y1, box1_y2, box2_y1, box2_y2)

    # concat
    intersect1_idx = torch.stack(
        (intersect_idx_y[0][0], intersect_idx_x[0][0],
         intersect_idx_y[0][1], intersect_idx_x[0][1]),
        dim=0)
    intersect2_idx = torch.stack(
        (intersect_idx_y[1][0], intersect_idx_x[1][0],
         intersect_idx_y[1][1], intersect_idx_x[1][1]),
        dim=0)
    return (intersect1_idx, intersect2_idx)


def _gen_grid(mask, y1, x1, y2, x2):
    dy, dx = (y2 - y1).int(), (x2 - x1).int()
    indices_y = y2.repeat((dy, dx))
    indices_x = x2.repeat((dy, dx))
    x_range = torch.arange(0.0, dx).repeat((dy,)).reshape((dy, dx))
    y_range = torch.arange(0.0, dy).repeat((dx,)).reshape((dx, dy)).t()
    indices_y = indices_y + y_range.to(ExeCfg.DEVICE)
    indices_x = indices_x + x_range.to(ExeCfg.DEVICE)
    indices = torch.stack((indices_y, indices_x), dim=2)
    shape = torch.FloatTensor((mask.shape[0], mask.shape[1])).to(ExeCfg.DEVICE)
    indices = (indices - shape)/shape
    return indices


def _extract(mask, y1, x1, y2, x2):
    grid = _gen_grid(mask, y1, x1, y2, x2)
    return F.grid_sample(mask, grid)


def _compute_intersection(pred_mask, gt_mask, pred_inter_idx, gt_inter_idx):
    pred_y1, pred_x1, pred_y2, pred_x2 = pred_inter_idx.int()
    gt_y1, gt_x1, _, _ = gt_inter_idx.int()
    proj_pred_gt = torch.zeros_like(pred_mask)

    min_range_x = torch.min(proj_pred_gt.shape[1] - pred_x1 - 1,
                            gt_mask.shape[1] - gt_x1 - 1)
    dx = torch.clamp(pred_x2 - pred_x1, 1.0, min_range_x)

    min_range_y = torch.min(proj_pred_gt.shape[0] - pred_y1 - 1,
                            gt_mask.shape[0] - gt_y1 - 1)
    dy = torch.clamp(pred_y2 - pred_y1, 1.0, min_range_y)

    # out = _extract(gt_mask, gt_y1, gt_x1, gt_y1+dy, gt_x1+dx)
    proj_pred_gt[pred_y1:pred_y1+dy, pred_x1:pred_x1+dx] = \
        gt_mask[gt_y1:gt_y1+dy, gt_x1:gt_x1+dx]
    return proj_pred_gt*pred_mask


def _compute_factor(pred_inter_idx):
    """Returns a value inside the range [-1, 0]. The higher the value, closer
    are the boxes."""
    register_hook(pred_inter_idx, 'pred_inter_idx:')
    factor = pred_inter_idx.sum().to(ExeCfg.DEVICE)
    factor = (factor/100).sigmoid() - 1.0
    register_hook(factor, 'factor:')

    assert (factor <= 0).byte().all(), 'Factor cannot be greater than 0.'
    assert (factor >= -1).byte().all(), 'Factor cannot be less than -1.'
    return factor


def _compute_factor2(pred_box, gt_box):
    center_pred = ((pred_box[2] - pred_box[0])/2,
                   (pred_box[3] - pred_box[1])/2)
    center_gt = ((gt_box[2] - gt_box[0])/2,
                 (gt_box[3] - gt_box[1])/2)
    distance = ((center_pred[0] - center_gt[0])**2 +
                (center_gt[1] - center_gt[1])).sqrt()
    return -distance


def _compute_iou(gt_box, gt_mask, pred_box, pred_mask):
    pred_inter_idx, gt_inter_idx = _get_intersection_idx(pred_box, gt_box)

    if (pred_inter_idx < 0).any() or (gt_inter_idx < 0).any():
        # return _compute_factor(pred_inter_idx)
        return _compute_factor2(pred_box, gt_box)

    intersection = _compute_intersection(
        pred_mask, gt_mask, pred_inter_idx, gt_inter_idx)

    pred_y1, pred_x1, pred_y2, pred_x2 = pred_inter_idx

    # computing areas for mask
    intersection_area = intersection.sum()
    pred_area = pred_mask.sum()
    gt_area = gt_mask.sum()
    union_area = pred_area + gt_area - intersection_area
    # computing areas for boxes
    intersection_area_box = (pred_y2 - pred_y1)*(pred_x2 - pred_x1)
    gt_area_box = gt_mask.shape[0]*gt_mask.shape[1]
    pred_area_box = (pred_box[2] - pred_box[0])*(pred_box[3] - pred_box[1])
    union_area_box = pred_area_box + gt_area_box - intersection_area_box

    # average two IOUs
    iou = ((intersection_area/union_area) +
           (intersection_area_box/union_area_box))/2.0
    # iou = intersection_area/union_area
    assert union_area >= 1.0, f"Union area is less than 1.0 ({union_area})"
    assert intersection_area >= 0.0, \
        f"Intersection area is negative ({intersection_area})"
    return iou


def _compute_ious(gt_boxes, gt_masks, pred_boxes, pred_masks):
    """Compute Intersection over Union of ground truth and predicted masks.
    gt_masks and pred_masks must be in mini-mask format. Boxes indicate mask
    position inside original image.

    Args:
        gt_boxes (torch.FloatTensor((nb_gt_masks, 4))):
            Ground truth bounding boxes.
        gt_masks (list of torch.IntTensor with length nb_gt_masks):
            Ground truth masks with shape (mask_height, mask_width).
            mask_height and mask_width are variable.
        pred_boxes (torch.FloatTensor((nb_pred_masks, 4))):
            Predicted boxes.
        pred_masks (torch.FloatTensor((nb_pred_masks, img_height, img_width))):
            Predicted masks.
        pred_masks (list of torch.IntTensor with length nb_pred_masks):
            Predicted masks with shape (mask_height, mask_width). mask_height
            and mask_width are variable.

    Returns:
        ious (torch.FloatTensor((nb_gt_masks, nb_pred_masks))):
            Intersection over Union for every combination of ground truth and
            and predicted masks.
    """
    # compute IOUs
    register_hook(pred_boxes, 'pred_boxes: ')
    ious = torch.zeros((len(gt_masks), len(pred_masks)), dtype=torch.float)
    logging.info(f"{len(gt_masks)} x {len(pred_masks)}")
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            pred_mask = pred_masks[pred_idx]
            gt_mask = gt_masks[gt_idx]
            ious[gt_idx, pred_idx] = _compute_iou(
                gt_box, gt_mask, pred_box, pred_mask)

    # assert (ious < 0).sum() == 0, 'IoU has negative values.'
    assert (ious <= 1).byte().all(), 'IoU cannot have values larger than 1.'
    return ious


MAGNIFIER = 100


def _compute_map(ious):
    """Compute mean average precision."""
    thresholds = torch.arange(0.5, 1.0, 0.05)
    precisions = torch.empty_like(thresholds, device=ExeCfg.DEVICE)
    register_hook(ious, 'ious:')
    for thresh_idx, threshold in enumerate(thresholds):
        hits = ((ious - threshold)*MAGNIFIER).sigmoid()
        register_hook(hits, 'hits_thresh:')
        gt_sum = ((hits.sum(dim=0) - 0.5)*MAGNIFIER).sigmoid()
        pred_sum = ((hits.sum(dim=1) - 0.5)*MAGNIFIER).sigmoid()

        tp = pred_sum.sum()
        overpred = gt_sum.sum() - tp
        if overpred > 0.5:
            print(overpred)
        # register_hook(tp, 'tp_thresh:')
        fp = (1 - gt_sum).sum()
        # register_hook(fp, 'fp_thresh:')
        fn = (1 - pred_sum).sum()
        # register_hook(fn, 'fn_thresh:')
        precisions[thresh_idx] = (tp/(tp + fp + fn)) - overpred
        # register_hook(precisions[thresh_idx], 'precision_thresh:')

    # average precisions
    return precisions.mean()
