"""Non-standard metrics."""
import logging

import torch

from tools.time_profiling import profilable
from tools.config import Config


def compute_map_metric(gt_masks, pred_masks, gt_boxes, pred_boxes):
    with torch.no_grad():
        ious = compute_ious(gt_masks, pred_masks, gt_boxes, pred_boxes)
        precision = compute_map(ious)
    return precision


@profilable
def alt_compute_ious(gt_masks, pred_masks):
    """Compute Intersection over Union of ground truth and predicted masks.

    Args:
        gt_masks (torch.IntTensor((img_height, img_width, nb_gt_masks))):
            Ground truth masks.
        pred_masks (torch.FloatTensor((img_height, img_width, nb_pred_masks))):
            Predicted masks.

    Returns:
        ious (torch.FloatTensor((nb_gt_masks, nb_pred_masks))):
            Intersection over Union.
    Note: This version vectorize compute_ious, but
    uses a lot of memory and does not show improvements in timing
    """
    # compute IOUs
    gt_masks = gt_masks.to(torch.uint8)
    pred_masks = pred_masks.to(torch.uint8)
    height, width = gt_masks.shape[0:2]
    nb_gts, nb_preds = gt_masks.shape[2], pred_masks.shape[2]
    logging.info(f"{nb_gts} x {nb_preds} (GT x predictions)")

    gt_masks = gt_masks.unsqueeze(3).repeat(1, 1, 1, nb_preds).view(
        height, width, -1)
    pred_masks = pred_masks.repeat(1, 1, nb_gts)
    inter = (gt_masks & pred_masks).sum((0, 1))
    union = (gt_masks | pred_masks).sum((0, 1))

    ious = torch.div(inter.float(), union.float())
    ious[union == 0] = 0
    return ious.view(nb_gts, nb_preds)


@profilable
def compute_ious(gt_masks, pred_masks, gt_boxes, pred_boxes):
    """Compute Intersection over Union of ground truth and predicted masks.

    Args:
        gt_masks (torch.IntTensor((img_height, img_width, nb_gt_masks))):
            Ground truth masks.
        pred_masks (torch.FloatTensor((img_height, img_width, nb_pred_masks))):
            Predicted masks.

    Returns:
        ious (torch.FloatTensor((nb_gt_masks, nb_pred_masks))):
            Intersection over Union.
    """
    # compute IOUs
    gt_masks = gt_masks.to(torch.uint8)
    pred_masks = pred_masks.to(torch.uint8)
    nb_gts, nb_preds = gt_masks.shape[2], pred_masks.shape[2]
    ious = torch.zeros((nb_gts, nb_preds), dtype=torch.float)
    gt_areas = gt_masks.sum((0, 1))
    pred_areas = pred_masks.sum((0, 1))
    logging.info(f"{nb_gts} x {nb_preds} (GT x predictions)")
    for gt_idx in range(0, gt_masks.shape[2]):
        gt_mask = gt_masks[:, :, gt_idx]
        for pred_idx in range(0, pred_masks.shape[2]):
            # skip masks whose boxes do not intercept
            if (gt_boxes[gt_idx, 0] > pred_boxes[pred_idx, 2] or
                    pred_boxes[pred_idx, 0] > gt_boxes[gt_idx, 2] or
                    gt_boxes[gt_idx, 1] > pred_boxes[pred_idx, 3] or
                    pred_boxes[pred_idx, 1] > gt_boxes[gt_idx, 3]):
                iou = 0.0
            else:
                intersection = pred_masks[:, :, pred_idx] & gt_mask
                intersection = intersection.nonzero().shape[0]
                union = (pred_areas[pred_idx] + gt_areas[gt_idx]
                         - intersection).item()
                iou = intersection/union if union != 0.0 else 0.0
            ious[gt_idx, pred_idx] = iou
    return ious


@profilable
def compute_map(ious):
    """Compute mean average precision.

    Args:
        ious (torch.FloatTensor((nb_gt_masks, nb_pred_masks))):
            Intersection over Union.

    Returns:
        precision: torch.FloatTensor((1))

    Note: when 2 or more predictions hit the same gt, only 1 hit is counted
    """
    # compute hits
    thresholds = torch.arange(0.5, 1.0, 0.05)
    precisions = torch.empty_like(thresholds, device=Config.DEVICE)
    for thresh_idx, threshold in enumerate(thresholds):
        hits = ious > threshold
        pred_sum = hits.sum(dim=1)
        gt_sum = hits.sum(dim=0)
        tp = pred_sum.nonzero().shape[0]
        fp = (gt_sum == 0).nonzero().shape[0]
        fn = (pred_sum == 0).nonzero().shape[0]
        precisions[thresh_idx] = tp/(tp + fp + fn)

    # average precisions
    return precisions.mean()
