import logging

import torch

from mrcnn.config import ExecutionConfig as ExeCfg


def compute_map_metric(gt_masks, pred_masks):
    ious = compute_ious(gt_masks, pred_masks)
    precision = compute_map(ious)
    return precision


def compute_ious(gt_masks, pred_masks):
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
    ious = torch.zeros((gt_masks.shape[2], pred_masks.shape[2]),
                       dtype=torch.float)
    logging.info(f"{gt_masks.shape[2]} x {pred_masks.shape[2]}")
    for gt_idx in range(0, gt_masks.shape[2]):
        for pred_idx in range(0, pred_masks.shape[2]):
            intersection = pred_masks[:, :, pred_idx] & gt_masks[:, :, gt_idx]
            intersection = torch.nonzero(intersection).shape[0]
            union = pred_masks[:, :, pred_idx] | gt_masks[:, :, gt_idx]
            union = torch.nonzero(union).shape[0]
            iou = intersection/union if union != 0.0 else 0.0
            ious[gt_idx, pred_idx] = iou
    return ious


def compute_map(ious):
    """Compute mean average precision."""
    # compute hits
    thresholds = torch.arange(0.5, 1.0, 0.05)
    precisions = torch.empty_like(thresholds, device=ExeCfg.DEVICE)
    for thresh_idx, threshold in enumerate(thresholds):
        hits = ious > threshold
        tp = torch.nonzero(hits.sum(dim=1)).shape[0]
        fp = torch.nonzero(hits.sum(dim=0) == 0).shape[0]
        fn = torch.nonzero(hits.sum(dim=1) == 0).shape[0]
        precisions[thresh_idx] = tp/(tp + fp + fn)

    # average precisions
    return precisions.mean()
