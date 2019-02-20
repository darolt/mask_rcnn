"""
Differentiable MaP loss
Not used anymore.
"""
import logging
import numpy as np

# from mrcnn.utils.utils import unmold_boxes_x, unmold_detections_x

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
    indices_y = indices_y + y_range.to(Config.DEVICE)
    indices_x = indices_x + x_range.to(Config.DEVICE)
    indices = torch.stack((indices_y, indices_x), dim=2)
    shape = torch.FloatTensor((mask.shape[0], mask.shape[1])).to(Config.DEVICE)
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
    factor = pred_inter_idx.sum().to(Config.DEVICE)
    factor = (factor/100).sigmoid() - 1.0

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
    precisions = torch.empty_like(thresholds, device=Config.DEVICE)
    for thresh_idx, threshold in enumerate(thresholds):
        hits = ((ious - threshold)*MAGNIFIER).sigmoid()
        gt_sum = ((hits.sum(dim=0) - 0.5)*MAGNIFIER).sigmoid()
        pred_sum = ((hits.sum(dim=1) - 0.5)*MAGNIFIER).sigmoid()

        tp = pred_sum.sum()
        overpred = gt_sum.sum() - tp
        if overpred > 0.5:
            print(overpred)
        fp = (1 - gt_sum).sum()
        fn = (1 - pred_sum).sum()
        precisions[thresh_idx] = (tp/(tp + fp + fn)) - overpred

    # average precisions
    return precisions.mean()
