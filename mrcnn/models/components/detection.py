
import torch

from tools.config import Config
from mrcnn.models.components.nms import nms_wrapper  # pylint: disable=E0401,E0611
from mrcnn.utils import utils
from mrcnn.utils.exceptions import NoBoxToKeep


def _take_top_detections(probs, deltas):
    """For each ROI, takes TOP probs, ids and deltas."""
    top_class_probs, top_class_ids = probs.max(dim=1)
    top_deltas = deltas[range(top_class_ids.shape[0]), top_class_ids]
    return top_class_probs, top_class_ids, top_deltas


def _to_input_domain(rois, probs, deltas):
    # Currently only supports batchsize 1

    top_class_probs, top_class_ids, top_deltas = \
        _take_top_detections(probs, deltas)

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    top_deltas = top_deltas * Config.RPN.BBOX_STD_DEV_GPU
    refined_rois = utils.apply_box_deltas(
        rois, top_deltas.unsqueeze(0)).squeeze(0)

    # Convert coordinates to image domain
    refined_rois = refined_rois * Config.RPN.NORM
    # Clip boxes to image window
    refined_rois = utils.clip_boxes(
        refined_rois, Config.RPN.CLIP_WINDOW, squeeze=True)

    return refined_rois, top_class_ids, top_class_probs


def _apply_nms(class_ids, class_probs, refined_rois, keep):
    pre_nms_class_ids = class_ids[keep]
    pre_nms_probs = class_probs[keep]
    pre_nms_rois = refined_rois[keep]
    for class_id in pre_nms_class_ids.unique():
        # Pick detections of this class
        class_idxs = (pre_nms_class_ids == class_id).nonzero().squeeze(1)
        if class_idxs.nelement() == 0:
            continue
        class_rois = pre_nms_rois[class_idxs]
        class_probs = pre_nms_probs[class_idxs]
        # Sort
        class_probs, order = class_probs.sort(descending=True)
        class_rois = class_rois[order, :]

        class_keep = nms_wrapper.nms_indexes(
            class_rois.unsqueeze(0),
            class_probs.unsqueeze(0),
            Config.DETECTION.NMS_THRESHOLD,
            class_rois.shape[0]).squeeze(0)

        class_keep = class_keep.unique()
        # Map indices back to keep list
        class_keep = keep[class_idxs[order[class_keep]]]
        keep = utils.set_intersection(keep.unique(), class_keep.unique())

    return keep


def detection_layer(rois, probs, deltas):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """
    det_out = _to_input_domain(rois, probs, deltas)
    refined_rois, class_ids, class_probs = det_out

    # Filter out background boxes
    keep_fg = class_ids > 0

    # Filter out low confidence boxes
    if Config.DETECTION.MIN_CONFIDENCE:
        greater_scores = class_probs >= Config.DETECTION.MIN_CONFIDENCE
        keep_fg = keep_fg & greater_scores
    keep = keep_fg.nonzero().squeeze(1)

    if keep.nelement() == 0:
        raise NoBoxToKeep

    # Apply per-class NMS
    keep = _apply_nms(class_ids, class_probs, refined_rois, keep)

    if keep.nelement() == 0:
        raise NoBoxToKeep

    # Keep top detections
    _, top_scores_idxs = class_probs[keep].sort(descending=True)[1]
    top_scores_idxs = top_scores_idxs[:Config.DETECTION.MAX_INSTANCES]
    keep = keep[top_scores_idxs]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    return torch.cat((refined_rois[keep],
                      class_ids[keep].unsqueeze(1).float(),
                      class_probs[keep].unsqueeze(1)), dim=1)
