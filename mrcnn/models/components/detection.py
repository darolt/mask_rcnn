
import numpy as np
import torch

from tools.config import Config
from mrcnn.utils import utils
from mrcnn.utils.image_metas import ImageMetasBuilder
import nms_wrapper  # pylint: disable=E0401


def _to_input_domain(rois, probs, deltas, image_meta):
    # Currently only supports batchsize 1
    rois = rois.squeeze(0)

    image_metas = ImageMetasBuilder.from_numpy(image_meta)
    window = image_metas.window
    # _, _, window, _ = utils.parse_image_meta(image_meta)
    # window = window[0]

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long().to(Config.DEVICE)

    class_scores = probs[idx.detach(), class_ids.detach()]
    deltas_specific = deltas[idx.detach(), class_ids[0].detach()]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    deltas_specific = (deltas_specific*Config.RPN.BBOX_STD_DEV_GPU)
    refined_rois = utils.apply_box_deltas(rois.unsqueeze(0),
                                          deltas_specific.unsqueeze(0))
    refined_rois = refined_rois.squeeze(0)

    # Convert coordinates to image domain
    height, width = Config.IMAGE.SHAPE[:2]
    scale = torch.from_numpy(np.array([height, width, height, width]))
    scale = scale.to(Config.DEVICE).float()

    refined_rois = refined_rois * scale
    # Clip boxes to image window
    refined_rois = utils.clip_to_window(window, refined_rois)

    # Round and cast to int since we're dealing with pixels now
    return refined_rois, class_ids, class_scores


def _apply_nms(class_ids, class_scores, refined_rois, keep):
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_rois = refined_rois[keep]
    for i, class_id in enumerate(pre_nms_class_ids.unique()):
        # Pick detections of this class
        class_idxs = (pre_nms_class_ids == class_id).nonzero().squeeze(1)
        class_rois = pre_nms_rois[class_idxs]
        class_scores = pre_nms_scores[class_idxs]
        # Sort
        class_scores, order = class_scores.sort(descending=True)
        class_rois = class_rois[order, :]

        class_keep = nms_wrapper.nms_indexes(
            class_rois.unsqueeze(0),
            class_scores.unsqueeze(0),
            Config.DETECTION.NMS_THRESHOLD,
            class_rois.shape[0]).squeeze(0)

        # remove some boxes that were deleted by NMS
        class_keep = class_keep[class_keep != -1]
        # Map indices
        class_keep = keep[class_idxs[order[class_keep]]]

        if i == 0:
            nms_keep = class_keep
        else:
            nms_keep = (torch.cat((nms_keep, class_keep))).unique()
    keep = utils.intersect1d(keep, nms_keep)
    return keep


def detection_layer(rois, probs, deltas, image_meta):
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
    det_out = _to_input_domain(rois, probs, deltas, image_meta)
    refined_rois, class_ids, class_scores = det_out

    # Filter out background boxes
    keep_fg = class_ids > 0

    # Filter out low confidence boxes
    if Config.DETECTION.MIN_CONFIDENCE:
        greater_scores = class_scores >= Config.DETECTION.MIN_CONFIDENCE
        keep_fg = keep_fg & greater_scores
    keep = keep_fg.nonzero().squeeze(1)

    # Apply per-class NMS
    keep = _apply_nms(class_ids, class_scores, refined_rois, keep)

    # Keep top detections
    roi_count = Config.DETECTION.MAX_INSTANCES
    top_scores_idxs = class_scores[keep].sort(descending=True)[1][:roi_count]
    keep = keep[top_scores_idxs]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    return torch.cat((refined_rois[keep],
                      class_ids[keep].unsqueeze(1).float(),
                      class_scores[keep].unsqueeze(1)), dim=1)
