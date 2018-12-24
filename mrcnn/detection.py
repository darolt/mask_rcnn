import numpy as np
import torch

from mrcnn.config import ExecutionConfig as ExeCfg
from mrcnn import utils
from nms.nms_wrapper import nms


def to_input_domain(config, rois, probs, deltas, image_meta):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Args:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns:
        detections: [N, (y1, x1, y2, x2, class_id, score)]
    """
    out = _to_input_domain(config, rois, probs, deltas, image_meta)
    refined_rois, class_ids, class_scores = out
    return torch.cat((refined_rois,
                      class_ids.unsqueeze(1).float(),
                      class_scores.unsqueeze(1)), dim=1)


def _to_input_domain(config, rois, probs, deltas, image_meta):
    # Currently only supports batchsize 1
    rois = rois.squeeze(0)

    _, _, window, _ = utils.parse_image_meta(image_meta)
    window = window[0]

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long().to(ExeCfg.DEVICE)

    class_scores = probs[idx.detach(), class_ids.detach()]
    utils.register_hook(class_scores, 'class_scores:')
    deltas_specific = deltas[idx.detach(), class_ids[0].detach()]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    deltas_specific = (deltas_specific *
                       config.RPN_BBOX_STD_DEV.to(ExeCfg.DEVICE))
    refined_rois = utils.apply_box_deltas(rois.unsqueeze(0),
                                          deltas_specific.unsqueeze(0))
    refined_rois = refined_rois.squeeze(0)

    # Convert coordinates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = torch.from_numpy(np.array([height, width, height, width]))
    scale = scale.to(ExeCfg.DEVICE).float()

    refined_rois = refined_rois * scale
    utils.register_hook(refined_rois, 'refined_rois1:')
    # Clip boxes to image window
    refined_rois = utils.clip_to_window(window, refined_rois)

    # Round and cast to int since we're dealing with pixels now
    utils.register_hook(refined_rois, 'refined_rois2:')
    # refined_rois = torch.round(refined_rois)
    utils.register_hook(refined_rois, 'refined_rois3:')
    return refined_rois, class_ids, class_scores


def _apply_nms(class_ids, class_scores, refined_rois, keep, config):
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    pre_nms_rois = refined_rois[keep]
    for i, class_id in enumerate(utils.unique1d(pre_nms_class_ids)):
        # Pick detections of this class
        ixs = torch.nonzero(pre_nms_class_ids.squeeze(0) == class_id)[:, 0]
        class_rois = pre_nms_rois[ixs]
        class_scores = pre_nms_scores[ixs]
        # Sort
        class_scores, order = class_scores.sort(descending=True)
        class_rois = class_rois[order, :]

        class_keep = nms(
            torch.cat((class_rois, class_scores.unsqueeze(1)), dim=1),
            config.DETECTION_NMS_THRESHOLD)

        # Map indices
        class_keep = keep[ixs[order[class_keep]]]

        if i == 0:
            nms_keep = class_keep
        else:
            nms_keep = utils.unique1d(torch.cat((nms_keep, class_keep)))
    keep = utils.intersect1d(keep, nms_keep)
    return keep


def detection_layer(config, rois, probs, deltas, image_meta):
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
    det_out = _to_input_domain(config, rois, probs, deltas, image_meta)
    refined_rois, class_ids, class_scores = det_out

    # Filter out background boxes
    keep_bool = class_ids > 0

    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        greater_scores = class_scores >= config.DETECTION_MIN_CONFIDENCE
        keep_bool = keep_bool & greater_scores
    keep = torch.nonzero(keep_bool)[:, 0]

    # Apply per-class NMS
    keep = _apply_nms(class_ids, class_scores, refined_rois, keep, config)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    return torch.cat((refined_rois[keep],
                      class_ids[keep].unsqueeze(1).float(),
                      class_scores[keep].unsqueeze(1)), dim=1)
