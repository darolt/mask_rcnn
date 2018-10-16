import torch
import numpy as np
import mrcnn.config

from mrcnn import utils
from mrcnn.proposal import apply_box_deltas
from nms.nms_wrapper import nms


############################################################
#  Detection Layer
############################################################


def detection_layer2(config, rois, probs, deltas, image_meta):
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
    refined_rois, class_ids, class_scores = detection_layer3(config, rois,
                                                             probs, deltas,
                                                             image_meta)
    return torch.cat((refined_rois,
                     class_ids.unsqueeze(1).float(),
                     class_scores.unsqueeze(1)), dim=1)


def detection_layer3(config, rois, probs, deltas, image_meta):
    # Currently only supports batchsize 1
    rois = rois.squeeze(0)

    _, _, window, _ = utils.parse_image_meta(image_meta)
    window = window[0]

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    idx = idx.to(mrcnn.config.DEVICE)
    class_scores = probs[idx, class_ids.detach()]
    deltas_specific = deltas[idx, class_ids[0].detach()]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = torch.from_numpy(
                  np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])
              ).float()
    std_dev = std_dev.to(mrcnn.config.DEVICE)
    refined_rois = apply_box_deltas(rois.unsqueeze(0),
                                    (deltas_specific * std_dev).unsqueeze(0))
    refined_rois = refined_rois.squeeze(0)

    # Convert coordinates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = torch.from_numpy(np.array([height, width, height, width])).float()
    scale = scale.to(mrcnn.config.DEVICE)
    refined_rois *= scale

    # Clip boxes to image window
    refined_rois = utils.clip_to_window(window, refined_rois)

    # Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    return refined_rois, class_ids, class_scores


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
    det_out = detection_layer3(config, rois, probs, deltas, image_meta)
    refined_rois, class_ids, class_scores = det_out

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep_bool = class_ids > 0

    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool &= (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    keep = torch.nonzero(keep_bool)[:, 0]

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep.detach()]
    pre_nms_scores = class_scores[keep.detach()]
    pre_nms_rois = refined_rois[keep.detach()]
    for i, class_id in enumerate(utils.unique1d(pre_nms_class_ids)):
        # Pick detections of this class
        ixs = torch.nonzero(pre_nms_class_ids.squeeze(0) == class_id)[:, 0]
        ix_rois = pre_nms_rois[ixs.detach()]
        ix_scores = pre_nms_scores[ixs.detach()]
        # Sort
        ix_scores, order = ix_scores.sort(descending=True, )
        ix_rois = ix_rois[order.detach(), :]

        class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)),
                                   dim=1).detach(),
                         config.DETECTION_NMS_THRESHOLD)

        # Map indices
        class_keep = keep[ixs[order[class_keep].detach()].detach()]

        if i == 0:
            nms_keep = class_keep
        else:
            nms_keep = utils.unique1d(torch.cat((nms_keep, class_keep)))
    keep = utils.intersect1d(keep, nms_keep)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.detach()].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.detach()]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    return torch.cat((refined_rois[keep.detach()],
                     class_ids[keep.detach()].unsqueeze(1).float(),
                     class_scores[keep.detach()].unsqueeze(1)), dim=1)
