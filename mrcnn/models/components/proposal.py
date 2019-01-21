import logging

import numpy as np
import torch
import torch.nn.functional as F

from tools.config import Config
from mrcnn.utils import utils
from nms.nms_wrapper import nms


def proposal_layer(scores, deltas, proposal_count, nms_threshold, anchors):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment to anchors. Proposals are zero padded.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = scores[:, :, 1]

    deltas = deltas * Config.RPN.BBOX_STD_DEV.to(Config.DEVICE)

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(Config.PROPOSALS.PRE_NMS_LIMIT, anchors.shape[1])
    scores, order = scores.topk(pre_nms_limit)

    order = order.unsqueeze(2).expand(-1, -1, 4)
    deltas = deltas.gather(1, order)
    anchors = anchors.gather(1, order)

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = utils.apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = Config.IMAGE.SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = utils.clip_boxes(boxes, window)

    # Non-max suppression
    boxes = _apply_nms(boxes, scores, nms_threshold, proposal_count)

    # Normalize dimensions to range of 0 to 1.
    norm = torch.tensor(np.array([height, width, height, width]),
                        requires_grad=False, dtype=torch.float32,
                        device=Config.DEVICE)
    normalized_boxes = boxes / norm

    return normalized_boxes


def _apply_nms(boxes, scores, nms_threshold, proposal_count):
    nms_input = torch.cat((boxes,
                           scores.reshape((boxes.shape[0:2] + (1,)))),
                          2)
    boxes_end = []
    for img_idx in range(boxes.shape[0]):
        keep = nms(nms_input[img_idx], nms_threshold)
        keep = keep[:proposal_count]
        pad_size = proposal_count - keep.shape[0]
        boxes_ = boxes[img_idx, keep, :]
        if pad_size > 0:
            logging.debug(f"Proposal pad size after NMS is {pad_size}")
            boxes_ = F.pad(boxes_, (0, 0, 0, pad_size), value=0)
        boxes_end.append(boxes_.unsqueeze(0))

    boxes = torch.cat(boxes_end, 0)
    return boxes
