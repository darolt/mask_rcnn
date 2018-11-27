import logging

import numpy as np
import torch
from torch.nn.functional import pad

import mrcnn.config
from mrcnn import utils
from nms.nms_wrapper import nms


############################################################
#  Proposal Layer
############################################################


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [batch_size, N, 4] where each row is y1, x1, y2, x2
    deltas: [batch_size, N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, :, 2] - boxes[:, :, 0]
    width = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + 0.5 * height
    center_x = boxes[:, :, 1] + 0.5 * width
    # Apply deltas
    center_y = center_y + deltas[:, :, 0] * height
    center_x = center_x + deltas[:, :, 1] * width
    height = height * torch.exp(deltas[:, :, 2])
    width = width * torch.exp(deltas[:, :, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=2)
    return result


def proposal_layer(scores, deltas, proposal_count, nms_threshold,
                   anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors. Proposals are zero padded.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    # TODO this only accepts one class
    scores = scores[:, :, 1]

    # TODO constant
    std_dev = torch.from_numpy(
                  np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])
              ).float()
    std_dev = std_dev.to(mrcnn.config.DEVICE)
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    # TODO 6000 is hardcoded
    pre_nms_limit = min(6000, anchors.size()[1])
    scores, order = scores.topk(pre_nms_limit)

    expanded_order = order.view(order.size()[0], -1, 1).expand(-1, -1, 4)
    deltas = deltas.gather(1, expanded_order)
    anchors = anchors.gather(1, expanded_order)

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = utils.clip_boxes(boxes, window)

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    nms_input = torch.cat((boxes,
                           scores.reshape((boxes.size()[0:2] + (1,)))),
                          2).detach()
    boxes_end = []
    for batch in range(boxes.size()[0]):
        keep = nms(nms_input[batch], nms_threshold)
        keep = keep[:proposal_count]
        pad_size = proposal_count - keep.shape[0]
        boxes_ = boxes[batch, keep, :]
        if pad_size > 0:
            logging.info(f"Proposal pad size after NMS is {pad_size}")
            boxes_ = pad(boxes_, (0, 0, 0, pad_size), value=0)
        boxes_end.append(boxes_.unsqueeze(0))

    boxes = torch.cat(boxes_end, 0)
    # boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.
    norm = torch.tensor(np.array([height, width, height, width]),
                        requires_grad=False,
                        dtype=torch.float32)
    norm = norm.to(mrcnn.config.DEVICE)
    normalized_boxes = boxes / norm

    return normalized_boxes
