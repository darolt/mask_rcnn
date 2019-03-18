
from mrcnn.models.components.nms import nms_wrapper  # pylint: disable=E0611
from mrcnn.utils import utils
from tools.config import Config
from tools.time_profiling import profilable


@profilable
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

    deltas = deltas*Config.RPN.BBOX_STD_DEV_GPU

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
    boxes = utils.clip_boxes(boxes, Config.RPN.CLIP_WINDOW)

    # Non-max suppression
    boxes = nms_wrapper.nms_wrapper(
        boxes,
        scores,
        nms_threshold,
        proposal_count)

    # Normalize dimensions to range of 0 to 1.
    normalized_boxes = boxes/Config.RPN.NORM

    return normalized_boxes
