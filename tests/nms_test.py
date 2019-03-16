"""
Test for Non-Maximum Suppression.

Licensed under the MIT License
Written by Jean Da Rolt
"""
import logging
import sys

import torch

from mrcnn.models.components.nms import nms_wrapper  # pylint: disable=E0611


logging.basicConfig(stream=sys.stderr, level=logging.INFO)

NMS_INPUT = './tests/nms_input.pt'
NMS_OUTPUT = './tests/nms_output.pt'
PROPOSAL_COUNT = 2000
THRESHOLD = 0.9
DEVICE = 1


def nms_test():
    logging.info('Testing NMS...')
    logging.debug('Loading golden truth input and outputs...')
    gt_nms_in = torch.load(NMS_INPUT)
    gt_nms_out = torch.load(NMS_OUTPUT)

    logging.debug('Running NMS...')
    with torch.cuda.device(DEVICE):
        scores = gt_nms_in.select(2, 4)
        boxes = gt_nms_in[:, :, 0:4]
        nms_out = nms_wrapper.nms_wrapper(boxes, scores, THRESHOLD,
                                          PROPOSAL_COUNT)

    if torch.equal(nms_out, gt_nms_out):
        logging.info('NMS passed.')
        return 0

    logging.info('NMS failed!')
    return 1


if __name__ == '__main__':
    nms_test()
