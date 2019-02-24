
from mrcnn.structs.tensor_container import TensorContainer


class DetectionOutput(TensorContainer):
    """
    MRCNN detection output.
        rois: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
    """
    def __init__(self, rois, class_ids, scores, masks):
        self.rois = rois
        self.class_ids = class_ids
        self.scores = scores
        self.masks = masks
