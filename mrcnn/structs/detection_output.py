
from mrcnn.structs.tensor_container import TensorContainer


class DetectionOutput(TensorContainer):
    def __init__(self, rois, class_ids, scores, masks):
        self.rois = rois
        self.class_ids = class_ids
        self.scores = scores
        self.masks = masks
