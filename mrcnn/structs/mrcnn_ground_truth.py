import torch

from mrcnn.structs.tensor_container import TensorContainer


class MRCNNGroundTruth(TensorContainer):
    def __init__(self,
                 class_ids=torch.FloatTensor(),
                 boxes=torch.FloatTensor(),
                 masks=torch.FloatTensor()):
        self.class_ids = class_ids
        self.boxes = boxes
        self.masks = masks
