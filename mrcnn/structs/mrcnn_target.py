import torch

from mrcnn.structs.tensor_container import TensorContainer


class MRCNNTarget():
    def __init__(self,
                 class_ids=torch.FloatTensor(),
                 deltas=torch.FloatTensor(),
                 masks=torch.FloatTensor()):
        self.class_ids = class_ids
        self.deltas = deltas
        self.masks = masks
