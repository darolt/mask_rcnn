import torch

from mrcnn.structs.tensor_container import TensorContainer


class MRCNNOutput(TensorContainer):
    def __init__(self,
                 class_logits=torch.FloatTensor(),
                 deltas=torch.FloatTensor(),
                 masks=torch.FloatTensor()):
        self.class_logits = class_logits
        self.deltas = deltas
        self.masks = masks
