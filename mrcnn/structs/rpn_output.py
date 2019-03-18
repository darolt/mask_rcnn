
import torch

from mrcnn.structs.tensor_container import TensorContainer


class RPNOutput(TensorContainer):
    def __init__(self,
                 class_logits=torch.FloatTensor(),
                 classes=torch.IntTensor(),
                 deltas=torch.FloatTensor()):
        self.class_logits = class_logits
        self.classes = classes
        self.deltas = deltas
