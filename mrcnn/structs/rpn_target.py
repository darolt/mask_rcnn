import torch

from mrcnn.structs.tensor_container import TensorContainer


class RPNTarget(TensorContainer):
    def __init__(self,
                 match=torch.FloatTensor(),
                 deltas=torch.FloatTensor()):
        self.match = match
        self.deltas = deltas
