
import torch

from mrcnn.structs.tensor_container import TensorContainer
from tools.config import Config


class MRCNNTarget(TensorContainer):
    def __init__(self,
                 mask_shape,
                 class_ids=torch.IntTensor(),
                 deltas=torch.FloatTensor(),
                 masks=torch.FloatTensor()):
        self._mask_shape = mask_shape
        self.class_ids = class_ids
        self.deltas = deltas
        self.masks = masks

    def zeros(self, size):
        self.class_ids = torch.zeros(size, dtype=torch.int)
        self.deltas = torch.zeros(size, 4, dtype=torch.float32)
        self.masks = torch.zeros(size, self._mask_shape[0],
                                 self._mask_shape[1])
        return self

    def fill_zeros(self, fill_size):
        zeros = torch.zeros(fill_size, dtype=torch.int, device=Config.DEVICE)
        self.class_ids = torch.cat([self.class_ids, zeros], dim=0)
        zeros = torch.zeros(fill_size, 4, dtype=torch.float32,
                            device=Config.DEVICE)
        self.deltas = torch.cat([self.deltas, zeros], dim=0)
        zeros = torch.zeros(
            fill_size, self._mask_shape[0], self._mask_shape[1],
            dtype=torch.float32, device=Config.DEVICE)
        self.masks = torch.cat([self.masks, zeros], dim=0)
        return self
