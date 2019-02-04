import torch

from torch.autograd import Function

import crop_and_resize as _backend


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        crops = torch.zeros_like(image)

        _backend.crop_and_resize_gpu_forward(
            image, boxes, box_ind, self.extrapolation_value,
            self.crop_height, self.crop_width, crops)

        # save for backward
        self.im_size = image.size()
        self.save_for_backward(boxes, box_ind)

        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        grad_image = torch.zeros_like(grad_outputs).resize_(*self.im_size)

        _backend.crop_and_resize_gpu_backward(
            grad_outputs, boxes, box_ind, grad_image
        )

        return grad_image, None, None
