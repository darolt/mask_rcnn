"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import random
import warnings

import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F

from mrcnn.structs.detection_output import DetectionOutput
from mrcnn.utils.exceptions import NoBoxHasPositiveArea
from mrcnn.utils.image_metas import ImageMetas
from tools.config import Config


############################################################
#  Bounding Boxes
############################################################


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.

    Args:
        boxes: [batch_size, N, 4] where each row is y1, x1, y2, x2
        deltas: [batch_size, N, 4] where each row is [dy, dx, log(dh), log(dw)]

    Returns:
        results: [batch_size, N, 4], where each row is [y1, x1, y2, x2]
    """
    # Convert to y, x, h, w
    height = boxes[:, :, 2] - boxes[:, :, 0]
    width = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + 0.5 * height
    center_x = boxes[:, :, 1] + 0.5 * width
    # Apply deltas
    center_y = center_y + deltas[:, :, 0] * height
    center_x = center_x + deltas[:, :, 1] * width
    height = height * torch.exp(deltas[:, :, 2])
    width = width * torch.exp(deltas[:, :, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=2)
    return result


def clip_boxes(boxes, window, squeeze=False):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    if squeeze:
        boxes = boxes.unsqueeze(0)
    boxes = torch.stack(
        [boxes[:, :, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, :, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, :, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, :, 3].clamp(float(window[1]), float(window[3]))], 2)
    if squeeze:
        boxes = boxes.squeeze(0)
    return boxes


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 = x2 + 1
            y2 = y2 + 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    return torch.stack([dy, dx, dh, dw], dim=1) / Config.BBOX_STD_DEV


def subtract_mean(images):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - Config.IMAGE.MEAN_PIXEL


def mold_image(image):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    molded_image, image_metas = resize_image(
        image,
        min_dim=Config.IMAGE.MIN_DIM,
        max_dim=Config.IMAGE.MAX_DIM,
        min_scale=Config.IMAGE.MIN_SCALE,
        mode=Config.IMAGE.RESIZE_MODE)
    molded_image = subtract_mean(molded_image)

    return molded_image, image_metas


def mold_inputs(images):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matricies [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matricies:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    """
    molded_images = []
    images_metas = []
    # windows = []
    for image in images:
        # Resize image to fit the model expected size
        molded_image, image_metas = mold_image(image)
        molded_images.append(molded_image)
        images_metas.append(image_metas)
    # Pack into arrays
    molded_images = np.stack(molded_images)
    return molded_images, image_metas


def unmold_detections(detections, mrcnn_mask, image_metas):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    Args:
        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_metas: ImageMetas object, contains meta about image

    Returns:
        DetectionOutput object. Rois, class_ids, scores and masks.
    """
    nb_dets = detections.shape[0]
    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:nb_dets, :4]
    class_ids = detections[:nb_dets, 4].to(torch.long)
    scores = detections[:nb_dets, 5]
    masks = mrcnn_mask[torch.arange(nb_dets, dtype=torch.long),
                       :, :, class_ids]
    final_rois, final_class_ids, final_scores, final_masks = \
        unmold_boxes(boxes, class_ids, masks, image_metas, scores)
    return DetectionOutput(final_rois, final_class_ids, final_scores,
                           final_masks)


def unmold_boxes(boxes, class_ids, masks, image_metas, scores=None):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)]
    masks: [N, height, width]
    image_shape: [height, width, depth] Original size of the image
                 before resizing
    window: [y1, x1, y2, x2] Box in the image where the real image is
            excluding the padding.

    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    """
    # Extract boxes, class_ids, scores, and class-specific masks
    class_ids = class_ids.to(torch.long)

    boxes = to_img_domain(boxes, image_metas).to(torch.int32)

    boxes, class_ids, masks, scores = remove_zero_area(boxes, class_ids,
                                                       masks, scores)

    full_masks = unmold_masks(masks, boxes, image_metas)

    return boxes, class_ids, scores, full_masks


def resize_image(image, min_dim=None, max_dim=None, min_scale=None,
                 mode='square'):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    original_shape = image.shape
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == 'none':
        return image, ImageMetas(original_shape, window,
                                 scale, padding, crop)

    # Scale?
    if min_dim and mode != 'pad64':
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale:
        scale = max(scale, min_scale)
    if mode == 'pad64':
        scale = min_dim/max(h, w)

    # Does it exceed max dim?
    if max_dim and mode == 'square':
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            image = skimage.transform.resize(
                image, (round(h * scale), round(w * scale)),
                order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    h, w = image.shape[:2]
    if mode == 'square':
        # Get new height and width
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == 'pad64':
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, 'Minimum dimension must be a multiple of 64'
        # Height
        if min_dim != h:
            # max_h = h - (min_dim % 64) + 64
            max_h = min_dim
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if max_dim != w:
            # max_w = w - (max_dim % 64) + 64
            max_w = max_dim
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        # TODO: zero is ok as padding value?
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == 'crop':
        # Pick a random crop
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception(f"Mode {mode} not supported")
    return (image.astype(image_dtype),
            ImageMetas(original_shape, window, scale, padding, crop))


def resize_mask(mask, scale, padding, crop):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_masks(boxes, masks, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_shape = tuple(mini_shape)
    mini_masks = np.zeros(mini_shape + (masks.shape[-1],), dtype=bool)
    for i in range(masks.shape[-1]):
        m = masks[:, :, i].astype(bool)
        y1, x1, y2, x2 = boxes[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = skimage.transform.resize(m, mini_shape, order=1,
                                         mode="constant")
        mini_masks[:, :, i] = np.around(m).astype(np.bool)
    return mini_masks


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
        mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    return mask


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    shape = (y2 - y1, x2 - x1)

    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=shape,
                         mode='bilinear', align_corners=True)
    mask = mask.squeeze(0).squeeze(0)
    mask = torch.where(mask >= threshold,
                       torch.tensor(1, device=Config.DEVICE),
                       torch.tensor(0, device=Config.DEVICE))

    # Put the mask in the right location.
    full_mask = torch.zeros(image_shape[:2], dtype=torch.uint8)
    full_mask[y1:y2, x1:x2] = mask.to(torch.uint8)
    return full_mask


def unmold_masks(masks, boxes, image_metas):
    # Resize masks to original image size and set boundary threshold.
    image_shape = (image_metas.original_shape[0],
                   image_metas.original_shape[1])
    nb_masks = masks.shape[0]
    full_masks = []
    for i in range(nb_masks):
        # Convert neural network mask to full size mask
        full_mask = unmold_mask(masks[i], boxes[i], image_shape)
        full_masks.append(full_mask)
    full_masks = torch.stack(full_masks, dim=-1)\
        if full_masks else torch.empty((0,) + masks.shape[1:3])
    return full_masks


def remove_zero_area(boxes, class_ids, masks, scores=None):
    # Filter out detections with zero area. Often only happens in early
    # stages of training when the network weights are still a bit random.
    dx, dy = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    too_small = dx * dy <= 0.0
    too_short = dx <= 2.0
    too_thin = dy <= 2.0
    skip = too_small + too_short + too_thin
    positive_area = torch.nonzero(skip == 0)
    if positive_area.nelement() == 0:
        raise NoBoxHasPositiveArea

    keep_ix = positive_area[:, 0]
    if keep_ix.shape[0] != boxes.shape[0]:
        boxes = boxes[keep_ix]
        class_ids = class_ids[keep_ix]
        scores = scores[keep_ix] if scores is not None else None
        masks = masks[keep_ix]
    return boxes, class_ids, masks, scores


def to_img_domain(boxes, image_metas):
    image_shape = torch.tensor(image_metas.original_shape,
                               dtype=torch.float32,
                               device=Config.DEVICE)
    window = torch.tensor(image_metas.window,
                          dtype=torch.float32,
                          device=Config.DEVICE)
    # Compute shift to translate coordinates to image domain.
    shifts = torch.tensor([window[0], window[1], window[0], window[1]],
                          device=Config.DEVICE)

    # Translate bounding boxes to image domain
    boxes = ((boxes - shifts)/image_metas.scale)
    original_box = (0, 0, image_shape[0], image_shape[1])
    boxes = clip_boxes(boxes, original_box, squeeze=True)
    return boxes


def to_mini_mask(rois, boxes):
    """
    Transform ROI coordinates from normalized image space
    to normalized mini-mask space.
    """
    y1, x1, y2, x2 = rois.chunk(4, dim=1)
    gt_y1, gt_x1, gt_y2, gt_x2 = boxes.chunk(4, dim=1)
    gt_h = gt_y2 - gt_y1
    gt_w = gt_x2 - gt_x1
    y1 = (y1 - gt_y1) / gt_h
    x1 = (x1 - gt_x1) / gt_w
    y2 = (y2 - gt_y1) / gt_h
    x2 = (x2 - gt_x1) / gt_w
    return torch.cat([y1, x1, y2, x2], dim=1)


def set_intersection(tensor1, tensor2):
    """Intersection of elements present in tensor1 and tensor2.
    Note: it only works if elements are unique in each tensor.
    """
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).detach()]


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding."""

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[-1]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom),
                     'constant', 0)

    def __repr__(self):
        return self.__class__.__name__
