
import torch
from torch import isnan

from mrcnn.models.components.roialign.crop_and_resize_function \
    import CropAndResizeFunction
from tools.config import Config
from tools.time_profiling import profilable


@profilable
def pyramid_roi_align(boxes, feature_maps, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    # Currently only supports batchsize 1
    boxes = boxes.squeeze(0)

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    # Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = (y2 - y1).float()
    w = (x2 - x1).float()

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = torch.tensor([float(image_shape[0]*image_shape[1])],
                              dtype=torch.float32, device=Config.DEVICE)
    roi_level = 4 + torch.log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)

    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        assert isnan(feature_maps[i]).sum() == 0, 'Feature map contains NaNs.'
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.detach(), :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.detach())

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = torch.zeros(level_boxes.shape[0], dtype=torch.int,
                          device=Config.DEVICE)
        # CropAndResizeFunction needs batch dimension
        pooled_features = (CropAndResizeFunction(pool_size, pool_size, 0)
                           (feature_maps[i], level_boxes, ind))
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled
