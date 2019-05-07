
import numpy as np
import torch as th


def generate_anchors(scale, ratios, shape, feature_stride, anchor_stride):
    """
    scale: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scale and ratios
    scales, ratios = np.meshgrid(np.array(scale), np.array(ratios))
    scales = scales.flatten()
    ratios = np.sqrt(ratios.flatten())

    # Enumerate heights and widths from scales and ratios
    heights = scales / ratios
    widths = scales * ratios

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Meshgrid of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    print(boxes.shape)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride, batch_size):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    anchors = []
    for i, scale in enumerate(scales):
        anchors.append(generate_anchors(scale, ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    anchors = np.concatenate(anchors, axis=0)
    new_anchors_shape = (batch_size,) + anchors.shape
    anchors = np.broadcast_to(anchors, new_anchors_shape)

    return th.from_numpy(anchors).float()
