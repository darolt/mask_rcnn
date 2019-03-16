"""
Utils for analyzing dataset and optimizing the model.

Licensed under The MIT License
Written by Jean Da Rolt
"""

import logging

import numpy as np
from scipy import stats
from skimage.measure import regionprops

from mrcnn.utils import utils


class DatasetAnalyzer:
    def __init__(self, dataset_handler):
        self.dataset_handler = dataset_handler

    def boxes_stats(self):
        """Stores min, max and avg box sides (for height and width)"""
        all_boxes = []
        nb_detections = []
        convexities = []
        all_ids = set()
        for image_id in self.dataset_handler.image_ids:
            masks, ids = self.dataset_handler.load_mask(image_id)
            all_ids = all_ids.union(set(ids))
            boxes = utils.extract_bboxes(masks)
            all_boxes.append(boxes)
            nb_detections.append(boxes.shape[0])
            for mask_idx in range(masks.shape[2]):
                mask = masks[:, :, mask_idx]
                props = regionprops(mask.astype(np.int8))[0]
                convexities.append(props.filled_area/props.convex_area)

        self.nb_classes = len(all_ids) + 1

        convexities = np.array(convexities)
        self.convexity_stats = stats.describe(convexities)

        nb_detections = np.array(nb_detections)
        self.nb_detections_stats = stats.describe(nb_detections)

        all_boxes = np.concatenate(all_boxes, axis=0)
        heights = all_boxes[:, 2] - all_boxes[:, 0]
        widths = all_boxes[:, 3] - all_boxes[:, 1]

        self.height_stats = stats.describe(heights)
        self.width_stats = stats.describe(widths)

        ratios = heights/widths
        self.ratio_stats = stats.describe(ratios)

        mean_pixel = [np.mean(img, axis=(0, 1))
                      for img in self.dataset_handler.images]
        self.mean_pixel = np.mean(np.array(mean_pixel), axis=0)

    def filter(self, result):
        """Filter results according to stats."""
        convexities = []
        for mask_idx in range(result.masks.shape[2]):
            mask = result.masks[:, :, mask_idx]
            props = regionprops(mask.numpy().astype(np.int8))[0]
            convexities.append(props.filled_area/props.convex_area)

        convexities = np.array(convexities)
        convexity_stats = stats.describe(convexities)

        heights = result.rois[:, 2] - result.rois[:, 0]
        widths = result.rois[:, 3] - result.rois[:, 1]
        gt_max_width = widths > self.width_stats.minmax[1]
        lt_min_width = widths < self.width_stats.minmax[0]
        gt_max_height = heights > self.height_stats.minmax[1]
        lt_min_height = heights < self.height_stats.minmax[0]
        keep = ~(gt_max_width | lt_min_width |
                 gt_max_height | lt_min_height)
        initial_size = heights.shape[0]
        new_size = keep.sum()
        if initial_size != new_size:
            logging.info(f"Analyzer filtered {initial_size - new_size}"
                         f" out of {initial_size}.")
        result.masks = result.masks.permute(2, 0, 1)
        new_result = result.select(keep)
        new_result.masks = new_result.masks.permute(1, 2, 0)
        return new_result

    # TODO analyze convexity, forms...


def analyze(dataset_handler):
    analyzer = DatasetAnalyzer(dataset_handler)
    analyzer.boxes_stats()

    return analyzer
