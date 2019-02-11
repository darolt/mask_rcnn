
import numpy as np

import torch

from mrcnn.data.data_generator import load_image_gt
from mrcnn.functions.losses import compute_map_loss
from mrcnn.functions.metrics import compute_map_metric
from mrcnn.utils import utils
from tools.config import Config


def compute_metric(model, dataset_handler, config):
    """Run detection on dataset and compute Kaggle's 2018 Databowl metric."""
    print(f"Running on {dataset_handler.dataset_dir}")

    # Load over images
    precisions = torch.empty((len(dataset_handler)), device=Config.DEVICE)
    for idx, image_id in enumerate(dataset_handler.image_ids):
        # Load image and run detection
        image = dataset_handler.load_image(image_id)
        masks, _ = dataset_handler.load_mask(image_id)
        gt_masks = torch.from_numpy(masks.astype(int))
        gt_bboxes = utils.extract_bboxes(gt_masks)
        gt_bboxes = torch.from_numpy(gt_bboxes).int()
        # Detect objects
        predictions, image_metas = model.detect([image])
        precision = compute_map_metric(gt_masks, predictions['masks'])
        precisions[idx] = precision
        print(f"mAP1 {precision}")

        _, _, gt_class_ids, gt_boxes, gt_masks = load_image_gt(
            dataset_handler, config, image_id, use_mini_mask=True)

        gt_boxes = torch.from_numpy(gt_boxes)
        gt_class_ids = torch.from_numpy(gt_class_ids).to(Config.DEVICE)
        gt_masks = torch.from_numpy(gt_masks.astype(np.uint8))
        gt_boxes = gt_boxes.to(Config.DEVICE).to(torch.float32)
        gt_masks = gt_masks.permute(2, 0, 1)
        pred_masks = predictions['mrcnn_masks'].squeeze(0).to(Config.DEVICE).float()
        precision = compute_map_loss(
            gt_masks.to(Config.DEVICE).float(),
            gt_boxes.to(Config.DEVICE).float(),
            gt_class_ids,
            image_metas.squeeze(0),
            predictions['detections'].squeeze(0).to(Config.DEVICE).float(),
            pred_masks)
        print(f"precision {precision}")

    print(f"final precision: {precisions.mean()}")
