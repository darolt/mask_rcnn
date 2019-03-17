"""
Make predictions on test set, compute metric and prepare submission file.

Licensed under The MIT License
Written by Jean Da Rolt
"""
import datetime
import json
import logging
import os

import matplotlib
import torch

from mrcnn.functions.metrics import compute_map_metric
from mrcnn.utils import utils, visualize
from mrcnn.utils.exceptions import NoBoxHasPositiveArea, NoBoxToKeep
from mrcnn.utils.rle import mask_to_rle
from tools.config import Config


def submit(model, dataset, results_dir, analyzer=None):
    """Run detection on images in the given directory."""

    Config.dump(os.path.join(model.log_dir, 'config.yml'))
    matplotlib.use('Agg')
    logging.info(f"Running on {dataset.dataset_dir}")

    # Create results directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    submit_dir = "submit_{:%Y.%m.%d_%H:%M:%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(results_dir, submit_dir)
    os.makedirs(submit_dir)

    # Predict on dataset images, save predictions and convert masks to RLE
    submission = []
    summary = {}
    precisions = torch.empty((len(dataset)), device=Config.DEVICE)
    for image_id in dataset.image_ids:
        logging.debug(f"Predicting for image {image_id}")
        # Load image and run detection
        image = dataset.load_image(image_id)
        image_name = dataset.image_info[image_id]['id']
        # Detect objects
        try:
            result, _ = model.detect(image)
        except (NoBoxHasPositiveArea, NoBoxToKeep) as e:
            print(e)
            continue

        if analyzer is not None:
            result = analyzer.filter(result)

        # Compute metric
        gt_masks, _ = dataset.load_mask(image_id)
        gt_boxes = torch.from_numpy(
            utils.extract_bboxes(gt_masks)).to(Config.DEVICE)
        gt_masks = torch.from_numpy(gt_masks.astype(int))
        precision = compute_map_metric(gt_masks, result.masks,
                                       gt_boxes, result.rois)
        logging.info(f"{image_name} MaP: {precision}")
        precisions[image_id] = precision
        summary[image_name] = {'precision': float(precision.item()),
                               'nb_gts': gt_masks.shape[2],
                               'nb_preds': result.masks.shape[2]}

        result.cpu().numpy()
        # Encode image to RLE. Returns a string of multiple lines
        rle = mask_to_rle(image_name, result.masks, result.scores)
        submission.append(rle)
        # Save image with masks
        fig = visualize.display_instances(
            image, result.rois, result.masks, result.class_ids,
            dataset.class_names, result.scores,
            show_bbox=True, show_mask_pixels=False,
            title=f"Predictions for {image_name}")
        fig.savefig(f"{submit_dir}/{image_name}.png")
        matplotlib.pyplot.close()

    logging.info(f"Mean MaP: {precisions.mean()}")
    summary['mean_MaP'] = float(precisions.mean().item())
    # Save submission to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)

    # Save summary to json file
    file_path = os.path.join(submit_dir, 'summary.json')
    with open(file_path, 'w') as fp:
        json.dump(summary, fp, indent=4)

    file_path = os.path.join(submit_dir, 'submit.csv')
    with open(file_path, 'w') as fp:
        fp.write(submission)
    logging.info(f"Saved to {submit_dir}")
