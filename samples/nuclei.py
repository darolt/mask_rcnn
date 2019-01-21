"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights, example:
    python samples/nuclei.py train --dataset=/home/jro/wk/kaggle/input/ --model=coco

    # Detect nuclei from dataset, example:
    python samples/nuclei.py detect --dataset=/home/jro/wk/kaggle/input/ --model=/home/jro/wk/kaggle/old_nuclei/logs/nuclei20180531_1649/mask_rcnn_nuclei_3.pth

    # Get Kaggle's 2018 Databowl metric from dataset, example:
    python samples/nuclei.py metric --dataset=/home/jro/wk/kaggle/input/ --model=/home/jro/wk/kaggle/old_nuclei/logs/nuclei20180531_1649/mask_rcnn_nuclei_3.pth
"""

import argparse
import datetime
import logging
import os
import sys

import numpy as np
import torch
from skimage.io import imread
from imgaug import augmenters as iaa

from mrcnn.utils import visualize
from mrcnn.utils.rle import mask_to_rle
from mrcnn import mrcnn_config
from mrcnn.utils import utils
from mrcnn.data.dataset_handler import DatasetHandler
from mrcnn.data.data_generator import load_image_gt
from mrcnn.models import model as modellib
from mrcnn.functions.losses import compute_map_loss
from mrcnn.functions.metrics import compute_map_metric
from tools.config import Config


logging.basicConfig(stream=sys.stderr, level=logging.INFO)

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nuclei/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = [
    "0c2550a23b8a0f29a7575de8c61690d3c31bc897dd5ba66caec201d201a278c2",
    "92f31f591929a30e4309ab75185c96ff4314ce0a7ead2ed2c2171897ad1da0c7",
    "1e488c42eb1a54a3e8412b1f12cde530f950f238d71078f2ede6a85a02168e1f",
    "c901794d1a421d52e5734500c0a2a8ca84651fb93b19cec2f411855e70cae339",
    "8e507d58f4c27cd2a82bee79fe27b069befd62a46fdaed20970a95a2ba819c7b",
    "60cb718759bff13f81c4055a7679e81326f78b6a193a2d856546097c949b20ff",
    "da5f98f2b8a64eee735a398de48ed42cd31bf17a6063db46a9e0783ac13cd844",
    "9ebcfaf2322932d464f15b5662cae4d669b2d785b8299556d73fffcae8365d32",
    "1b44d22643830cd4f23c9deadb0bd499fb392fb2cd9526d81547d93077d983df",
    "97126a9791f0c1176e4563ad679a301dac27c59011f579e808bbd6e9f4cd1034",
    "e81c758e1ca177b0942ecad62cf8d321ffc315376135bcbed3df932a6e5b40c0",
    "f29fd9c52e04403cd2c7d43b6fe2479292e53b2f61969d25256d2d2aca7c6a81",
    "0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1",
    "3ab9cab6212fabd723a2c5a1949c2ded19980398b56e6080978e796f45cbbc90",
    "ebc18868864ad075548cc1784f4f9a237bb98335f9645ee727dac8332a3e3716",
    "bb61fc17daf8bdd4e16fdcf50137a8d7762bec486ede9249d92e511fcb693676",
    "e1bcb583985325d0ef5f3ef52957d0371c96d4af767b13e48102bca9d5351a9b",
    "947c0d94c8213ac7aaa41c4efc95d854246550298259cf1bb489654d0e969050",
    "cbca32daaae36a872a11da4eaff65d1068ff3f154eedc9d3fc0c214a4e5d32bd",
    "f4c4db3df4ff0de90f44b027fc2e28c16bf7e5c75ea75b0a9762bbb7ac86e7a3",
    "4193474b2f1c72f735b13633b219d9cabdd43c21d9c2bb4dfc4809f104ba4c06",
    "f73e37957c74f554be132986f38b6f1d75339f636dfe2b681a0cf3f88d2733af",
    "a4c44fc5f5bf213e2be6091ccaed49d8bf039d78f6fbd9c4d7b7428cfcb2eda4",
    "cab4875269f44a701c5e58190a1d2f6fcb577ea79d842522dcab20ccb39b7ad2",
    "8ecdb93582b2d5270457b36651b62776256ade3aaa2d7432ae65c14f07432d49",
]


class NucleusDatasetHandler(DatasetHandler):
    """Handles nuclei dataset."""

    def __init__(self, dataset_dir, subset):
        super().__init__()
        self.load_nucleus(dataset_dir, subset)
        self.prepare()

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train",
                          "stage1_test", "stage2_test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_dir))[1]
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in image_ids:
            image_name = "images/{}.png".format(image_id)
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, image_name))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])),
                                "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = imread(os.path.join(mask_dir, f), as_gray=True).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, dataset_dir, subset):
    """Train the model."""
    dataset_train = NucleusDatasetHandler(dataset_dir, subset)
    dataset_val = NucleusDatasetHandler(dataset_dir, "val")

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train_model(dataset_train, dataset_val,
                      Config.TRAINING.LEARNING.RATE,
                      20, 'heads', augmentation=augmentation)

    print("Train all layers")
    model.train_model(dataset_train, dataset_val,
                      Config.TRAINING.LEARNING.RATE,
                      40, 'all', augmentation=augmentation)


def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDatasetHandler(dataset_dir, subset)
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image])[0]
        r = {k: v.detach().cpu().numpy() for k, v in r.items()}
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        img = visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        img.savefig("{}/{}.png".format(submit_dir,
                                       dataset.image_info[image_id]["id"]))
        plt.close()

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


def compute_metric(model, dataset_dir, subset, config):
    """Run detection on dataset and compute Kaggle's 2018 Databowl metric."""
    print(f"Running on {dataset_dir}")

    # Read dataset
    dataset_handler = NucleusDatasetHandler(dataset_dir, subset)
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


DESCR = "Train Mask R-CNN on Kaggle's Data Science Bowl 2018 dataset."


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description=DESCR)  # pylint: disable=C0103
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--dev', required=False,
                        default=0, type=int,
                        help='CUDA current device.')
    parser.add_argument('--debug', required=False,
                        type=int, help='Turn on GPU profiler.')
    parser.add_argument('--debug_function', required=False,
                        help='name of the function to be debbuged.')
    args = parser.parse_args()  # pylint: disable=C0103

    print(f"Command: {args.command}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Logs: {args.logs}")
    print(f"Debug: {args.debug}")
    print(f"Debug function: {args.debug_function}")

    if args.debug and torch.cuda.device_count() > 0:
        from tools.gpu_profile import trace_calls
        os.environ['GPU_DEBUG'] = str(args.dev)
        os.environ['TRACE_INTO'] = args.debug_function
        sys.settrace(trace_calls)
    else:
        logging.info('Not using GPU profiler.')

    # Configurations
    configs = ['./samples/nuclei_config.yml']
    if args.command == "detect":
        configs.append('./samples/nuclei_config_inference.yml')
    mrcnn_config.init_config(configs, args.dev)

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(model_dir=args.logs)

    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = Config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
    else:
        model_path = ""

    # Load weights
    print("Loading weights ", model_path)
    if args.command == "train":
        EXCLUDE = ['classifier.linear_class.weight',
                   'classifier.linear_class.bias',
                   'classifier.linear_bbox.weight',
                   'classifier.linear_bbox.bias',
                   'mask.conv5.weight',
                   'mask.conv5.bias']
        model.load_weights(model_path, exclude=EXCLUDE)
    else:
        model.load_weights(model_path)

    if torch.cuda.device_count() > 0:
        with torch.cuda.device(args.dev):
            model.to(Config.DEVICE)

    with torch.cuda.device(args.dev):
        if args.command == "train":
            train(model, args.dataset, 'train')
        elif args.command == "detect":
            detect(model, args.dataset, 'val')
        elif args.command == "metric":
            compute_metric(model, args.dataset, 'stage1_test')
        else:
            print(f"'{args.command}' is not recognized. Use 'train', 'detect'"
                  f" or 'metric'")
