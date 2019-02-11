"""
Mask R-CNN
Configurations and data loading code for Kaggle's Data Science Bowl 2018
dataset.

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

import logging
import os
import sys


import torch

from samples.nucleus_dataset_handler import NucleusDatasetHandler
from mrcnn.actions.compute_metric import compute_metric
from mrcnn.actions.train import train
from mrcnn.actions.detect import detect

from mrcnn.config import mrcnn_config

from mrcnn.utils.mrcnn_parser import MRCNNParser
from mrcnn.models import model as modellib

from tools.config import Config


logging.basicConfig(stream=sys.stderr, level=logging.INFO)

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs

DEFAULT_DATASET_YEAR = "2014"

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nuclei/")

DESCR = "Train Mask R-CNN on Kaggle's Data Science Bowl 2018 dataset."


if __name__ == '__main__':
    parser = MRCNNParser(DESCR, ROOT_DIR)
    args = parser.args

    if args.debug and torch.cuda.device_count() > 0:
        from tools.gpu_mem_profiling import init_profiler
        init_profiler(args.dev, args.debug_function)
        logging.info('Using GPU profiler.')
    else:
        logging.info('Not using GPU profiler.')

    if 'TIME_PROF' in os.environ:
        logging.info('Using time profiling.')

    # Configurations
    configs = ['./samples/nuclei_config.yml']
    if args.command == "detect":
        configs.append('./samples/nuclei_config_inference.yml')
    mrcnn_config.init_config(configs, args)

    # Create model
    model = modellib.MaskRCNN(model_dir=args.logs)

    # Select weights file to load
    model_path = parser.args.model

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
            dataset_train = NucleusDatasetHandler(args.dataset, 'train')
            dataset_val = NucleusDatasetHandler(args.dataset, "val")
            train(model, dataset_train, dataset_val)
        elif args.command == "detect":
            dataset = NucleusDatasetHandler(args.dataset, 'val')
            detect(model, dataset, RESULTS_DIR)
        elif args.command == "metric":
            dataset = NucleusDatasetHandler(args.dataset, 'stage1_test')
            compute_metric(model, dataset)
        else:
            print(f"'{args.command}' is not recognized. Use 'train', 'detect'"
                  f" or 'metric'")
