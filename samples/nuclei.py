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
    python samples/nuclei.py submit --dataset=/home/jro/wk/kaggle/input/ --model=/home/jro/wk/kaggle/old_nuclei/logs/nuclei20180531_1649/mask_rcnn_nuclei_3.pth

    # Get Kaggle's 2018 Databowl metric from dataset, example:
    python samples/nuclei.py metric --dataset=/home/jro/wk/kaggle/input/ --model=/home/jro/wk/kaggle/old_nuclei/logs/nuclei20180531_1649/mask_rcnn_nuclei_3.pth
"""

import logging
import os
import sys

import torch

from mrcnn.actions.train import train
from mrcnn.actions.submit import submit
from mrcnn.config import mrcnn_config
from mrcnn.utils.mrcnn_parser import MRCNNParser
from mrcnn.utils.model_utils import load_weights
from mrcnn.models import model as modellib
from samples.nucleus_dataset_handler import NucleusDatasetHandler
from tools.config import Config


logging.basicConfig(stream=sys.stderr, level=logging.INFO)

# Root directory of the project
ROOT_DIR = os.getcwd()
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
# Results directory: Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, 'results/nuclei/')
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_DATASET_YEAR = '2014'

DESCR = "Train Mask R-CNN on Kaggle's Data Science Bowl 2018 dataset."
EXCLUDE = ['classifier.linear_class.weight',
           'classifier.linear_class.bias',
           'classifier.linear_bbox.weight',
           'classifier.linear_bbox.bias',
           'mask.conv5.weight',
           'mask.conv5.bias']

if __name__ == '__main__':
    args = MRCNNParser(DESCR, ROOT_DIR).args

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
    if args.command == 'submit':
        configs.append('./samples/nuclei_config_inference.yml')
    mrcnn_config.init_config(configs, args)

    with torch.cuda.device(Config.DEVICE_NB):
        # Create model
        model = modellib.MaskRCNN(model_dir=args.logs)

        if args.command == 'train':
            dataset_train = NucleusDatasetHandler(Config.DATASET_PATH,
                                                  'train')
            dataset_val = NucleusDatasetHandler(Config.DATASET_PATH,
                                                'val')
            # analyzer = analyze(dataset_train)
            load_weights(model, args.model, exclude=EXCLUDE)
            train(model, dataset_train, dataset_val)
        elif args.command == 'submit':
            dataset = NucleusDatasetHandler(Config.DATASET_PATH, 'stage1_test')
            # analyzer = analyze(dataset)
            load_weights(model, args.model)
            submit(model, dataset, RESULTS_DIR)
